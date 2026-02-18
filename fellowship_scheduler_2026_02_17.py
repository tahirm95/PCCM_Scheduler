#@title Attempt 71 - PGY-4 Boards

# Colab-ready Fellowship Scheduler (OR-Tools CP-SAT)
#
# Usage in Colab:
#   !pip -q install ortools pandas
#   %run fellowship_scheduler_colab.py
#
# Adjust the configuration section as needed (dates, weights, requests).

from datetime import date, timedelta
from pathlib import Path
import time
from typing import Dict, List, Tuple
from ortools.sat.python import cp_model
import pandas as pd

# ---------------------------
# Configuration
# ---------------------------
START_DATE = date(2026, 6, 29)  # Monday
NUM_WEEKS = 52

ATS_WEEK = date(2027, 5, 17)    # Monday
CHEST_WEEK = date(2026, 10, 19) # Monday
PULM_BOARDS_WEEK = date(2026, 10, 5) # Monday

# Boundaries used across several rules
PRE_POST_BOUNDARY = date(2027, 1, 4)  # Monday (BRONCH pre/post split; pre < 1/4/27, post >= 1/4/27)
FALL_END = date(2026, 12, 28)           # Monday (Fall includes this week)
SPRING_START = date(2027, 1, 4)         # Monday (Spring starts this week)

# RRT/CCU window
RRT_CCU_START = date(2027, 1, 4)
RRT_CCU_END = date(2027, 6, 26)  # week of 6/21/27

# AIRWAY rules
# PGY-4: only during or after the week of 9/28/26
# PGY-5: any time
AIRWAY_PGY4_START = date(2026, 9, 28)

# TPLT rules
# PGY-4: only during or after the week of 11/2/26
TPLT_PGY4_START = date(2026, 11, 2)

# Vacation blackouts
PGY4_NO_VACA_BEFORE = date(2026, 7, 27)
PGY4_NO_VACA_WEEKS = {ATS_WEEK, CHEST_WEEK, date(2027, 6, 14), date(2027, 6, 21)}
PGY5_NO_VACA_EXTRA = {date(2027, 6, 14), date(2027, 6, 21)}

# Toggle for SV visa exception rule
ENABLE_SV_VISA_EXCEPTION = True

# Toggle for PGY-4 first BRONCH week pairing rule
ENFORCE_FIRST_BRONCH_PAIRING = True
# Toggle for PGY-4 max one BRONCH per week
ENFORCE_PGY4_BRONCH_MAX_ONE = True
# Toggle for max one PGY-4 on PULM (see cutoff date below)
ENFORCE_PGY4_PULM_MAX_ONE_PRE = True
# Toggle for max one PGY-4 across MICU1+MICU2 (see cutoff date below)
ENFORCE_PGY4_MISD_MAX_ONE_PRE = True
# Toggle for PGY-4 first-time MICU rule
ENFORCE_PGY4_FIRST_MISD_RULE = True
# Toggle for DD MICU1 back-to-back prevention
ENFORCE_DD_NO_CONSEC_MICU1 = True
# Toggle for PGY-6 Chiefs on RSCH during ONBD week
ENFORCE_PGY6_CHIEFS_RSCH_ONBD = True

# Toggle for AG not working with PGY-4 on PULM/MICU1/MICU2 before 12/28/26
ENFORCE_AG_NO_PGY4_EARLY = True

# Fixed weeks for PGY-4
ONBD_WEEK = date(2026, 6, 29)
PGY4_ELECT_WEEKS = {date(2026, 7, 20), CHEST_WEEK}

# SICU blackout
SICU_BLACKOUT = date(2026, 7, 6)

# Weights for soft constraints (tune as needed)
WEIGHTS = {
    "vac_rank1": 5000,
    "vac_rank2": 1500,
    "vac_rank3": 3000,
    "fair_hard_range": 200,
    "fair_micu6_range": 150,
    "hard_5_run": 300,
    "hard_3_run": 40,
    "pulm_pair_pgy4": 400,
    "micu_pair_pgy4": 400,
    "post_nf_medium": 30,
    "post_nf_hard": 60,
    "nsicu_nonconsec": 50,
    "ccu_nonconsec": 50,
    "rif_when_rads_min": 20,
    "rsch_around_vaca": 100,
    "micu_after_first_nf": 80,
    "pgy6_pre_boards_rsch": 350,
    "fair_phtn_range": 250,
    "non_sinai_pgy4_micu_early": 300,
    "rsch_exact3": 40,
}

BASE_WEIGHTS = WEIGHTS.copy()

# PGY-6 last 8 weeks preference (RSCH/VACA), weighted toward end
PGY6_LAST8_BASE = 450
PGY6_LAST8_DECAY = 30
PGY6_LAST8_CURVE = [1, 1, 1, 1, 2, 2, 3, 4]

# Non-Sinai PGY-4s (update yearly)
NON_SINAI_PGY4_NAMES = {"HD", "VK"}
# PGY-6 Chiefs (update yearly)
PGY6_CHIEF_NAMES = {"AD", "TM"}
TM_RSCH_WEEKS = {date(2026, 11, 23), date(2026, 11, 30)}
# AD must be on RSCH during these week-containing dates
AD_RSCH_WEEK_DATES = {date(2026, 11, 23), date(2026, 12, 21), date(2026, 12, 28)}

# Internal Medicine Boards blackout dates for PGY-4s.
# Fellows listed here cannot be on these rotations during their boards week.
IM_BOARDS_BLACKOUT_DATES = {
    "BX": date(2026, 8, 19),
    "JD": date(2026, 8, 18),
    "HD": date(2026, 8, 25),
    "DD": date(2026, 8, 27),
    "NM": date(2026, 11, 10),
    "VK": None,  # Not taking IM boards
}
IM_BOARDS_BLOCKED_ROTATIONS = {"PULM", "MICU1", "MICU2", "BRONCH"}

PGY_VAC_PRIORITY_MULT = {6: 3, 5: 2, 4: 1}  # higher = higher priority

# Runtime sweep (optional)
RUN_TIME_SWEEP = False
TIME_SWEEP_LIMITS_SECONDS = [240, 480, 720, 900]
TIME_SWEEP_REPEATS = 3

# Fellows
FELLOWS = [
    # PGY-4
    ("DD", 4), ("NM", 4), ("JD", 4), ("BX", 4), ("HD", 4), ("VK", 4),
    # PGY-5
    ("PH", 5), ("IT", 5), ("YA", 5), ("AZ", 5), ("SV", 5), ("AG", 5),
    # PGY-6
    ("SL", 6), ("YD", 6), ("TM", 6), ("JB", 6), ("AD", 6), ("TT", 6),
]

ROTATIONS = [
    "PULM", "ELECT", "PHYSIO", "BRONCH", "MICU1", "MICU2", "RADS", "RIF",
    "SICU", "NSICU", "RSCH", "AIRWAY", "TPLT", "RRT", "CCU", "NF", "VACA", "ONBD",
    "PHTN"
]

HARD = {"PULM", "BRONCH", "MICU1", "TPLT", "RRT", "CCU", "NF", "PHTN"}
MEDIUM = {"MICU2", "SICU", "NSICU"}
EASY = {"ELECT", "PHYSIO", "RADS", "RIF", "RSCH", "AIRWAY", "VACA", "ONBD"}

# Rotation eligibility by PGY
ALLOWED_PGY = {r: {4, 5, 6} for r in ROTATIONS}
ALLOWED_PGY["RRT"] = {5, 6}
ALLOWED_PGY["CCU"] = {6}
ALLOWED_PGY["RSCH"] = {5, 6}
ALLOWED_PGY["RADS"] = {4}
ALLOWED_PGY["RIF"] = {4}
ALLOWED_PGY["SICU"] = {4}
ALLOWED_PGY["NSICU"] = {4, 5}
ALLOWED_PGY["ELECT"] = {4}
ALLOWED_PGY["ONBD"] = {4}
ALLOWED_PGY["AIRWAY"] = {4, 5}
ALLOWED_PGY["PHTN"] = {5, 6}

# Vacation requests test data (grouped requests with ranked options)
# Expected format:
# {
#   "FELLOW": [
#     {
#       "options": [
#         {"start": date(YYYY,MM,DD), "length": 1/2/3, "rank": 1},
#         {"start": date(YYYY,MM,DD), "length": 1/2/3, "rank": 2},
#         {"start": date(YYYY,MM,DD), "length": 1/2/3, "rank": 3},
#       ]
#     },
#     ...
#   ]
# }
VACATION_REQUESTS = {
    # PGY-4 (two 2-week blocks: fall + spring, ranked options)
    "BX": [
        {"options": [
            {"start": date(2026, 9, 7), "length": 2, "rank": 1},
            {"start": date(2026, 12, 14), "length": 2, "rank": 2},
            {"start": date(2026, 10, 5), "length": 2, "rank": 3},
        ]},
        {"options": [
            {"start": date(2027, 3, 8), "length": 2, "rank": 1},
            {"start": date(2027, 3, 22), "length": 2, "rank": 2},
            {"start": date(2027, 4, 5), "length": 2, "rank": 3},
        ]},
    ],
    "JD": [
        {"options": [
            {"start": date(2026, 9, 7), "length": 2, "rank": 1},
            {"start": date(2026, 10, 5), "length": 2, "rank": 2},
            {"start": date(2026, 9, 7), "length": 2, "rank": 3},
        ]},
        {"options": [
            {"start": date(2027, 5, 3), "length": 2, "rank": 1},
            {"start": date(2027, 3, 8), "length": 2, "rank": 2},
            {"start": date(2027, 4, 5), "length": 2, "rank": 3},
        ]},
    ],
    "HD": [
        {"options": [
            {"start": date(2026, 12, 14), "length": 2, "rank": 1},
            {"start": date(2026, 8, 24), "length": 2, "rank": 2},
            {"start": date(2026, 9, 21), "length": 2, "rank": 3},
        ]},
        {"options": [
            {"start": date(2027, 2, 22), "length": 2, "rank": 1},
            {"start": date(2027, 5, 24), "length": 2, "rank": 2},
            {"start": date(2027, 4, 19), "length": 2, "rank": 3},
        ]},
    ],
    "DD": [
        {"options": [
            {"start": date(2026, 11, 30), "length": 2, "rank": 1},
            {"start": date(2026, 11, 16), "length": 2, "rank": 2},
        ]},
        {"options": [
            {"start": date(2027, 4, 19), "length": 2, "rank": 1},
            {"start": date(2027, 5, 3), "length": 2, "rank": 2},
        ]},
    ],
    "NM": [
        {"options": [
            {"start": date(2026, 12, 14), "length": 2, "rank": 1},
            {"start": date(2026, 11, 16), "length": 2, "rank": 2},
            {"start": date(2026, 11, 30), "length": 2, "rank": 3},
        ]},
        {"options": [
            {"start": date(2027, 5, 24), "length": 2, "rank": 1},
            {"start": date(2027, 5, 3), "length": 2, "rank": 2},
            {"start": date(2027, 4, 19), "length": 2, "rank": 3},
        ]},
    ],
    "VK": [
        {"options": [
            {"start": date(2026, 9, 7), "length": 2, "rank": 1},
            {"start": date(2026, 10, 5), "length": 2, "rank": 2},
            {"start": date(2026, 11, 16), "length": 2, "rank": 3},
        ]},
        {"options": [
            {"start": date(2026, 12, 28), "length": 2, "rank": 1},
            {"start": date(2027, 4, 19), "length": 2, "rank": 2},
            {"start": date(2027, 2, 8), "length": 2, "rank": 3},
        ]},
    ],

    # PGY-5/6 requests from cleaned CSV
    "AD": [
        {"options": [
            {"start": date(2026, 8, 3), "length": 2, "rank": 1},
            {"start": date(2026, 8, 3), "length": 2, "rank": 2},
            {"start": date(2026, 8, 3), "length": 2, "rank": 3},
        ]},
        {"options": [
            {"start": date(2027, 4, 26), "length": 2, "rank": 1},
            {"start": date(2027, 4, 26), "length": 2, "rank": 2},
            {"start": date(2027, 4, 26), "length": 2, "rank": 3},
        ]},
    ],
    "AG": [
        {"options": [
            {"start": date(2026, 12, 21), "length": 2, "rank": 1},
            {"start": date(2026, 12, 14), "length": 2, "rank": 2},
            {"start": date(2027, 3, 8), "length": 1, "rank": 3},
        ]},
        {"options": [
            {"start": date(2027, 3, 1), "length": 1, "rank": 1},
            {"start": date(2027, 3, 8), "length": 1, "rank": 2},
            {"start": date(2027, 2, 22), "length": 2, "rank": 3},
        ]},
        {"options": [
            {"start": date(2027, 6, 7), "length": 1, "rank": 1},
            {"start": date(2027, 6, 7), "length": 2, "rank": 2},
            {"start": date(2027, 5, 31), "length": 2, "rank": 3},
        ]},
    ],
    "AZ": [
        {"options": [
            {"start": date(2026, 8, 24), "length": 1, "rank": 1},
            {"start": date(2026, 8, 17), "length": 1, "rank": 2},
            {"start": date(2026, 8, 10), "length": 1, "rank": 3},
        ]},
        {"options": [
            {"start": date(2026, 10, 12), "length": 1, "rank": 1},
            {"start": date(2026, 10, 5), "length": 1, "rank": 2},
            {"start": date(2026, 9, 21), "length": 1, "rank": 3},
        ]},
        {"options": [
            {"start": date(2027, 1, 25), "length": 1, "rank": 1},
            {"start": date(2027, 2, 1), "length": 1, "rank": 2},
            {"start": date(2027, 1, 11), "length": 1, "rank": 3},
        ]},
        {"options": [
            {"start": date(2027, 5, 10), "length": 1, "rank": 1},
            {"start": date(2027, 5, 3), "length": 1, "rank": 2},
            {"start": date(2027, 4, 26), "length": 1, "rank": 3},
        ]},
    ],
    "IT": [
        {"options": [
            {"start": date(2026, 11, 30), "length": 2, "rank": 1},
            {"start": date(2026, 12, 7), "length": 2, "rank": 2},
        ]},
        {"options": [
            {"start": date(2027, 1, 11), "length": 2, "rank": 1},
            {"start": date(2027, 1, 18), "length": 2, "rank": 2},
            {"start": date(2027, 1, 25), "length": 2, "rank": 3},
        ]},
    ],
    "JB": [
        {"options": [
            {"start": date(2026, 11, 2), "length": 1, "rank": 1},
            {"start": date(2026, 10, 26), "length": 1, "rank": 2},
            {"start": date(2026, 11, 9), "length": 1, "rank": 3},
        ]},
        {"options": [
            {"start": date(2026, 11, 9), "length": 1, "rank": 1},
            {"start": date(2026, 10, 26), "length": 1, "rank": 2},
        ]},
        {"options": [
            {"start": date(2027, 4, 26), "length": 1, "rank": 1},
            {"start": date(2027, 4, 19), "length": 1, "rank": 2},
            {"start": date(2027, 5, 3), "length": 1, "rank": 3},
        ]},
        {"options": [
            {"start": date(2027, 5, 3), "length": 1, "rank": 1},
            {"start": date(2027, 4, 26), "length": 1, "rank": 2},
            {"start": date(2027, 4, 19), "length": 1, "rank": 3},
        ]},
    ],
    "PH": [
        {"options": [
            {"start": date(2026, 10, 5), "length": 1, "rank": 1},
            {"start": date(2026, 9, 28), "length": 1, "rank": 2},
            {"start": date(2026, 9, 21), "length": 1, "rank": 3},
        ]},
        {"options": [
            {"start": date(2026, 12, 21), "length": 1, "rank": 1},
            {"start": date(2026, 12, 28), "length": 1, "rank": 2},
            {"start": date(2026, 12, 14), "length": 1, "rank": 3},
        ]},
        {"options": [
            {"start": date(2027, 2, 1), "length": 1, "rank": 1},
            {"start": date(2027, 1, 25), "length": 1, "rank": 2},
            {"start": date(2027, 2, 8), "length": 1, "rank": 3},
        ]},
        {"options": [
            {"start": date(2027, 6, 7), "length": 1, "rank": 1},
            {"start": date(2027, 5, 31), "length": 1, "rank": 2},
            {"start": date(2027, 5, 24), "length": 1, "rank": 3},
        ]},
    ],
    "SL": [
        {"options": [
            {"start": date(2027, 5, 31), "length": 2, "rank": 1},
            {"start": date(2027, 6, 7), "length": 2, "rank": 2},
            {"start": date(2027, 6, 14), "length": 2, "rank": 3},
        ]},
        {"options": [
            {"start": date(2027, 1, 4), "length": 2, "rank": 1},
            {"start": date(2027, 1, 11), "length": 2, "rank": 2},
            {"start": date(2027, 1, 18), "length": 2, "rank": 3},
        ]},
    ],
    "SV": [
        {"options": [
            {"start": date(2027, 4, 19), "length": 3, "rank": 1},
            {"start": date(2027, 5, 24), "length": 3, "rank": 2},
            {"start": date(2027, 3, 15), "length": 3, "rank": 3},
        ]},
        {"options": [
            {"start": date(2026, 11, 9), "length": 1, "rank": 1},
            {"start": date(2026, 11, 16), "length": 1, "rank": 2},
            {"start": date(2026, 11, 23), "length": 1, "rank": 3},
        ]},
    ],
    "TM": [
        {"options": [
            {"start": date(2026, 8, 10), "length": 1, "rank": 1},
            {"start": date(2026, 8, 3), "length": 1, "rank": 2},
            {"start": date(2026, 11, 23), "length": 1, "rank": 3},
        ]},
        {"options": [
            {"start": date(2026, 12, 28), "length": 2, "rank": 1},
            {"start": date(2026, 12, 21), "length": 2, "rank": 2},
            {"start": date(2027, 1, 4), "length": 2, "rank": 3},
        ]},
        {"options": [
            {"start": date(2027, 6, 7), "length": 1, "rank": 1},
            {"start": date(2027, 3, 15), "length": 1, "rank": 2},
            {"start": date(2027, 6, 21), "length": 1, "rank": 3},
        ]},
    ],
    "TT": [
        {"options": [
            {"start": date(2027, 5, 24), "length": 2, "rank": 1},
            {"start": date(2027, 5, 24), "length": 2, "rank": 2},
            {"start": date(2027, 5, 24), "length": 2, "rank": 3},
        ]},
        {"options": [
            {"start": date(2026, 12, 28), "length": 1, "rank": 1},
            {"start": date(2026, 10, 12), "length": 1, "rank": 2},
        ]},
        {"options": [
            {"start": date(2026, 10, 12), "length": 1, "rank": 1},
            {"start": date(2026, 12, 28), "length": 1, "rank": 2},
        ]},
    ],
    "YA": [
        {"options": [
            {"start": date(2026, 11, 23), "length": 1, "rank": 1},
            {"start": date(2026, 11, 30), "length": 1, "rank": 2},
            {"start": date(2026, 12, 7), "length": 1, "rank": 3},
        ]},
        {"options": [
            {"start": date(2026, 9, 21), "length": 1, "rank": 1},
            {"start": date(2026, 9, 28), "length": 1, "rank": 2},
            {"start": date(2026, 9, 14), "length": 1, "rank": 3},
        ]},
        {"options": [
            {"start": date(2027, 2, 22), "length": 1, "rank": 1},
            {"start": date(2027, 2, 15), "length": 1, "rank": 2},
            {"start": date(2027, 3, 1), "length": 1, "rank": 3},
        ]},
        {"options": [
            {"start": date(2027, 3, 1), "length": 1, "rank": 1},
            {"start": date(2027, 4, 19), "length": 1, "rank": 2},
            {"start": date(2027, 5, 10), "length": 1, "rank": 3},
        ]},
    ],
    "YD": [
        {"options": [
            {"start": date(2026, 9, 21), "length": 1, "rank": 1},
            {"start": date(2026, 9, 28), "length": 1, "rank": 2},
            {"start": date(2026, 9, 14), "length": 1, "rank": 3},
        ]},
        {"options": [
            {"start": date(2026, 12, 21), "length": 1, "rank": 1},
            {"start": date(2026, 12, 28), "length": 1, "rank": 2},
            {"start": date(2026, 11, 23), "length": 1, "rank": 3},
        ]},
        {"options": [
            {"start": date(2027, 1, 25), "length": 1, "rank": 1},
            {"start": date(2027, 1, 18), "length": 1, "rank": 2},
            {"start": date(2027, 2, 1), "length": 1, "rank": 3},
        ]},
        {"options": [
            {"start": date(2027, 5, 24), "length": 1, "rank": 1},
            {"start": date(2027, 6, 21), "length": 1, "rank": 2},
            {"start": date(2027, 6, 14), "length": 1, "rank": 3},
        ]},
    ],

}

# ---------------------------
# Helper data
# ---------------------------
WEEK_STARTS = [START_DATE + timedelta(weeks=i) for i in range(NUM_WEEKS)]
WEEK_INDEX = {d: i for i, d in enumerate(WEEK_STARTS)}

PGY4 = [i for i, (_, pgy) in enumerate(FELLOWS) if pgy == 4]
PGY5 = [i for i, (_, pgy) in enumerate(FELLOWS) if pgy == 5]
PGY6 = [i for i, (_, pgy) in enumerate(FELLOWS) if pgy == 6]
DD_IDX = next(i for i, (name, _) in enumerate(FELLOWS) if name == "DD")
NON_SINAI_PGY4 = [i for i, (name, pgy) in enumerate(FELLOWS) if pgy == 4 and name in NON_SINAI_PGY4_NAMES]
PGY6_CHIEFS = [i for i, (name, pgy) in enumerate(FELLOWS) if pgy == 6 and name in PGY6_CHIEF_NAMES]
TM_IDX = next(i for i, (name, _) in enumerate(FELLOWS) if name == "TM")
YA_IDX = next(i for i, (name, _) in enumerate(FELLOWS) if name == "YA")

ROT_IDX = {r: i for i, r in enumerate(ROTATIONS)}


def widx(d: date) -> int:
    if d not in WEEK_INDEX:
        raise ValueError(f"Date {d} is not a Monday in the schedule.")
    return WEEK_INDEX[d]


def weeks_in_range(start_inclusive: date, end_inclusive: date) -> List[int]:
    return [i for i, d in enumerate(WEEK_STARTS) if start_inclusive <= d <= end_inclusive]


def weeks_before(d: date) -> List[int]:
    return [i for i, dt in enumerate(WEEK_STARTS) if dt < d]


def weeks_on_or_after(d: date) -> List[int]:
    return [i for i, dt in enumerate(WEEK_STARTS) if dt >= d]


def week_start_for_date(d: date) -> date:
    """Return Monday week-start for any date."""
    return d - timedelta(days=d.weekday())


# Pre/post ranges for BRONCH
PRE_WEEKS = weeks_before(PRE_POST_BOUNDARY)
POST_WEEKS = weeks_on_or_after(PRE_POST_BOUNDARY)

# Fall/Spring ranges for vacations
FALL_WEEKS = [i for i, d in enumerate(WEEK_STARTS) if d <= FALL_END]
SPRING_WEEKS = [i for i, d in enumerate(WEEK_STARTS) if d >= SPRING_START]

# ATS and CHEST week indices
ATS_WEEK_IDX = widx(ATS_WEEK)
CHEST_WEEK_IDX = widx(CHEST_WEEK)
PULM_BOARDS_WEEK_IDX = widx(PULM_BOARDS_WEEK)
ONBD_WEEK_IDX = widx(ONBD_WEEK)
AD_RSCH_WEEKS = [widx(d) for d in sorted(AD_RSCH_WEEK_DATES)]
IM_BOARDS_BLACKOUT_WEEKS = {
    name: widx(week_start_for_date(d))
    for name, d in IM_BOARDS_BLACKOUT_DATES.items()
    if d is not None
}

# RRT/CCU window indices
RRT_CCU_WEEKS = weeks_in_range(RRT_CCU_START, date(2027, 6, 21))

# AIRWAY allowed weeks (PGY-4 only)
AIRWAY_PGY4_WEEKS = weeks_on_or_after(AIRWAY_PGY4_START)
# TPLT allowed weeks (PGY-4 only)
TPLT_PGY4_WEEKS = weeks_on_or_after(TPLT_PGY4_START)

# Pre-1/4/27 weeks (for MICU PGY-4 constraint)
PRE_MISD_WEEKS = [i for i, d in enumerate(WEEK_STARTS) if d < date(2027, 1, 4)]

# PGY-4 max-one constraints cutoff dates (inclusive of all weeks before the cutoff)
# Set these later to relax the constraint windows.
PGY4_PULM_MAX_ONE_CUTOFF = date(2027, 2, 8)
PGY4_MISD_MAX_ONE_CUTOFF = date(2027, 2, 8)
PGY4_PULM_MAX_ONE_WEEKS = weeks_before(PGY4_PULM_MAX_ONE_CUTOFF)
PGY4_MISD_MAX_ONE_WEEKS = weeks_before(PGY4_MISD_MAX_ONE_CUTOFF)

# AG restriction cutoff (weeks strictly before 12/28/26)
AG_NO_PGY4_CUTOFF = date(2026, 12, 28)
AG_NO_PGY4_WEEKS = weeks_before(AG_NO_PGY4_CUTOFF)

# July blackout weeks (PGY-5/6 vacation rule)
JULY_WEEKS = [i for i, d in enumerate(WEEK_STARTS) if d.month == 7 and d.year == 2026]

# ---------------------------
# Model
# ---------------------------

def build_model():
    model = cp_model.CpModel()
    
    F = len(FELLOWS)
    W = NUM_WEEKS
    R = len(ROTATIONS)
    
    # Decision vars: x[f, w, r]
    x = {}
    for f in range(F):
        for w in range(W):
            for r in range(R):
                x[(f, w, r)] = model.NewBoolVar(f"x_f{f}_w{w}_r{r}")
    
    
    def var(f: int, w: int, rname: str):
        return x[(f, w, ROT_IDX[rname])]
    
    
    # Exactly one rotation per fellow per week
    for f in range(F):
        for w in range(W):
            model.Add(sum(x[(f, w, r)] for r in range(R)) == 1)
    
    # Eligibility constraints
    for f, (_, pgy) in enumerate(FELLOWS):
        for w in range(W):
            for rname in ROTATIONS:
                if pgy not in ALLOWED_PGY[rname]:
                    model.Add(var(f, w, rname) == 0)
    
    # Max one PGY-4 across MICU1+MICU2 (within cutoff window)
    if ENFORCE_PGY4_MISD_MAX_ONE_PRE:
        for w in PGY4_MISD_MAX_ONE_WEEKS:
            model.Add(
                sum(var(f, w, "MICU1") for f in PGY4) +
                sum(var(f, w, "MICU2") for f in PGY4)
                <= 1
            )
    
    # Max one PGY-4 on PULM (within cutoff window)
    if ENFORCE_PGY4_PULM_MAX_ONE_PRE:
        for w in PGY4_PULM_MAX_ONE_WEEKS:
            model.Add(sum(var(f, w, "PULM") for f in PGY4) <= 1)

    # AG constraints before 12/28/26
    # - If a PGY-4 is on PULM, AG cannot be on PULM
    # - If a PGY-4 is on MICU1, AG cannot be on MICU2
    # - If a PGY-4 is on MICU2, AG cannot be on MICU1
    if ENFORCE_AG_NO_PGY4_EARLY:
        ag_idx = next(i for i, (name, _) in enumerate(FELLOWS) if name == "AG")
        for w in AG_NO_PGY4_WEEKS:
            # PULM same-rotation exclusion
            pg4_on_pulm = model.NewBoolVar(f"ag_pg4_pulm_w{w}")
            sum_pg4_pulm = sum(var(f, w, "PULM") for f in PGY4)
            model.Add(sum_pg4_pulm >= 1).OnlyEnforceIf(pg4_on_pulm)
            model.Add(sum_pg4_pulm == 0).OnlyEnforceIf(pg4_on_pulm.Not())
            model.Add(var(ag_idx, w, "PULM") == 0).OnlyEnforceIf(pg4_on_pulm)

            # MICU cross-rotation exclusion
            pg4_on_micu1 = model.NewBoolVar(f"ag_pg4_micu1_w{w}")
            sum_pg4_micu1 = sum(var(f, w, "MICU1") for f in PGY4)
            model.Add(sum_pg4_micu1 >= 1).OnlyEnforceIf(pg4_on_micu1)
            model.Add(sum_pg4_micu1 == 0).OnlyEnforceIf(pg4_on_micu1.Not())
            model.Add(var(ag_idx, w, "MICU2") == 0).OnlyEnforceIf(pg4_on_micu1)

            pg4_on_micu2 = model.NewBoolVar(f"ag_pg4_micu2_w{w}")
            sum_pg4_micu2 = sum(var(f, w, "MICU2") for f in PGY4)
            model.Add(sum_pg4_micu2 >= 1).OnlyEnforceIf(pg4_on_micu2)
            model.Add(sum_pg4_micu2 == 0).OnlyEnforceIf(pg4_on_micu2.Not())
            model.Add(var(ag_idx, w, "MICU1") == 0).OnlyEnforceIf(pg4_on_micu2)
    
    # Date-based disallowances
    for f in range(F):
        # PHYSIO is off during ATS week
        model.Add(var(f, ATS_WEEK_IDX, "PHYSIO") == 0)
    
        # ONBD only in the onboarding week
        for w in range(W):
            if WEEK_STARTS[w] != ONBD_WEEK:
                model.Add(var(f, w, "ONBD") == 0)
    
        # AIRWAY for PGY-4 only after 9/28/26; PGY-5 anytime
        if FELLOWS[f][1] == 4:
            for w in range(W):
                if w not in AIRWAY_PGY4_WEEKS:
                    model.Add(var(f, w, "AIRWAY") == 0)

                if w not in TPLT_PGY4_WEEKS:
                    model.Add(var(f, w, "TPLT") == 0)

            # Internal Medicine boards blackout: PGY-4 cannot be on specified
            # high-acuity rotations during their boards week.
            fellow_name = FELLOWS[f][0]
            if fellow_name in IM_BOARDS_BLACKOUT_WEEKS:
                bw = IM_BOARDS_BLACKOUT_WEEKS[fellow_name]
                for rname in IM_BOARDS_BLOCKED_ROTATIONS:
                    model.Add(var(f, bw, rname) == 0)
    
        # RRT/CCU only in window
        for w in range(W):
            if w not in RRT_CCU_WEEKS:
                model.Add(var(f, w, "RRT") == 0)
                model.Add(var(f, w, "CCU") == 0)
    
    # Fixed PGY-4 weeks
    for f in PGY4:
        model.Add(var(f, ONBD_WEEK_IDX, "ONBD") == 1)

    # PGY-6 Chiefs on RSCH during ONBD week
    if ENFORCE_PGY6_CHIEFS_RSCH_ONBD:
        for f in PGY6_CHIEFS:
            model.Add(var(f, ONBD_WEEK_IDX, "RSCH") == 1)

    # TM must be on RSCH for specified weeks
    for d in TM_RSCH_WEEKS:
        model.Add(var(TM_IDX, widx(d), "RSCH") == 1)

    # AD must be on RSCH for specified weeks
    ad_idx = next(i for i, (name, _) in enumerate(FELLOWS) if name == "AD")
    for w in AD_RSCH_WEEKS:
        model.Add(var(ad_idx, w, "RSCH") == 1)

    # AD must have exactly 2 BRONCH weeks
    model.Add(sum(var(ad_idx, w, "BRONCH") for w in range(W)) == 2)

    # YA must be on MICU2 on the first week (6/29/26)
    model.Add(var(YA_IDX, ONBD_WEEK_IDX, "MICU2") == 1)

    # TM must have exactly 2 weeks of MICU1 and 2 weeks of MICU2
    model.Add(sum(var(TM_IDX, w, "MICU1") for w in range(W)) == 2)
    model.Add(sum(var(TM_IDX, w, "MICU2") for w in range(W)) == 2)
    # TM must have exactly 4 weeks of NF
    model.Add(sum(var(TM_IDX, w, "NF") for w in range(W)) == 4)
    
    for d in PGY4_ELECT_WEEKS:
        w = widx(d)
        for f in PGY4:
            model.Add(var(f, w, "ELECT") == 1)
    
    # Pulmonary Boards week: all PGY-6 on RSCH
    for f in PGY6:
        model.Add(var(f, PULM_BOARDS_WEEK_IDX, "RSCH") == 1)
    
    # ATS week: all PGY-5 and PGY-6 on RSCH
    for f in PGY5 + PGY6:
        model.Add(var(f, ATS_WEEK_IDX, "RSCH") == 1)
    
    
    # Weekly coverage demands
    for w in range(W):
        model.Add(sum(var(f, w, "PULM") for f in range(F)) == 2)
        model.Add(sum(var(f, w, "MICU1") for f in range(F)) == 1)
        model.Add(sum(var(f, w, "MICU2") for f in range(F)) == 1)
        model.Add(sum(var(f, w, "TPLT") for f in range(F)) == 1)

        physio_demand = 0 if w == ATS_WEEK_IDX else 1
        model.Add(sum(var(f, w, "PHYSIO") for f in range(F)) == physio_demand)

        # PHTN coverage: 1 fellow per week, except ATS week
        phtn_demand = 0 if w == ATS_WEEK_IDX else 1
        model.Add(sum(var(f, w, "PHTN") for f in range(F)) == phtn_demand)

        # Research coverage: at least 3 fellows per week
        model.Add(sum(var(f, w, "RSCH") for f in range(F)) >= 3)

        bronch_count = sum(var(f, w, "BRONCH") for f in range(F))
        model.Add(bronch_count >= 1)
        if ENFORCE_PGY4_BRONCH_MAX_ONE:
            model.Add(sum(var(f, w, "BRONCH") for f in PGY4) <= 1)
        model.Add(bronch_count <= 2)

        # Singleton weekly coverage rules
        model.Add(sum(var(f, w, "RRT") for f in range(F)) <= 1)
        model.Add(sum(var(f, w, "CCU") for f in range(F)) <= 1)
        model.Add(sum(var(f, w, "NF") for f in range(F)) <= 1)
        model.Add(sum(var(f, w, "AIRWAY") for f in range(F)) <= 1)
        model.Add(sum(var(f, w, "NSICU") for f in range(F)) <= 1)
        model.Add(sum(var(f, w, "SICU") for f in range(F)) <= 1)
        model.Add(sum(var(f, w, "RIF") for f in range(F)) <= 1)
        model.Add(sum(var(f, w, "RADS") for f in range(F)) <= 1)
    
    # ---------------------------
    # Per-fellow rotation counts
    # ---------------------------
    
    def add_min_max(f: int, rname: str, min_weeks=None, max_weeks=None):
        total = sum(var(f, w, rname) for w in range(W))
        if min_weeks is not None:
            model.Add(total >= min_weeks)
        if max_weeks is not None:
            model.Add(total <= max_weeks)
        return total
    
    rotation_totals: Dict[Tuple[int, str], cp_model.IntVar] = {}
    
    for f, (_, pgy) in enumerate(FELLOWS):
        if pgy == 4:
            rotation_totals[(f, "PULM")] = add_min_max(f, "PULM", 7, 7)
            rotation_totals[(f, "MICU1")] = add_min_max(f, "MICU1", 5, 5)
            rotation_totals[(f, "MICU2")] = add_min_max(f, "MICU2", 5, 5)
            rotation_totals[(f, "TPLT")] = add_min_max(f, "TPLT", 3, 4)
            rotation_totals[(f, "PHYSIO")] = add_min_max(f, "PHYSIO", 5, 6)
            rotation_totals[(f, "NSICU")] = add_min_max(f, "NSICU", 2, 2)
            rotation_totals[(f, "AIRWAY")] = add_min_max(f, "AIRWAY", 1, 1)
            rotation_totals[(f, "RADS")] = add_min_max(f, "RADS", 2, 3)
            rotation_totals[(f, "NF")] = add_min_max(f, "NF", 2, 2)
            rotation_totals[(f, "VACA")] = add_min_max(f, "VACA", 4, 4)
            rotation_totals[(f, "BRONCH")] = add_min_max(f, "BRONCH", 6, 6)
            rotation_totals[(f, "ELECT")] = add_min_max(f, "ELECT", 2, 2)
            rotation_totals[(f, "SICU")] = add_min_max(f, "SICU", 2, 2)
            rotation_totals[(f, "ONBD")] = add_min_max(f, "ONBD", 1, 1)
            rotation_totals[(f, "RIF")] = add_min_max(f, "RIF", 4, 5)
        elif pgy == 5:
            rotation_totals[(f, "PULM")] = add_min_max(f, "PULM", 7, 7)
            rotation_totals[(f, "MICU1")] = add_min_max(f, "MICU1", 2, 2)
            rotation_totals[(f, "MICU2")] = add_min_max(f, "MICU2", 2, 2)
            rotation_totals[(f, "TPLT")] = add_min_max(f, "TPLT", 4, 5)
            rotation_totals[(f, "PHYSIO")] = add_min_max(f, "PHYSIO", 1, 3)
            rotation_totals[(f, "NSICU")] = add_min_max(f, "NSICU", 2, 2)
            rotation_totals[(f, "AIRWAY")] = add_min_max(f, "AIRWAY", 1, 1)
            rotation_totals[(f, "RRT")] = add_min_max(f, "RRT", 1, 1)
            rotation_totals[(f, "RSCH")] = add_min_max(f, "RSCH", 17, 23)
            rotation_totals[(f, "NF")] = add_min_max(f, "NF", 2, 2)
            rotation_totals[(f, "VACA")] = add_min_max(f, "VACA", 4, 4)
            rotation_totals[(f, "BRONCH")] = add_min_max(f, "BRONCH", 1, 3)
            rotation_totals[(f, "PHTN")] = add_min_max(f, "PHTN", 3, 5)
        else:  # pgy == 6
            rotation_totals[(f, "PULM")] = add_min_max(f, "PULM", 0, 4)
            rotation_totals[(f, "MICU1")] = add_min_max(f, "MICU1", 1, 2)
            rotation_totals[(f, "MICU2")] = add_min_max(f, "MICU2", 1, 2)
            rotation_totals[(f, "TPLT")] = add_min_max(f, "TPLT", 1, 2)
            rotation_totals[(f, "PHYSIO")] = add_min_max(f, "PHYSIO", 0, 2)
            rotation_totals[(f, "RRT")] = add_min_max(f, "RRT", 1, 1)
            rotation_totals[(f, "CCU")] = add_min_max(f, "CCU", 2, 2)
            rotation_totals[(f, "RSCH")] = add_min_max(f, "RSCH", 26, 33)
            if f == TM_IDX:
                rotation_totals[(f, "NF")] = add_min_max(f, "NF", 4, 4)
            else:
                rotation_totals[(f, "NF")] = add_min_max(f, "NF", 2, 2)
            rotation_totals[(f, "VACA")] = add_min_max(f, "VACA", 4, 4)
            rotation_totals[(f, "BRONCH")] = add_min_max(f, "BRONCH", 1, 3)
            rotation_totals[(f, "PHTN")] = add_min_max(f, "PHTN", 3, 5)
    
    # PGY-6 MICU coupling (total must be 3 or 4)
    for f in PGY6:
        micu_total = sum(var(f, w, "MICU1") + var(f, w, "MICU2") for w in range(W))
        model.Add(micu_total >= 3)
    
    # ---------------------------
    # Sequence constraints
    # ---------------------------
    # No >2 consecutive PULM
    for f in range(F):
        for w in range(W - 2):
            model.Add(var(f, w, "PULM") + var(f, w + 1, "PULM") + var(f, w + 2, "PULM") <= 2)
    
    # No >2 consecutive MICU (MICU1+MICU2)
    for f in range(F):
        for w in range(W - 2):
            model.Add(
                var(f, w, "MICU1") + var(f, w, "MICU2") +
                var(f, w + 1, "MICU1") + var(f, w + 1, "MICU2") +
                var(f, w + 2, "MICU1") + var(f, w + 2, "MICU2") <= 2
            )
    
    # No consecutive RADS or RIF
    for f in range(F):
        for w in range(W - 1):
            model.Add(var(f, w, "RADS") + var(f, w + 1, "RADS") <= 1)
            model.Add(var(f, w, "RIF") + var(f, w + 1, "RIF") <= 1)

    # No consecutive NF
    for f in range(F):
        for w in range(W - 1):
            model.Add(var(f, w, "NF") + var(f, w + 1, "NF") <= 1)

    # DD: no back-to-back MICU1 weeks
    if ENFORCE_DD_NO_CONSEC_MICU1:
        for w in range(W - 1):
            model.Add(var(DD_IDX, w, "MICU1") + var(DD_IDX, w + 1, "MICU1") <= 1)

    # PGY-6: last 2 weeks must be EASY rotations
    for f in PGY6:
        for w in range(W - 2, W):
            model.Add(sum(var(f, w, r) for r in EASY) == 1)
    
    objective_terms = []
    
    # ---------------------------
    # Soft constraint reporting (collect vars for post-solve report)
    # ---------------------------
    soft_report = {
        "vacation": [],
        "pulm_pair": [],
        "micu_pair": [],
        "hard3": [],
        "hard5": [],
        "post_nf_medium": [],
        "post_nf_hard": [],
        "nsicu_nonconsec": [],
        "ccu_nonconsec": [],
        "rif_when_rads_min": [],
        "rsch_around_vaca": [],
        "micu_after_first_nf": [],
        "pgy6_pre_boards_rsch": [],
        "pgy6_last8_rsch_vaca": [],
        "non_sinai_pgy4_micu_early": [],
        "rsch_exact3": [],
        "fair_hard": [],
        "fair_micu6": [],
        "fair_phtn": [],
    }

    # Prefer RSCH coverage > 3 by penalizing weeks with exactly 3 RSCH.
    for w in range(W):
        rsch_count = sum(var(f, w, "RSCH") for f in range(F))
        rsch_exact3 = model.NewBoolVar(f"rsch_exact3_w{w}")
        model.Add(rsch_count == 3).OnlyEnforceIf(rsch_exact3)
        model.Add(rsch_count != 3).OnlyEnforceIf(rsch_exact3.Not())
        objective_terms.append(WEIGHTS["rsch_exact3"] * rsch_exact3)
        soft_report["rsch_exact3"].append((w, rsch_exact3))

    # Non-Sinai PGY-4s: penalize MICU1/2 in the first 7 weeks of fellowship
    early_micu_weeks = list(range(min(7, W)))
    early_micu_len = len(early_micu_weeks)
    early_micu_base = WEIGHTS["non_sinai_pgy4_micu_early"]
    if early_micu_len <= 1:
        early_micu_weights = [early_micu_base] * early_micu_len
    else:
        early_micu_weights = [
            int(round(early_micu_base * (early_micu_len - i) / early_micu_len))
            for i in range(early_micu_len)
        ]
    early_micu_weight_by_week = dict(zip(early_micu_weeks, early_micu_weights))

    for f in NON_SINAI_PGY4:
        for w in early_micu_weeks:
            micu = var(f, w, "MICU1") + var(f, w, "MICU2")
            weight = early_micu_weight_by_week[w]
            objective_terms.append(weight * micu)
            soft_report["non_sinai_pgy4_micu_early"].append((f, w, micu, weight))
    
    # ---------------------------
    # PGY-4 BRONCH structure and double-staffing
    # ---------------------------
    # Track PGY-4 first BRONCH week flags by week
    first_bronch_by_week = {w: [] for w in range(W)}
    
    # First BRONCH week for each PGY-4 must be double-staffed with a PGY-5/6
    if ENFORCE_FIRST_BRONCH_PAIRING:
        for f in PGY4:
            is_first_weeks = []
            for w in range(W):
                b = model.NewBoolVar(f"pgy4_first_bronch_f{f}_w{w}")
                is_first_weeks.append(b)
                first_bronch_by_week[w].append(b)
    
                # b => this week is BRONCH
                model.Add(b <= var(f, w, "BRONCH"))
    
                # b => no prior BRONCH weeks
                if w > 0:
                    prev = [var(f, wp, "BRONCH") for wp in range(w)]
                    model.Add(sum(prev) == 0).OnlyEnforceIf(b)
    
                # If this is the first BRONCH week, require double-staffing with PGY-5/6
                bronch_count = sum(var(ff, w, "BRONCH") for ff in range(F))
                pg56_bronch = sum(var(ff, w, "BRONCH") for ff in PGY5 + PGY6)
                model.Add(bronch_count == 2).OnlyEnforceIf(b)
                model.Add(pg56_bronch == 1).OnlyEnforceIf(b)
    
            # Each PGY-4 must have exactly one first BRONCH week
            model.Add(sum(is_first_weeks) == 1)
    
    # Only allow 2 BRONCH fellows when it is a PGY-4 first BRONCH week
    for w in range(W):
        if first_bronch_by_week[w]:
            model.Add(sum(first_bronch_by_week[w]) <= 1)
            bronch_count = sum(var(f, w, "BRONCH") for f in range(F))
            model.Add(bronch_count <= 1 + sum(first_bronch_by_week[w]))
    
    # Pre/post totals and post consecutive block
    # Rule update:
    # - All PGY-4s must have exactly 4 BRONCH weeks before 12/28/26.
    for f in PGY4:
        pre_total = sum(var(f, w, "BRONCH") for w in PRE_WEEKS)
        post_total = sum(var(f, w, "BRONCH") for w in POST_WEEKS)
    
        # Pre-BRONCH requirement: exactly 4 weeks
        model.Add(pre_total == 4)
    
        # Post-BRONCH unchanged
        model.Add(post_total == 2)
    
        # Post block (2 consecutive weeks)
        post_block_starts = []
        for w in POST_WEEKS:
            if w + 1 < W and (w + 1) in POST_WEEKS:
                b = model.NewBoolVar(f"post_bronch_start_f{f}_w{w}")
                post_block_starts.append((w, b))
                model.Add(var(f, w, "BRONCH") == 1).OnlyEnforceIf(b)
                model.Add(var(f, w + 1, "BRONCH") == 1).OnlyEnforceIf(b)
        model.Add(sum(b for _, b in post_block_starts) == 1)
    
        # Ensure BRONCH only in the chosen post block
        for w in POST_WEEKS:
            covering = [b for sw, b in post_block_starts if sw == w or sw + 1 == w]
            if covering:
                model.Add(var(f, w, "BRONCH") <= sum(covering))
    
    # At least 5 of 6 PGY-4s must have a 4-week consecutive BRONCH streak pre-12/28/26
    pre_streak_flags = []
    pre_bronch4_starts = {f: [] for f in PGY4}
    for f in PGY4:
        streak_vars = []
        for w in PRE_WEEKS:
            if w + 3 < W and (w + 3) in PRE_WEEKS:
                s = model.NewBoolVar(f"pre_bronch4_f{f}_w{w}")
                bronch_sum = sum(var(f, w + k, "BRONCH") for k in range(4))
                model.Add(bronch_sum >= 4).OnlyEnforceIf(s)
                model.Add(bronch_sum <= 3).OnlyEnforceIf(s.Not())
                streak_vars.append(s)
                pre_bronch4_starts[f].append((w, s))
        has_streak = model.NewBoolVar(f"pre_bronch4_exists_f{f}")
        if streak_vars:
            model.AddMaxEquality(has_streak, streak_vars)
        else:
            model.Add(has_streak == 0)
        pre_streak_flags.append(has_streak)
    
    model.Add(sum(pre_streak_flags) >= 5)
    
    # The remaining PGY-4 (without a 4-week pre-streak) must have at least two
    # 2-week consecutive BRONCH blocks before 12/28/26.
    for idx, f in enumerate(PGY4):
        pre_pair_starts = []
        for w in PRE_WEEKS:
            if w + 1 < W and (w + 1) in PRE_WEEKS:
                b = model.NewBoolVar(f"pgy4_pre_bronch_pair_f{f}_w{w}")
                pre_pair_starts.append(b)
                model.Add(var(f, w, "BRONCH") == 1).OnlyEnforceIf(b)
                model.Add(var(f, w + 1, "BRONCH") == 1).OnlyEnforceIf(b)
        if pre_pair_starts:
            model.Add(sum(pre_pair_starts) >= 2).OnlyEnforceIf(pre_streak_flags[idx].Not())
    
    # PGY-4 NF hard rule: must have completed at least one MICU1 or MICU2 before any NF
    for f in PGY4:
        for w in range(W):
            if w == 0:
                model.Add(var(f, w, "NF") == 0)
                continue
            prior_any_micu = sum(
                var(f, wp, "MICU1") + var(f, wp, "MICU2") for wp in range(w)
            )
            model.Add(prior_any_micu >= 1).OnlyEnforceIf(var(f, w, "NF"))

    # PGY-4 MICU ordering hard rule: MICU1 must occur before any MICU2.
    for f in PGY4:
        for w in range(W):
            if w == 0:
                model.Add(var(f, w, "MICU2") == 0)
                continue
            prior_micu1 = sum(var(f, wp, "MICU1") for wp in range(w))
            model.Add(prior_micu1 >= 1).OnlyEnforceIf(var(f, w, "MICU2"))
    
    # PGY-4 MICU first-time rule: at most one PGY-4 can have their first MICU week
    # (MICU1 or MICU2) in the same week.
    if ENFORCE_PGY4_FIRST_MISD_RULE:
        for w in range(W):
            first_micu_flags = []
            for f in PGY4:
                # has done any MICU (1 or 2) before week w
                prior_any_micu = sum(
                    var(f, wp, "MICU1") + var(f, wp, "MICU2") for wp in range(w)
                )
                has_prior = model.NewBoolVar(f"pgy4_has_prior_micu_f{f}_w{w}")
                has_prior_not = model.NewBoolVar(f"pgy4_has_prior_micu_not_f{f}_w{w}")
                model.Add(has_prior_not == 1 - has_prior)
                model.Add(prior_any_micu >= 1).OnlyEnforceIf(has_prior)
                model.Add(prior_any_micu == 0).OnlyEnforceIf(has_prior.Not())

                on_micu = model.NewBoolVar(f"pgy4_on_micu_f{f}_w{w}")
                model.Add(on_micu <= var(f, w, "MICU1") + var(f, w, "MICU2"))
                model.Add(on_micu >= var(f, w, "MICU1"))
                model.Add(on_micu >= var(f, w, "MICU2"))

                is_first_micu = model.NewBoolVar(f"pgy4_first_micu_f{f}_w{w}")
                model.Add(is_first_micu <= on_micu)
                model.Add(is_first_micu <= has_prior_not)
                model.Add(is_first_micu >= on_micu + has_prior_not - 1)
                first_micu_flags.append(is_first_micu)

            model.Add(sum(first_micu_flags) <= 1)
    
    # ---------------------------
    # SICU consecutive block (PGY-4)
    # ---------------------------
    for f in PGY4:
        block_starts = []
        for w in range(W - 1):
            if WEEK_STARTS[w] == SICU_BLACKOUT or WEEK_STARTS[w + 1] == SICU_BLACKOUT:
                continue
            b = model.NewBoolVar(f"sicu_start_f{f}_w{w}")
            block_starts.append((w, b))
            model.Add(var(f, w, "SICU") == 1).OnlyEnforceIf(b)
            model.Add(var(f, w + 1, "SICU") == 1).OnlyEnforceIf(b)
    
        model.Add(sum(b for _, b in block_starts) == 1)
    
        # Ensure SICU only occurs in the chosen block
        for w in range(W):
            covering = [b for sw, b in block_starts if sw == w or sw + 1 == w]
            if covering:
                model.Add(var(f, w, "SICU") <= sum(covering))
    
    # ---------------------------
    # Vacation constraints
    # ---------------------------
    # PGY-4: two 2-week blocks (fall + spring)
    for f in PGY4:
        # Disallow PGY-4 vacation in blackouts
        for w in range(W):
            if WEEK_STARTS[w] < PGY4_NO_VACA_BEFORE or WEEK_STARTS[w] in PGY4_NO_VACA_WEEKS:
                model.Add(var(f, w, "VACA") == 0)
    
        fall_candidates = []
        spring_candidates = []
        for w in range(W - 1):
            w1 = WEEK_STARTS[w]
            w2 = WEEK_STARTS[w + 1]
            if w1 < PGY4_NO_VACA_BEFORE:
                continue
            if w1 in PGY4_NO_VACA_WEEKS or w2 in PGY4_NO_VACA_WEEKS:
                continue
            if w1 <= FALL_END and w2 <= FALL_END:
                fall_candidates.append(w)
            if w1 >= SPRING_START and w2 >= SPRING_START:
                spring_candidates.append(w)
    
        fall_starts = []
        for w in fall_candidates:
            b = model.NewBoolVar(f"pgy4_fall_vaca_start_f{f}_w{w}")
            fall_starts.append((w, b))
            model.Add(var(f, w, "VACA") == 1).OnlyEnforceIf(b)
            model.Add(var(f, w + 1, "VACA") == 1).OnlyEnforceIf(b)
    
        spring_starts = []
        for w in spring_candidates:
            b = model.NewBoolVar(f"pgy4_spring_vaca_start_f{f}_w{w}")
            spring_starts.append((w, b))
            model.Add(var(f, w, "VACA") == 1).OnlyEnforceIf(b)
            model.Add(var(f, w + 1, "VACA") == 1).OnlyEnforceIf(b)
    
        model.Add(sum(b for _, b in fall_starts) == 1)
        model.Add(sum(b for _, b in spring_starts) == 1)
    
        # Ensure PGY-4 VACA occurs only within the two blocks
        for w in range(W):
            covering = [b for sw, b in fall_starts if sw == w or sw + 1 == w]
            covering += [b for sw, b in spring_starts if sw == w or sw + 1 == w]
            if covering:
                model.Add(var(f, w, "VACA") <= sum(covering))
    
    # PGY-5/6: 4 weeks total; 2 fall + 2 spring; no vacation in July, CHEST, ATS; PGY-5 no last two weeks
    for f in PGY5 + PGY6:
        for w in range(W):
            d = WEEK_STARTS[w]
            if w in JULY_WEEKS or d == CHEST_WEEK or d == ATS_WEEK:
                model.Add(var(f, w, "VACA") == 0)
            if f in PGY5 and d in PGY5_NO_VACA_EXTRA:
                model.Add(var(f, w, "VACA") == 0)
    
        name = FELLOWS[f][0]
        # Removed fall/spring split requirement for PGY-5/6
    
    # No >2 consecutive vacation weeks for PGY-5/6
    # SV visa exception: allow up to 3 consecutive weeks (but not 4)
    for f in PGY5 + PGY6:
        name = FELLOWS[f][0]
        if name == "SV" and ENABLE_SV_VISA_EXCEPTION:
            for w in range(W - 3):
                model.Add(
                    var(f, w, "VACA") +
                    var(f, w + 1, "VACA") +
                    var(f, w + 2, "VACA") +
                    var(f, w + 3, "VACA") <= 3
                )
            continue
        for w in range(W - 2):
            model.Add(var(f, w, "VACA") + var(f, w + 1, "VACA") + var(f, w + 2, "VACA") <= 2)
    
    
    # ---------------------------
    # Soft constraints / Objective
    # ---------------------------
    
    # Vacation requests (grouped, ranked options)
    for fellow_name, requests in VACATION_REQUESTS.items():
        f_idx = next(i for i, (name, _) in enumerate(FELLOWS) if name == fellow_name)
        pgy = FELLOWS[f_idx][1]
        mult = PGY_VAC_PRIORITY_MULT[pgy]
    
        for req_idx, req in enumerate(requests):
            options = req.get("options", [])
            rank_bools = {1: [], 2: [], 3: []}
    
            for opt_idx, opt in enumerate(options):
                start = opt["start"]
                length = opt["length"]
                rank = opt["rank"]
                w0 = widx(start)
                weeks = [w0 + k for k in range(length)]
                granted = model.NewBoolVar(f"vac_req_{fellow_name}_{req_idx}_opt{opt_idx}_r{rank}")
    
                # granted if all weeks are VACA
                model.Add(sum(var(f_idx, w, "VACA") for w in weeks) >= length).OnlyEnforceIf(granted)
                model.Add(sum(var(f_idx, w, "VACA") for w in weeks) <= length - 1).OnlyEnforceIf(granted.Not())
    
                if rank in rank_bools:
                    rank_bools[rank].append(granted)
    
            # Aggregate per-rank granted flags
            rank1 = model.NewBoolVar(f"vac_req_{fellow_name}_{req_idx}_rank1")
            rank2 = model.NewBoolVar(f"vac_req_{fellow_name}_{req_idx}_rank2")
            rank3 = model.NewBoolVar(f"vac_req_{fellow_name}_{req_idx}_rank3")
    
            if rank_bools[1]:
                model.AddMaxEquality(rank1, rank_bools[1])
            else:
                model.Add(rank1 == 0)
    
            if rank_bools[2]:
                model.AddMaxEquality(rank2, rank_bools[2])
            else:
                model.Add(rank2 == 0)
    
            if rank_bools[3]:
                model.AddMaxEquality(rank3, rank_bools[3])
            else:
                model.Add(rank3 == 0)
    
            # Determine best granted rank (rank1 > rank2 > rank3 > none)
            rank2_only = model.NewBoolVar(f"vac_req_{fellow_name}_{req_idx}_rank2_only")
            model.Add(rank2_only <= rank2)
            model.Add(rank2_only + rank1 <= 1)
            model.Add(rank2_only >= rank2 - rank1)
    
            rank3_only = model.NewBoolVar(f"vac_req_{fellow_name}_{req_idx}_rank3_only")
            model.Add(rank3_only <= rank3)
            model.Add(rank3_only + rank1 <= 1)
            model.Add(rank3_only + rank2 <= 1)
            model.Add(rank3_only >= rank3 - rank1 - rank2)
    
            none = model.NewBoolVar(f"vac_req_{fellow_name}_{req_idx}_none")
            model.Add(rank1 + rank2_only + rank3_only + none == 1)
    
            # Penalties: rank1 = 0, rank2 = small, rank3 = larger, none = largest
            objective_terms.append(mult * WEIGHTS["vac_rank2"] * rank2_only)
            objective_terms.append(mult * WEIGHTS["vac_rank3"] * rank3_only)
            objective_terms.append(mult * WEIGHTS["vac_rank1"] * none)
    
            soft_report["vacation"].append({
                "fellow": fellow_name,
                "req_idx": req_idx,
                "rank2_only": rank2_only,
                "rank3_only": rank3_only,
                "none": none,
                "mult": mult,
            })
    
    # Pairing preferences
    for w in range(W):
        pgy4_pulm = sum(var(f, w, "PULM") for f in PGY4)
        both_pgy4 = model.NewBoolVar(f"pulm_both_pgy4_w{w}")
        model.Add(pgy4_pulm == 2).OnlyEnforceIf(both_pgy4)
        model.Add(pgy4_pulm <= 1).OnlyEnforceIf(both_pgy4.Not())
    
        # Linear decay penalty based on prior PULM weeks completed by PGY-4s
        # penalty = max(0, base - 20 * prior_pulm)
        base = WEIGHTS["pulm_pair_pgy4"]
        for f in PGY4:
            prior_pulm = model.NewIntVar(0, W, f"pgy4_prior_pulm_f{f}_w{w}")
            if w == 0:
                model.Add(prior_pulm == 0)
            else:
                model.Add(prior_pulm == sum(
                    var(f, wp, "PULM") for wp in range(w)
                ))
    
            expr = model.NewIntVar(base - 20 * W, base, f"pgy4_pulm_pair_expr_f{f}_w{w}")
            model.Add(expr == base - 20 * prior_pulm)
    
            decayed = model.NewIntVar(0, base, f"pgy4_pulm_pair_decayed_f{f}_w{w}")
            model.AddMaxEquality(decayed, [expr, 0])
    
            pulm_pair_pen = model.NewIntVar(0, base, f"pgy4_pulm_pair_pen_f{f}_w{w}")
            is_pulm = var(f, w, "PULM")
    
            pulm_pair_active = model.NewBoolVar(f"pgy4_pulm_pair_active_f{f}_w{w}")
            model.Add(pulm_pair_active <= both_pgy4)
            model.Add(pulm_pair_active <= is_pulm)
            model.Add(pulm_pair_active >= both_pgy4 + is_pulm - 1)
    
            model.Add(pulm_pair_pen <= decayed)
            model.Add(pulm_pair_pen <= base * pulm_pair_active)
            model.Add(pulm_pair_pen >= decayed - base * (1 - pulm_pair_active))
    
            objective_terms.append(pulm_pair_pen)
            soft_report["pulm_pair"].append((w, f, pulm_pair_pen))
    
        micu1_pgy4 = sum(var(f, w, "MICU1") for f in PGY4)
        micu2_pgy4 = sum(var(f, w, "MICU2") for f in PGY4)
        micu_both = model.NewBoolVar(f"micu_both_pgy4_w{w}")
        model.Add(micu1_pgy4 + micu2_pgy4 >= 2).OnlyEnforceIf(micu_both)
        model.Add(micu1_pgy4 + micu2_pgy4 <= 1).OnlyEnforceIf(micu_both.Not())
    
        # Linear decay penalty based on prior MICU weeks completed by PGY-4s
        # penalty = max(0, base - 20 * prior_micu)
        base = WEIGHTS["micu_pair_pgy4"]
        for f in PGY4:
            prior_micu = model.NewIntVar(0, W, f"pgy4_prior_micu_f{f}_w{w}")
            if w == 0:
                model.Add(prior_micu == 0)
            else:
                model.Add(prior_micu == sum(
                    var(f, wp, "MICU1") + var(f, wp, "MICU2") for wp in range(w)
                ))
    
            expr = model.NewIntVar(base - 20 * W, base, f"pgy4_micu_pair_expr_f{f}_w{w}")
            model.Add(expr == base - 20 * prior_micu)
    
            decayed = model.NewIntVar(0, base, f"pgy4_micu_pair_decayed_f{f}_w{w}")
            model.AddMaxEquality(decayed, [expr, 0])
    
            micu_pair_pen = model.NewIntVar(0, base, f"pgy4_micu_pair_pen_f{f}_w{w}")
            is_micu = var(f, w, "MICU1") + var(f, w, "MICU2")
    
            micu_pair_active = model.NewBoolVar(f"pgy4_micu_pair_active_f{f}_w{w}")
            model.Add(micu_pair_active <= micu_both)
            model.Add(micu_pair_active <= is_micu)
            model.Add(micu_pair_active >= micu_both + is_micu - 1)
    
            model.Add(micu_pair_pen <= decayed)
            model.Add(micu_pair_pen <= base * micu_pair_active)
            model.Add(micu_pair_pen >= decayed - base * (1 - micu_pair_active))
    
            objective_terms.append(micu_pair_pen)
            soft_report["micu_pair"].append((w, f, micu_pair_pen))
    
    # Fairness: minimize hard-week range within each PGY class
    for label, group in [("pgy4", PGY4), ("pgy5", PGY5), ("pgy6", PGY6)]:
        hard_counts = []
        for f in group:
            hard_count = model.NewIntVar(0, W, f"hard_count_{label}_f{f}")
            model.Add(hard_count == sum(var(f, w, r) for w in range(W) for r in HARD))
            if label == "pgy6" and f == TM_IDX:
                # Normalize TM's hard count by removing the 2 extra mandated NF weeks
                tm_fair_hard = model.NewIntVar(0, W, f"hard_count_{label}_f{f}_fair")
                model.Add(tm_fair_hard == hard_count - 2)
                hard_counts.append(tm_fair_hard)
            else:
                hard_counts.append(hard_count)
    
        max_hard = model.NewIntVar(0, W, f"max_hard_{label}")
        min_hard = model.NewIntVar(0, W, f"min_hard_{label}")
        model.AddMaxEquality(max_hard, hard_counts)
        model.AddMinEquality(min_hard, hard_counts)
        objective_terms.append(WEIGHTS["fair_hard_range"] * (max_hard - min_hard))
        soft_report["fair_hard"].append((label, max_hard, min_hard))
    
    # Fairness: minimize MICU range among PGY-6
    micu6_counts = []
    for f in PGY6:
        micu = model.NewIntVar(0, W, f"micu6_count_f{f}")
        model.Add(micu == sum(var(f, w, "MICU1") + var(f, w, "MICU2") for w in range(W)))
        micu6_counts.append(micu)
    
    max_micu6 = model.NewIntVar(0, W, "max_micu6")
    min_micu6 = model.NewIntVar(0, W, "min_micu6")
    model.AddMaxEquality(max_micu6, micu6_counts)
    model.AddMinEquality(min_micu6, micu6_counts)
    objective_terms.append(WEIGHTS["fair_micu6_range"] * (max_micu6 - min_micu6))
    soft_report["fair_micu6"].append((max_micu6, min_micu6))

    # Fairness: minimize PHTN range among PGY-5/6
    phtn_counts = []
    for f in PGY5 + PGY6:
        phtn_counts.append(rotation_totals[(f, "PHTN")])
    max_phtn = model.NewIntVar(0, W, "max_phtn")
    min_phtn = model.NewIntVar(0, W, "min_phtn")
    model.AddMaxEquality(max_phtn, phtn_counts)
    model.AddMinEquality(min_phtn, phtn_counts)
    objective_terms.append(WEIGHTS["fair_phtn_range"] * (max_phtn - min_phtn))
    soft_report["fair_phtn"].append((max_phtn, min_phtn))
    
    # Penalize longer hard streaks (soft)
    for f in range(F):
        for w in range(W - 3):
            hard3 = model.NewBoolVar(f"hard3_f{f}_w{w}")
            model.Add(sum(var(f, w + k, r) for k in range(3) for r in HARD) >= 3).OnlyEnforceIf(hard3)
            model.Add(sum(var(f, w + k, r) for k in range(3) for r in HARD) <= 2).OnlyEnforceIf(hard3.Not())
            if f in PGY4:
                # Exempt 3-week windows fully inside the 4-week pre-BRONCH block (all BRONCH)
                in_pre_bronch4 = model.NewBoolVar(f"hard3_in_prebronch4_f{f}_w{w}")
                start_flags = []
                for ws, s in pre_bronch4_starts[f]:
                    if w >= ws and w <= ws + 1:
                        start_flags.append(s)
                if start_flags:
                    model.AddMaxEquality(in_pre_bronch4, start_flags)
                else:
                    model.Add(in_pre_bronch4 == 0)

                hard3_active = model.NewBoolVar(f"hard3_active_f{f}_w{w}")
                model.Add(hard3_active <= hard3)
                model.Add(hard3_active <= in_pre_bronch4.Not())
                model.Add(hard3_active >= hard3 - in_pre_bronch4)
                objective_terms.append(WEIGHTS["hard_3_run"] * hard3_active)
                soft_report["hard3"].append((f, w, hard3_active))
            else:
                objective_terms.append(WEIGHTS["hard_3_run"] * hard3)
                soft_report["hard3"].append((f, w, hard3))
    
        for w in range(W - 5):
            hard5 = model.NewBoolVar(f"hard5_f{f}_w{w}")
            model.Add(sum(var(f, w + k, r) for k in range(5) for r in HARD) >= 5).OnlyEnforceIf(hard5)
            model.Add(sum(var(f, w + k, r) for k in range(5) for r in HARD) <= 4).OnlyEnforceIf(hard5.Not())
            objective_terms.append(WEIGHTS["hard_5_run"] * hard5)
            soft_report["hard5"].append((f, w, hard5))
    
    # Post-NF penalty preference: Easy > Medium > Hard
    for f in range(F):
        for w in range(W - 1):
            nf = var(f, w, "NF")
            medium_next = model.NewBoolVar(f"nf_medium_next_f{f}_w{w}")
            hard_next = model.NewBoolVar(f"nf_hard_next_f{f}_w{w}")
    
            model.Add(sum(var(f, w + 1, r) for r in MEDIUM) >= 1).OnlyEnforceIf(medium_next)
            model.Add(sum(var(f, w + 1, r) for r in MEDIUM) == 0).OnlyEnforceIf(medium_next.Not())
    
            model.Add(sum(var(f, w + 1, r) for r in HARD) >= 1).OnlyEnforceIf(hard_next)
            model.Add(sum(var(f, w + 1, r) for r in HARD) == 0).OnlyEnforceIf(hard_next.Not())
    
            nf_medium = model.NewBoolVar(f"nf_medium_f{f}_w{w}")
            model.Add(nf_medium <= nf)
            model.Add(nf_medium <= medium_next)
            model.Add(nf_medium >= nf + medium_next - 1)
    
            nf_hard = model.NewBoolVar(f"nf_hard_f{f}_w{w}")
            model.Add(nf_hard <= nf)
            model.Add(nf_hard <= hard_next)
            model.Add(nf_hard >= nf + hard_next - 1)
    
            objective_terms.append(WEIGHTS["post_nf_medium"] * nf_medium)
            objective_terms.append(WEIGHTS["post_nf_hard"] * nf_hard)
            soft_report["post_nf_medium"].append((f, w, nf_medium))
            soft_report["post_nf_hard"].append((f, w, nf_hard))
    
    # NSICU consecutive preference (penalize if split)
    for f in PGY4 + PGY5:
        consec_pairs = []
        for w in range(W - 1):
            b = model.NewBoolVar(f"nsicu_pair_f{f}_w{w}")
            model.Add(var(f, w, "NSICU") + var(f, w + 1, "NSICU") >= 2).OnlyEnforceIf(b)
            model.Add(var(f, w, "NSICU") + var(f, w + 1, "NSICU") <= 1).OnlyEnforceIf(b.Not())
            consec_pairs.append(b)
        nsicu_consec = model.NewBoolVar(f"nsicu_consec_f{f}")
        model.AddMaxEquality(nsicu_consec, consec_pairs)
        objective_terms.append(WEIGHTS["nsicu_nonconsec"] * (1 - nsicu_consec))
        soft_report["nsicu_nonconsec"].append((f, nsicu_consec))

    # CCU consecutive preference (penalize if split) - PGY-6 only
    for f in PGY6:
        consec_pairs = []
        for w in range(W - 1):
            b = model.NewBoolVar(f"ccu_pair_f{f}_w{w}")
            model.Add(var(f, w, "CCU") + var(f, w + 1, "CCU") >= 2).OnlyEnforceIf(b)
            model.Add(var(f, w, "CCU") + var(f, w + 1, "CCU") <= 1).OnlyEnforceIf(b.Not())
            consec_pairs.append(b)
        ccu_consec = model.NewBoolVar(f"ccu_consec_f{f}")
        model.AddMaxEquality(ccu_consec, consec_pairs)
        objective_terms.append(WEIGHTS["ccu_nonconsec"] * (1 - ccu_consec))
        soft_report["ccu_nonconsec"].append((f, ccu_consec))
    
    # Prefer more MICU before the first NF for PGY-4s (penalize MICU after first NF)
    # Dynamic penalty: base - 2 * (MICU weeks completed before first NF)
    for f in PGY4:
        micu_before_first_nf_terms = []
        had_nf_before_flags = []
    
        for w in range(W):
            prior_nf = sum(var(f, wp, "NF") for wp in range(w)) if w > 0 else 0
            had_nf_before = model.NewBoolVar(f"had_nf_before_f{f}_w{w}")
            if w == 0:
                model.Add(had_nf_before == 0)
            else:
                model.Add(prior_nf >= 1).OnlyEnforceIf(had_nf_before)
                model.Add(prior_nf == 0).OnlyEnforceIf(had_nf_before.Not())
            had_nf_before_flags.append(had_nf_before)
    
            is_micu = var(f, w, "MICU1") + var(f, w, "MICU2")
            micu_before_nf = model.NewBoolVar(f"micu_before_nf_f{f}_w{w}")
            model.Add(micu_before_nf <= is_micu)
            model.Add(micu_before_nf <= had_nf_before.Not())
            model.Add(micu_before_nf >= is_micu - had_nf_before)
            micu_before_first_nf_terms.append(micu_before_nf)
    
        micu_before_first_nf = model.NewIntVar(0, W, f"micu_before_first_nf_f{f}")
        model.Add(micu_before_first_nf == sum(micu_before_first_nf_terms))
    
        base = 20
        expr = model.NewIntVar(base - 2 * W, base, f"micu_after_nf_expr_f{f}")
        model.Add(expr == base - 2 * micu_before_first_nf)
        decayed = model.NewIntVar(0, base, f"micu_after_nf_decayed_f{f}")
        model.AddMaxEquality(decayed, [expr, 0])
    
        for w in range(W):
            is_micu = var(f, w, "MICU1") + var(f, w, "MICU2")
            micu_after_nf = model.NewBoolVar(f"micu_after_nf_f{f}_w{w}")
            model.Add(micu_after_nf <= is_micu)
            model.Add(micu_after_nf <= had_nf_before_flags[w])
            model.Add(micu_after_nf >= is_micu + had_nf_before_flags[w] - 1)
    
            micu_after_nf_pen = model.NewIntVar(0, base, f"micu_after_nf_pen_f{f}_w{w}")
            model.Add(micu_after_nf_pen <= decayed)
            model.Add(micu_after_nf_pen <= base * micu_after_nf)
            model.Add(micu_after_nf_pen >= decayed - base * (1 - micu_after_nf))
    
            objective_terms.append(micu_after_nf_pen)
            soft_report["micu_after_first_nf"].append((f, w, micu_after_nf_pen))
    
    # Prefer buffer rotations immediately before/after vacation blocks
    # PGY-4: EASY rotations excluding VACA/ONBD/RSCH
    # PGY-5/6: RSCH only
    PGY4_BUFFER_ROTATIONS = {"ELECT", "PHYSIO", "RADS", "RIF", "AIRWAY"}
    
    for f in range(F):
        pgy = FELLOWS[f][1]
        for w in range(W):
            is_vaca = var(f, w, "VACA")
            is_start = model.NewBoolVar(f"vaca_start_f{f}_w{w}")
            is_end = model.NewBoolVar(f"vaca_end_f{f}_w{w}")
    
            if w == 0:
                model.Add(is_start == is_vaca)
            else:
                prev_not = model.NewBoolVar(f"vaca_prev_not_f{f}_w{w}")
                model.Add(prev_not == 1 - var(f, w - 1, "VACA"))
                model.Add(is_start <= is_vaca)
                model.Add(is_start <= prev_not)
                model.Add(is_start >= is_vaca + prev_not - 1)
    
            if w == W - 1:
                model.Add(is_end == is_vaca)
            else:
                next_not = model.NewBoolVar(f"vaca_next_not_f{f}_w{w}")
                model.Add(next_not == 1 - var(f, w + 1, "VACA"))
                model.Add(is_end <= is_vaca)
                model.Add(is_end <= next_not)
                model.Add(is_end >= is_vaca + next_not - 1)
    
            if pgy == 4:
                # If this is a vacation block start, prefer EASY rotation in the previous week
                if w > 0:
                    before_easy = model.NewBoolVar(f"vaca_before_easy_f{f}_w{w}")
                    model.Add(
                        before_easy
                        <= sum(var(f, w - 1, rname) for rname in PGY4_BUFFER_ROTATIONS)
                    )
                    model.Add(before_easy <= is_start)
                    model.Add(
                        before_easy
                        >= sum(var(f, w - 1, rname) for rname in PGY4_BUFFER_ROTATIONS)
                        + is_start
                        - 1
                    )
                    objective_terms.append(WEIGHTS["rsch_around_vaca"] * (is_start - before_easy))
                    soft_report["rsch_around_vaca"].append((f, w, "before", is_start, before_easy))
    
                # If this is a vacation block end, prefer EASY rotation in the next week
                if w < W - 1:
                    after_easy = model.NewBoolVar(f"vaca_after_easy_f{f}_w{w}")
                    model.Add(
                        after_easy
                        <= sum(var(f, w + 1, rname) for rname in PGY4_BUFFER_ROTATIONS)
                    )
                    model.Add(after_easy <= is_end)
                    model.Add(
                        after_easy
                        >= sum(var(f, w + 1, rname) for rname in PGY4_BUFFER_ROTATIONS)
                        + is_end
                        - 1
                    )
                    objective_terms.append(WEIGHTS["rsch_around_vaca"] * (is_end - after_easy))
                    soft_report["rsch_around_vaca"].append((f, w, "after", is_end, after_easy))
            else:
                # PGY-5/6: prefer RSCH in the previous/next week
                if w > 0:
                    before_rsch = model.NewBoolVar(f"vaca_before_rsch_f{f}_w{w}")
                    model.Add(before_rsch <= var(f, w - 1, "RSCH"))
                    model.Add(before_rsch <= is_start)
                    model.Add(before_rsch >= var(f, w - 1, "RSCH") + is_start - 1)
                    objective_terms.append(WEIGHTS["rsch_around_vaca"] * (is_start - before_rsch))
                    soft_report["rsch_around_vaca"].append((f, w, "before", is_start, before_rsch))
    
                if w < W - 1:
                    after_rsch = model.NewBoolVar(f"vaca_after_rsch_f{f}_w{w}")
                    model.Add(after_rsch <= var(f, w + 1, "RSCH"))
                    model.Add(after_rsch <= is_end)
                    model.Add(after_rsch >= var(f, w + 1, "RSCH") + is_end - 1)
                    objective_terms.append(WEIGHTS["rsch_around_vaca"] * (is_end - after_rsch))
                    soft_report["rsch_around_vaca"].append((f, w, "after", is_end, after_rsch))
    
    # Pulmonary Boards: prefer PGY-6 on EASY rotations in the 4 weeks prior (weighted toward the end)
    pre_boards_weeks = [w for w, d in enumerate(WEEK_STARTS) if PULM_BOARDS_WEEK - timedelta(days=28) <= d < PULM_BOARDS_WEEK]
    pre_boards_curve = [1, 1, 2, 3]
    pre_boards_base = 350
    pre_boards_decay = 50
    for f in PGY6:
        credit_terms = []
        for i, w in enumerate(pre_boards_weeks):
            ok = model.NewIntVar(0, 1, f"pgy6_pre_boards_easy_f{f}_w{w}")
            model.Add(ok == sum(var(f, w, r) for r in EASY))
            credit_terms.append(pre_boards_curve[i] * ok)

        credit = model.NewIntVar(0, sum(pre_boards_curve), f"pgy6_pre_boards_credit_f{f}")
        model.Add(credit == sum(credit_terms))

        pen = model.NewIntVar(0, pre_boards_base, f"pgy6_pre_boards_pen_f{f}")
        model.AddMaxEquality(pen, [0, pre_boards_base - pre_boards_decay * credit])
        objective_terms.append(pen)
        soft_report["pgy6_pre_boards_rsch"].append((f, credit, pen))

    # PGY-6 last 8 weeks: prefer RSCH/VACA, weighted toward the end
    last8 = list(range(W - len(PGY6_LAST8_CURVE), W))
    for f in PGY6:
        credit_terms = []
        for i, w in enumerate(last8):
            ok = model.NewIntVar(0, 1, f"pgy6_last8_ok_f{f}_w{w}")
            model.Add(ok == var(f, w, "RSCH") + var(f, w, "VACA"))
            credit_terms.append(PGY6_LAST8_CURVE[i] * ok)

        credit = model.NewIntVar(0, sum(PGY6_LAST8_CURVE), f"pgy6_last8_credit_f{f}")
        model.Add(credit == sum(credit_terms))

        pen = model.NewIntVar(0, PGY6_LAST8_BASE, f"pgy6_last8_pen_f{f}")
        model.AddMaxEquality(pen, [0, PGY6_LAST8_BASE - PGY6_LAST8_DECAY * credit])
        objective_terms.append(pen)
        soft_report["pgy6_last8_rsch_vaca"].append((f, credit, pen))
    
    # If RADS is at 2 and RIF is at 4, favor assigning RADS (penalize RIF in that case)
    for f in PGY4:
        rads_total = rotation_totals[(f, "RADS")]
        rif_total = rotation_totals[(f, "RIF")]
    
        rads_min = model.NewBoolVar(f"rads_min_f{f}")
        rif_min = model.NewBoolVar(f"rif_min_f{f}")
    
        model.Add(rads_total == 2).OnlyEnforceIf(rads_min)
        model.Add(rads_total != 2).OnlyEnforceIf(rads_min.Not())
    
        model.Add(rif_total == 4).OnlyEnforceIf(rif_min)
        model.Add(rif_total != 4).OnlyEnforceIf(rif_min.Not())
    
        rad_and_rif_min = model.NewBoolVar(f"rads2_rif4_f{f}")
        model.Add(rad_and_rif_min <= rads_min)
        model.Add(rad_and_rif_min <= rif_min)
        model.Add(rad_and_rif_min >= rads_min + rif_min - 1)
    
        for w in range(W):
            rif_when_min = model.NewBoolVar(f"rif_when_min_f{f}_w{w}")
            model.Add(rif_when_min <= rad_and_rif_min)
            model.Add(rif_when_min <= var(f, w, "RIF"))
            model.Add(rif_when_min >= rad_and_rif_min + var(f, w, "RIF") - 1)
            objective_terms.append(WEIGHTS["rif_when_rads_min"] * rif_when_min)
            soft_report["rif_when_rads_min"].append((f, w, rif_when_min))
    
    # Objective
    model.Minimize(sum(objective_terms))
    
    # ---------------------------
    # Solve
    # ---------------------------
    return model, var, F, W, soft_report

def write_reports(solver, var, soft_report, suffix: str, output_dir: str = "."):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    W_local = len(WEEK_STARTS)
    # Build schedule table: rows = weeks, cols = fellows
    schedule = []
    for w in range(W_local):
        row = {"WeekStart": WEEK_STARTS[w].isoformat()}
        for f, (name, _) in enumerate(FELLOWS):
            assigned = None
            for rname in ROTATIONS:
                if solver.Value(var(f, w, rname)) == 1:
                    assigned = rname
                    break
            row[name] = assigned
        schedule.append(row)

    df = pd.DataFrame(schedule)
    print(df.head())

    # Summary counts
    summary = []
    for f, (name, _) in enumerate(FELLOWS):
        counts = {"Fellow": name}
        for rname in ROTATIONS:
            counts[rname] = sum(solver.Value(var(f, w, rname)) for w in range(W_local))
        summary.append(counts)
    summary_df = pd.DataFrame(summary)
    print("\nRotation totals (per fellow):")
    print(summary_df)

    # Save outputs
    schedule_path = out_dir / f"schedule_by_week_{suffix}.csv"
    totals_path = out_dir / f"rotation_totals_{suffix}.csv"
    df.to_csv(schedule_path, index=False)
    summary_df.to_csv(totals_path, index=False)
    print(f"\nWrote: {schedule_path}, {totals_path}")

    # Soft constraint report
    soft_rows = []

    # Vacation penalties
    for entry in soft_report["vacation"]:
        fellow = entry["fellow"]
        req_idx = entry["req_idx"]
        mult = entry["mult"]
        if solver.Value(entry["rank2_only"]) == 1:
            soft_rows.append({
                "constraint": "vacation_rank2",
                "fellow": fellow,
                "week": "",
                "detail": f"request {req_idx} rank2",
                "weight": WEIGHTS["vac_rank2"] * mult,
                "penalty": WEIGHTS["vac_rank2"] * mult,
            })
        if solver.Value(entry["rank3_only"]) == 1:
            soft_rows.append({
                "constraint": "vacation_rank3",
                "fellow": fellow,
                "week": "",
                "detail": f"request {req_idx} rank3",
                "weight": WEIGHTS["vac_rank3"] * mult,
                "penalty": WEIGHTS["vac_rank3"] * mult,
            })
        if solver.Value(entry["none"]) == 1:
            soft_rows.append({
                "constraint": "vacation_none",
                "fellow": fellow,
                "week": "",
                "detail": f"request {req_idx} none",
                "weight": WEIGHTS["vac_rank1"] * mult,
                "penalty": WEIGHTS["vac_rank1"] * mult,
            })

    # Pairing penalties
    for w, f, varb in soft_report["pulm_pair"]:
        pen = solver.Value(varb)
        if pen > 0:
            soft_rows.append({
                "constraint": "pulm_pair_pgy4",
                "fellow": FELLOWS[f][0],
                "week": WEEK_STARTS[w].isoformat(),
                "detail": "both PULM are PGY-4 (decayed)",
                "weight": WEIGHTS["pulm_pair_pgy4"],
                "penalty": pen,
            })
    for w, f, varb in soft_report["micu_pair"]:
        pen = solver.Value(varb)
        if pen > 0:
            soft_rows.append({
                "constraint": "micu_pair_pgy4",
                "fellow": FELLOWS[f][0],
                "week": WEEK_STARTS[w].isoformat(),
                "detail": "both MICU are PGY-4 (decayed)",
                "weight": WEIGHTS["micu_pair_pgy4"],
                "penalty": pen,
            })

    # Fairness penalties
    for label, max_var, min_var in soft_report["fair_hard"]:
        max_val = solver.Value(max_var)
        min_val = solver.Value(min_var)
        rng = max_val - min_val
        if rng > 0:
            soft_rows.append({
                "constraint": "fair_hard_range",
                "fellow": label,
                "week": "",
                "detail": f"range {min_val}-{max_val}",
                "weight": WEIGHTS["fair_hard_range"],
                "penalty": WEIGHTS["fair_hard_range"] * rng,
            })
    for max_var, min_var in soft_report["fair_micu6"]:
        max_val = solver.Value(max_var)
        min_val = solver.Value(min_var)
        rng = max_val - min_val
        if rng > 0:
            soft_rows.append({
                "constraint": "fair_micu6_range",
                "fellow": "pgy6",
                "week": "",
                "detail": f"range {min_val}-{max_val}",
                "weight": WEIGHTS["fair_micu6_range"],
                "penalty": WEIGHTS["fair_micu6_range"] * rng,
            })

    for max_var, min_var in soft_report["fair_phtn"]:
        max_val = solver.Value(max_var)
        min_val = solver.Value(min_var)
        rng = max_val - min_val
        if rng > 0:
            soft_rows.append({
                "constraint": "fair_phtn_range",
                "fellow": "pgy5_6",
                "week": "",
                "detail": f"range {min_val}-{max_val}",
                "weight": WEIGHTS["fair_phtn_range"],
                "penalty": WEIGHTS["fair_phtn_range"] * rng,
            })

    # Hard streak penalties
    for f, w, varb in soft_report["hard3"]:
        if solver.Value(varb) > 0:
            soft_rows.append({
                "constraint": "hard_3_run",
                "fellow": FELLOWS[f][0],
                "week": WEEK_STARTS[w].isoformat(),
                "detail": "3-week hard streak",
                "weight": WEIGHTS["hard_3_run"],
                "penalty": solver.Value(varb),
            })
    for f, w, varb in soft_report["hard5"]:
        if solver.Value(varb) == 1:
            soft_rows.append({
                "constraint": "hard_5_run",
                "fellow": FELLOWS[f][0],
                "week": WEEK_STARTS[w].isoformat(),
                "detail": "5-week hard streak",
                "weight": WEIGHTS["hard_5_run"],
                "penalty": WEIGHTS["hard_5_run"],
            })

    # Post-NF penalties
    for f, w, varb in soft_report["post_nf_medium"]:
        if solver.Value(varb) == 1:
            soft_rows.append({
                "constraint": "post_nf_medium",
                "fellow": FELLOWS[f][0],
                "week": WEEK_STARTS[w].isoformat(),
                "detail": "NF -> MEDIUM",
                "weight": WEIGHTS["post_nf_medium"],
                "penalty": WEIGHTS["post_nf_medium"],
            })
    for f, w, varb in soft_report["post_nf_hard"]:
        if solver.Value(varb) == 1:
            soft_rows.append({
                "constraint": "post_nf_hard",
                "fellow": FELLOWS[f][0],
                "week": WEEK_STARTS[w].isoformat(),
                "detail": "NF -> HARD",
                "weight": WEIGHTS["post_nf_hard"],
                "penalty": WEIGHTS["post_nf_hard"],
            })

    # RSCH exact-3 penalty
    for w, varb in soft_report["rsch_exact3"]:
        if solver.Value(varb) == 1:
            soft_rows.append({
                "constraint": "rsch_exact3",
                "fellow": "",
                "week": WEEK_STARTS[w].isoformat(),
                "detail": "RSCH coverage exactly 3",
                "weight": WEIGHTS["rsch_exact3"],
                "penalty": WEIGHTS["rsch_exact3"],
            })

    # NSICU non-consecutive
    for f, varb in soft_report["nsicu_nonconsec"]:
        if solver.Value(varb) == 0:
            soft_rows.append({
                "constraint": "nsicu_nonconsec",
                "fellow": FELLOWS[f][0],
                "week": "",
                "detail": "NSICU not consecutive",
                "weight": WEIGHTS["nsicu_nonconsec"],
                "penalty": WEIGHTS["nsicu_nonconsec"],
            })

    # CCU non-consecutive
    for f, varb in soft_report["ccu_nonconsec"]:
        if solver.Value(varb) == 0:
            soft_rows.append({
                "constraint": "ccu_nonconsec",
                "fellow": FELLOWS[f][0],
                "week": "",
                "detail": "CCU not consecutive",
                "weight": WEIGHTS["ccu_nonconsec"],
                "penalty": WEIGHTS["ccu_nonconsec"],
            })

    for f, w, varb in soft_report["rif_when_rads_min"]:
        if solver.Value(varb) == 1:
            soft_rows.append({
                "constraint": "rif_when_rads_min",
                "fellow": FELLOWS[f][0],
                "week": WEEK_STARTS[w].isoformat(),
                "detail": "RIF when RADS=2 and RIF=4",
                "weight": WEIGHTS["rif_when_rads_min"],
                "penalty": WEIGHTS["rif_when_rads_min"],
            })

    # RSCH around vacation blocks
    for f, w, pos, is_block, has_rsch in soft_report["rsch_around_vaca"]:
        is_block_val = solver.Value(is_block)
        has_rsch_val = solver.Value(has_rsch)
        if is_block_val == 1 and has_rsch_val == 0:
            soft_rows.append({
                "constraint": "rsch_around_vaca",
                "fellow": FELLOWS[f][0],
                "week": WEEK_STARTS[w].isoformat(),
                "detail": f"{pos} vacation block missing RSCH",
                "weight": WEIGHTS["rsch_around_vaca"],
                "penalty": WEIGHTS["rsch_around_vaca"],
            })

    # MICU after first NF (PGY-4 preference)
    for f, w, varb in soft_report["micu_after_first_nf"]:
        if solver.Value(varb) == 1:
            soft_rows.append({
                "constraint": "micu_after_first_nf",
                "fellow": FELLOWS[f][0],
                "week": WEEK_STARTS[w].isoformat(),
                "detail": "MICU after first NF",
                "weight": WEIGHTS["micu_after_first_nf"],
                "penalty": WEIGHTS["micu_after_first_nf"],
            })

    # PGY-6 pre-boards EASY preference (weighted)
    for f, credit, pen in soft_report["pgy6_pre_boards_rsch"]:
        pen_val = solver.Value(pen)
        if pen_val > 0:
            soft_rows.append({
                "constraint": "pgy6_pre_boards_easy",
                "fellow": FELLOWS[f][0],
                "week": "pre-boards window",
                "detail": "EASY rotations weighted toward boards",
                "weight": 350,
                "penalty": pen_val,
            })

    # PGY-6 last 8 weeks RSCH/VACA preference (weighted)
    for f, credit, pen in soft_report["pgy6_last8_rsch_vaca"]:
        pen_val = solver.Value(pen)
        if pen_val > 0:
            soft_rows.append({
                "constraint": "pgy6_last8_rsch_vaca",
                "fellow": FELLOWS[f][0],
                "week": "last 8 weeks",
                "detail": f"credit={solver.Value(credit)}; base={PGY6_LAST8_BASE}; decay={PGY6_LAST8_DECAY}; curve={PGY6_LAST8_CURVE}",
                "weight": PGY6_LAST8_BASE,
                "penalty": pen_val,
            })

    # Non-Sinai PGY-4 MICU early penalty
    for f, w, varb, weight in soft_report["non_sinai_pgy4_micu_early"]:
        if solver.Value(varb) > 0:
            soft_rows.append({
                "constraint": "non_sinai_pgy4_micu_early",
                "fellow": FELLOWS[f][0],
                "week": WEEK_STARTS[w].isoformat(),
                "detail": "Non-Sinai PGY-4 on MICU early",
                "weight": weight,
                "penalty": weight,
            })

    soft_df = pd.DataFrame(soft_rows)
    soft_path = out_dir / f"soft_constraint_report_{suffix}.csv"
    soft_df.to_csv(soft_path, index=False)
    print(f"Wrote: {soft_path}")

# ---------------------------
# Staged solve: vacations first, then full optimization
# ---------------------------
VAC_KEYS = {"vac_rank1", "vac_rank2", "vac_rank3", "rsch_around_vaca"}
PGY4_ONLY_KEYS = {"pulm_pair_pgy4", "micu_pair_pgy4", "micu_after_first_nf"}

def weights_for_stage(stage: int):
    if stage == 1:
        return {k: (BASE_WEIGHTS[k] if k in VAC_KEYS else 0) for k in BASE_WEIGHTS}
    if stage == 2:
        return {k: (BASE_WEIGHTS[k] if k in VAC_KEYS.union(PGY4_ONLY_KEYS) else 0) for k in BASE_WEIGHTS}
    return BASE_WEIGHTS

STAGE1_TIME = 240
STAGE2_TIME = 360
STAGE3_TIME = 600
NUM_WORKERS = 8
OUTPUT_BASE_DIR = "/content/drive/MyDrive/Colab_Notebook"
RUN_FOLDER_NAME = ""  # Set e.g. "attempt_72"; blank auto-generates a timestamped folder.


def resolve_output_dir(base_dir: str, run_folder_name: str = "") -> str:
    name = run_folder_name.strip()
    if not name:
        name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    return str(Path(base_dir) / name)

def run_three_stage(stage1_time: int, stage2_time: int, stage3_time: int, output_dir: str):
    global WEIGHTS
    start = time.time()
    result = {
        "stage1_time": stage1_time,
        "stage2_time": stage2_time,
        "stage3_time": stage3_time,
    }

    WEIGHTS = weights_for_stage(1)
    stage1_model, stage1_var, F, W, stage1_soft_report = build_model()
    stage1_solver = cp_model.CpSolver()
    stage1_solver.parameters.max_time_in_seconds = stage1_time
    stage1_solver.parameters.num_search_workers = NUM_WORKERS
    stage1_status = stage1_solver.Solve(stage1_model)
    result["stage1_status"] = stage1_solver.StatusName(stage1_status)
    result["stage1_objective"] = stage1_solver.ObjectiveValue()
    print("Stage 1 Status (vacations only):", result["stage1_status"])
    print("Stage 1 Objective:", result["stage1_objective"])

    if stage1_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        result["run_seconds"] = round(time.time() - start, 1)
        return result

    write_reports(stage1_solver, stage1_var, stage1_soft_report, "stage1", output_dir=output_dir)

    fixed_vaca = {}
    for f, _ in enumerate(FELLOWS):
        for w in range(W):
            fixed_vaca[(f, w)] = stage1_solver.Value(stage1_var(f, w, "VACA"))

    WEIGHTS = weights_for_stage(2)
    stage2_model, stage2_var, F, W, stage2_soft_report = build_model()
    for (f, w), val in fixed_vaca.items():
        stage2_model.Add(stage2_var(f, w, "VACA") == val)

    stage2_solver = cp_model.CpSolver()
    stage2_solver.parameters.max_time_in_seconds = stage2_time
    stage2_solver.parameters.num_search_workers = NUM_WORKERS
    stage2_status = stage2_solver.Solve(stage2_model)
    result["stage2_status"] = stage2_solver.StatusName(stage2_status)
    result["stage2_objective"] = stage2_solver.ObjectiveValue()
    print("Stage 2 Status (vacations + PGY-4 soft):", result["stage2_status"])
    print("Stage 2 Objective:", result["stage2_objective"])

    if stage2_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        result["run_seconds"] = round(time.time() - start, 1)
        return result

    write_reports(stage2_solver, stage2_var, stage2_soft_report, "stage2", output_dir=output_dir)

    WEIGHTS = weights_for_stage(3)
    stage3_model, stage3_var, F, W, stage3_soft_report = build_model()
    for (f, w), val in fixed_vaca.items():
        stage3_model.Add(stage3_var(f, w, "VACA") == val)

    # Hint Stage 3 with Stage 2 solution (soft guidance, not a hard lock)
    for f in range(F):
        for w in range(W):
            for rname in ROTATIONS:
                stage3_model.AddHint(
                    stage3_var(f, w, rname),
                    stage2_solver.Value(stage2_var(f, w, rname)),
                )

    stage3_solver = cp_model.CpSolver()
    stage3_solver.parameters.max_time_in_seconds = stage3_time
    stage3_solver.parameters.num_search_workers = NUM_WORKERS
    stage3_status = stage3_solver.Solve(stage3_model)
    result["stage3_status"] = stage3_solver.StatusName(stage3_status)
    result["stage3_objective"] = stage3_solver.ObjectiveValue()
    print("Stage 3 Status (full model):", result["stage3_status"])
    print("Stage 3 Objective:", result["stage3_objective"])

    if stage3_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        write_reports(stage3_solver, stage3_var, stage3_soft_report, "stage3", output_dir=output_dir)
        soft_path = Path(output_dir) / "soft_constraint_report_stage3.csv"
        if soft_path.exists():
            soft_df = pd.read_csv(soft_path)
            result["stage3_total_penalty"] = float(soft_df["penalty"].sum()) if not soft_df.empty else 0.0
        else:
            result["stage3_total_penalty"] = None
    else:
        print("No feasible solution found in Stage 3.")
        result["stage3_total_penalty"] = None

    result["run_seconds"] = round(time.time() - start, 1)
    return result

run_output_dir = resolve_output_dir(OUTPUT_BASE_DIR, RUN_FOLDER_NAME)
print(f"Max times (s): Stage1={STAGE1_TIME}, Stage2={STAGE2_TIME}, Stage3={STAGE3_TIME}")
print(f"Output folder: {run_output_dir}")
result = run_three_stage(STAGE1_TIME, STAGE2_TIME, STAGE3_TIME, run_output_dir)
if result and "run_seconds" in result:
    print(f"Total runtime (s): {result['run_seconds']}")
