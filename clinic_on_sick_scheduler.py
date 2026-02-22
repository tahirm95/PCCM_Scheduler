from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
import re
from typing import Dict, List, Set, Tuple

import pandas as pd
from ortools.sat.python import cp_model


PGY4 = {"DD", "NM", "JD", "BX", "HD", "VK"}
PGY5 = {"PH", "IT", "YA", "SV", "AG"}
PGY6 = {"SL", "YD", "TM", "JB", "AD", "TT"}
CHIEFS = {"TM", "AD"}

START_WEEK = date(2026, 6, 29)
END_WEEK = date(2027, 6, 21)
PGY4_OVERNIGHT_BLOCK_WEEK = date(2026, 10, 26)

W = {
    "hardish_rule": 500,
    "friday_senior_shortfall": 350,
    "consecutive": 300,
    "physio_wed": 80,
    "chief_friday": 60,
    "chief_sick": 60,
    "wed_range": 20,
    "fri_range": 25,
    "overnight_range": 25,
    "sick_range": 25,
    "group_clinic_range": 70,
    "group_overnight_range": 85,
    "group_sick_range": 85,
}


@dataclass
class WeekRow:
    week: date
    physio: str
    clinic_available: Set[str]
    pulm_weekend: Set[str]
    micu_weekend: Set[str]


@dataclass
class ScheduleResult:
    output_df: pd.DataFrame
    console_text: str


def _split_names(raw: str) -> List[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    text = str(raw).strip()
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"[;,|]", text) if p.strip()]
    cleaned: List[str] = []
    for p in parts:
        if p and p not in cleaned:
            cleaned.append(p)
    return cleaned


def _parse_week(value: str) -> date:
    parsed = pd.to_datetime(value).date()
    return parsed


def load_input(path: str) -> List[WeekRow]:
    df = pd.read_csv(path)
    if df.shape[1] < 5:
        raise ValueError("Input CSV must have at least 5 columns (A-E).")

    rows: List[WeekRow] = []
    for _, row in df.iterrows():
        week = _parse_week(row.iloc[0])
        if week < START_WEEK or week > END_WEEK:
            continue

        physio_list = _split_names(row.iloc[1])
        physio = physio_list[0] if physio_list else ""

        clinic_available = set(_split_names(row.iloc[2]))
        if physio:
            clinic_available.add(physio)

        pulm_weekend = set(_split_names(row.iloc[3]))
        micu_weekend = set(_split_names(row.iloc[4]))

        rows.append(
            WeekRow(
                week=week,
                physio=physio,
                clinic_available=clinic_available,
                pulm_weekend=pulm_weekend,
                micu_weekend=micu_weekend,
            )
        )

    rows.sort(key=lambda r: r.week)
    return rows


def _names_for_solver(rows: List[WeekRow]) -> List[str]:
    names: Set[str] = set()
    for r in rows:
        names |= r.clinic_available
        names |= r.pulm_weekend
        names |= r.micu_weekend
        if r.physio:
            names.add(r.physio)
    return sorted(n for n in names if n)


def _pgy(name: str) -> int:
    if name in PGY4:
        return 4
    if name in PGY5:
        return 5
    if name in PGY6:
        return 6
    return 0


def _add_group_range_penalty(model: cp_model.CpModel, penalties, totals_by_fellow: Dict[str, cp_model.IntVar], group_members: List[str], upper_bound: int, label: str, weight: int) -> None:
    if len(group_members) <= 1:
        return
    max_total = model.NewIntVar(0, upper_bound, f"max_{label}")
    min_total = model.NewIntVar(0, upper_bound, f"min_{label}")
    for fellow in group_members:
        total_var = totals_by_fellow[fellow]
        model.Add(max_total >= total_var)
        model.Add(min_total <= total_var)
    rng = model.NewIntVar(0, upper_bound, f"range_{label}")
    model.Add(rng == max_total - min_total)
    penalties.append((rng, weight))


def build_schedule(rows: List[WeekRow]) -> ScheduleResult:
    model = cp_model.CpModel()
    weeks = list(range(len(rows)))
    fellows = _names_for_solver(rows)

    wed = {}
    fri = {}
    ovn = {}
    sick = {}

    for w in weeks:
        avail = rows[w].clinic_available
        for f in fellows:
            avail_flag = f in avail
            wed[(w, f)] = model.NewBoolVar(f"wed_{w}_{f}")
            fri[(w, f)] = model.NewBoolVar(f"fri_{w}_{f}")
            ovn[(w, f)] = model.NewBoolVar(f"ovn_{w}_{f}")
            sick[(w, f)] = model.NewBoolVar(f"sick_{w}_{f}")
            if not avail_flag:
                model.Add(wed[(w, f)] == 0)
                model.Add(fri[(w, f)] == 0)
                model.Add(ovn[(w, f)] == 0)
                model.Add(sick[(w, f)] == 0)

    penalties = []

    for w in weeks:
        avail = rows[w].clinic_available
        avail_n = len(avail)
        wed_need = min(6, avail_n)
        fri_need = min(4, avail_n)

        model.Add(sum(wed[(w, f)] for f in fellows) == wed_need)
        model.Add(sum(fri[(w, f)] for f in fellows) == fri_need)
        model.Add(sum(ovn[(w, f)] for f in fellows) == 1)
        model.Add(sum(sick[(w, f)] for f in fellows) == 1)

        req_senior = min(2, fri_need)
        senior_count = sum(fri[(w, f)] for f in fellows if _pgy(f) in {5, 6})
        shortfall = model.NewIntVar(0, req_senior, f"fri_senior_shortfall_{w}")
        model.Add(shortfall >= req_senior - senior_count)
        penalties.append((shortfall, W["friday_senior_shortfall"]))

        physio = rows[w].physio
        if physio and physio in fellows:
            penalties.append((wed[(w, physio)], W["physio_wed"]))

        weekend_workers = rows[w].pulm_weekend | rows[w].micu_weekend
        for f in fellows:
            # cannot be on sick call if Friday clinic (soft)
            v_sf = model.NewBoolVar(f"viol_sick_fri_{w}_{f}")
            model.Add(v_sf >= sick[(w, f)] + fri[(w, f)] - 1)
            penalties.append((v_sf, W["hardish_rule"]))

            # cannot do overnight/sick on weekend (soft)
            if f in weekend_workers:
                penalties.append((ovn[(w, f)], W["hardish_rule"]))
                penalties.append((sick[(w, f)], W["hardish_rule"]))

            # physio cannot be sick (soft least-bad fallback)
            if rows[w].physio and f == rows[w].physio:
                penalties.append((sick[(w, f)], W["hardish_rule"]))

            # PGY-4 overnight blocked week (soft)
            if rows[w].week == PGY4_OVERNIGHT_BLOCK_WEEK and _pgy(f) == 4:
                penalties.append((ovn[(w, f)], W["hardish_rule"]))

            # Chiefs should do fewer Friday clinics
            if f in CHIEFS:
                penalties.append((fri[(w, f)], W["chief_friday"]))
                penalties.append((sick[(w, f)], W["chief_sick"]))

                # TM/AD do not work overnight call
                model.Add(ovn[(w, f)] == 0)

    for w in weeks[:-1]:
        for f in fellows:
            v_s = model.NewBoolVar(f"viol_sick_consec_{w}_{f}")
            model.Add(v_s >= sick[(w, f)] + sick[(w + 1, f)] - 1)
            penalties.append((v_s, W["consecutive"]))

            v_o = model.NewBoolVar(f"viol_ovn_consec_{w}_{f}")
            model.Add(v_o >= ovn[(w, f)] + ovn[(w + 1, f)] - 1)
            penalties.append((v_o, W["consecutive"]))

    # fairness ranges
    shift_vars = {
        "wed": wed,
        "fri": fri,
        "overnight": ovn,
        "sick": sick,
    }
    range_weights = {
        "wed": W["wed_range"],
        "fri": W["fri_range"],
        "overnight": W["overnight_range"],
        "sick": W["sick_range"],
    }

    totals_by_shift: Dict[str, Dict[str, cp_model.IntVar]] = {}
    for shift, sv in shift_vars.items():
        totals: Dict[str, cp_model.IntVar] = {}
        max_total = model.NewIntVar(0, len(weeks), f"max_{shift}")
        min_total = model.NewIntVar(0, len(weeks), f"min_{shift}")
        for f in fellows:
            t = model.NewIntVar(0, len(weeks), f"total_{shift}_{f}")
            model.Add(t == sum(sv[(w, f)] for w in weeks))
            totals[f] = t
            model.Add(max_total >= t)
            model.Add(min_total <= t)
        rng = model.NewIntVar(0, len(weeks), f"range_{shift}")
        model.Add(rng == max_total - min_total)
        penalties.append((rng, range_weights[shift]))
        totals_by_shift[shift] = totals

    clinic_totals: Dict[str, cp_model.IntVar] = {}
    for f in fellows:
        clinic_total = model.NewIntVar(0, 2 * len(weeks), f"total_clinic_{f}")
        model.Add(clinic_total == totals_by_shift["wed"][f] + totals_by_shift["fri"][f])
        clinic_totals[f] = clinic_total

    pgy_groups = {
        "pgy4": sorted(f for f in fellows if f in PGY4),
        "pgy5": sorted(f for f in fellows if f in PGY5),
        "pgy6_nonchief": sorted(f for f in fellows if f in PGY6 and f not in CHIEFS),
    }

    for group_name, members in pgy_groups.items():
        _add_group_range_penalty(
            model=model,
            penalties=penalties,
            totals_by_fellow=clinic_totals,
            group_members=members,
            upper_bound=2 * len(weeks),
            label=f"{group_name}_clinic",
            weight=W["group_clinic_range"],
        )
        _add_group_range_penalty(
            model=model,
            penalties=penalties,
            totals_by_fellow=totals_by_shift["overnight"],
            group_members=members,
            upper_bound=len(weeks),
            label=f"{group_name}_overnight",
            weight=W["group_overnight_range"],
        )
        _add_group_range_penalty(
            model=model,
            penalties=penalties,
            totals_by_fellow=totals_by_shift["sick"],
            group_members=members,
            upper_bound=len(weeks),
            label=f"{group_name}_sick",
            weight=W["group_sick_range"],
        )

    model.Minimize(sum(var * wt for var, wt in penalties))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 120
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible solution found.")

    out_rows = []
    lines = ["Clinic_ON_Sick schedule\n"]
    for w in weeks:
        week = rows[w].week.isoformat()
        wed_list = [f for f in fellows if solver.Value(wed[(w, f)])]
        fri_list = [f for f in fellows if solver.Value(fri[(w, f)])]
        ovn_name = next(f for f in fellows if solver.Value(ovn[(w, f)]))
        sick_name = next(f for f in fellows if solver.Value(sick[(w, f)]))

        out_rows.append(
            {
                "Week": week,
                "Wednesday Clinic": "; ".join(wed_list),
                "Friday Clinic": "; ".join(fri_list),
                "Overnight": ovn_name,
                "Sick Call": sick_name,
            }
        )

        lines.append(
            f"{week}: Wed=[{', '.join(wed_list)}] | Fri=[{', '.join(fri_list)}] | Overnight={ovn_name} | Sick={sick_name}"
        )

    return ScheduleResult(output_df=pd.DataFrame(out_rows), console_text="\n".join(lines))


def _validate_week_series(rows: List[WeekRow]) -> None:
    if not rows:
        raise ValueError("No valid weeks found in input.")
    week_set = {r.week for r in rows}
    cursor = START_WEEK
    while cursor <= END_WEEK:
        if cursor not in week_set:
            print(f"Warning: missing week in input CSV: {cursor.isoformat()}")
        cursor += timedelta(days=7)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clinic + Overnight + Sick scheduler using OR-Tools")
    parser.add_argument("--input", required=True, help="Input CSV path (A-E columns)")
    parser.add_argument("--output", default="Clinic_ON_Sick.csv", help="Output CSV path")
    args = parser.parse_args()

    rows = load_input(args.input)
    _validate_week_series(rows)
    result = build_schedule(rows)
    result.output_df.to_csv(args.output, index=False)

    print(result.console_text)
    print(f"\nSaved CSV: {args.output}")


if __name__ == "__main__":
    main()
