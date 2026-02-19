#!/usr/bin/env python3
"""Calculate maximum penalty for the scheduler objective.

Modes
-----
1) exact: maximize the full-weight model objective via CP-SAT.
2) bound: compute a fast theoretical upper bound from objective variable domains.

This script is intentionally standalone (does not import fellowship_scheduler_2026_02_17.py).
It loads and executes that file's definitions (excluding its bottom run block) in an isolated namespace.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

from ortools.sat.python import cp_model

SCHEDULER_FILE = Path(__file__).with_name("fellowship_scheduler_2026_02_17.py")

# In-file defaults (used when CLI flags are omitted).
DEFAULT_MODE = "bound"
DEFAULT_TIME_LIMIT = 600
DEFAULT_WORKERS = 8
DEFAULT_OUTPUT_DIR = "max_penalty_outputs"


def _load_scheduler_namespace(maximize: bool) -> dict:
    src = SCHEDULER_FILE.read_text()
    src = src.replace(
        "import pandas as pd",
        """
try:
    import pandas as pd
except ModuleNotFoundError:
    import csv

    class _MiniDataFrame:
        def __init__(self, rows):
            self._rows = list(rows)

        def to_csv(self, path, index=False):
            if not self._rows:
                with open(path, "w", newline="") as f:
                    f.write("")
                return
            fieldnames = list(self._rows[0].keys())
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(self._rows)

        def head(self, n=5):
            return self._rows[:n]

        def __str__(self):
            return str(self._rows)

    class _MiniPandas:
        @staticmethod
        def DataFrame(rows):
            return _MiniDataFrame(rows)

    pd = _MiniPandas()
"""
    )
    sentinel = "run_output_dir = resolve_output_dir"
    if sentinel in src:
        src = src.split(sentinel)[0]
    if maximize:
        src = src.replace("model.Minimize(sum(objective_terms))", "model.Maximize(sum(objective_terms))")
    ns: dict = {}
    exec(compile(src, str(SCHEDULER_FILE), "exec"), ns, ns)
    return ns


def _ensure_output_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _constraint_bucket(var_name: str) -> str:
    prefixes = [
        ("clinic_underfill_ge_", "clinic_underfill"),
        ("rsch_exact3_", "rsch_exact3"),
        ("pgy4_pulm_pair_pen_", "pulm_pair_pgy4"),
        ("pgy4_micu_pair_pen_", "micu_pair_pgy4"),
        ("hard3_", "hard_3_run"),
        ("hard5_", "hard_5_run"),
        ("post_nf_medium_", "post_nf_medium"),
        ("post_nf_hard_", "post_nf_hard"),
        ("rif_when_min_", "rif_when_rads_min"),
        ("non_sinai", "non_sinai_pgy4_micu_early"),
        ("pgy4_preim_medium", "pgy4_pre_im_boards_medium"),
        ("pgy4_preim_hard", "pgy4_pre_im_boards_hard"),
        ("pgy6_pre_boards_pen_", "pgy6_pre_boards_rsch"),
        ("pgy6_last8_pen_", "pgy6_last8_rsch_vaca"),
        ("micu_after_nf_", "micu_after_first_nf"),
        ("vac_", "vacation"),
        ("fair_", "fairness"),
    ]
    for pref, bucket in prefixes:
        if var_name.startswith(pref) or pref in var_name:
            return bucket
    if var_name.startswith("x_f"):
        return "assignment_linear_terms"
    return "other"


def run_exact(time_limit: int, workers: int, out_dir: Path) -> Tuple[str, float, Path]:
    ns = _load_scheduler_namespace(maximize=True)
    ns["WEIGHTS"] = ns["BASE_WEIGHTS"].copy()
    model, var, _, _, soft_report = ns["build_model"]()

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    status = solver.Solve(model)
    status_name = solver.StatusName(status)
    objective = solver.ObjectiveValue()

    suffix = "maxpen_exact"
    ns["write_reports"](solver, var, soft_report, suffix, output_dir=str(out_dir))
    soft_path = out_dir / f"soft_constraint_report_{suffix}.csv"

    # Aggregate per-constraint breakdown from produced report.
    breakdown_rows = []
    total_penalty = 0.0
    if soft_path.exists() and soft_path.stat().st_size > 0:
        by_constraint = defaultdict(float)
        with soft_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    by_constraint[row.get("constraint", "")] += float(row.get("penalty", 0) or 0)
                except ValueError:
                    continue
        for c, val in by_constraint.items():
            total_penalty += val
            breakdown_rows.append((c, val))

    breakdown_csv = out_dir / "max_penalty_exact_breakdown.csv"
    with breakdown_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["constraint", "penalty_sum"])
        for c, p in sorted(breakdown_rows):
            writer.writerow([c, p])
        writer.writerow(["TOTAL_FROM_SOFT_REPORT", total_penalty])
        writer.writerow(["OBJECTIVE_VALUE", objective])
        writer.writerow(["STATUS", status_name])

    print(f"Mode: exact")
    print(f"Status: {status_name}")
    print(f"Best objective (maximize): {objective}")
    print(f"Per-constraint CSV: {breakdown_csv}")
    return status_name, objective, breakdown_csv


def run_bound(out_dir: Path) -> Path:
    ns = _load_scheduler_namespace(maximize=False)
    ns["WEIGHTS"] = ns["BASE_WEIGHTS"].copy()
    model, _, _, _, _ = ns["build_model"]()
    proto = model.Proto()

    contrib = defaultdict(float)
    total = 0.0

    for var_idx, coeff in zip(proto.objective.vars, proto.objective.coeffs):
        v = proto.variables[var_idx]
        dom = list(v.domain)
        # CP-SAT domains are [l1,u1,l2,u2,...]
        min_val = min(dom[::2])
        max_val = max(dom[1::2])
        chosen = max_val if coeff >= 0 else min_val
        term_val = float(coeff) * float(chosen)
        bucket = _constraint_bucket(v.name)
        contrib[bucket] += term_val
        total += term_val

    breakdown_csv = out_dir / "max_penalty_bound_breakdown.csv"
    with breakdown_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["constraint_bucket", "upper_bound_contribution"])
        for k in sorted(contrib):
            writer.writerow([k, contrib[k]])
        writer.writerow(["TOTAL_BOUND", total])

    print("Mode: bound")
    print("Method: objective coefficient × variable-domain extrema (fast upper bound)")
    print(f"Upper bound total: {total}")
    print(f"Per-constraint bucket CSV: {breakdown_csv}")
    return breakdown_csv


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Maximum penalty calculator")
    parser.add_argument("--mode", choices=["exact", "bound"], default=DEFAULT_MODE)
    parser.add_argument("--time-limit", type=int, default=DEFAULT_TIME_LIMIT, help="Exact mode: solve time limit in seconds")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Exact mode: CP-SAT workers")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    _ensure_output_dir(out_dir)

    if args.mode == "exact":
        run_exact(args.time_limit, args.workers, out_dir)
    else:
        run_bound(out_dir)


if __name__ == "__main__":
    main()
