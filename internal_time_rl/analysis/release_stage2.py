from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from internal_time_rl.analysis.common import require_path


@dataclass(frozen=True)
class ReleaseRoots:
    warmup_delay10_root: str = "runs/sweeps/stage2_delay_delaybiased_warmup_both20k_v1"
    warmup_sweep_root: str = "runs/sweeps/stage2_delay_sweep_warmup_both20k_v1"
    nonwarmup_root: str = "runs/sweeps/stage2_delay_delaybiased_ls3e-1"
    flicker_root: str = "runs/sweeps/stage2_delay_delaybiased_warmup_both20k_flicker01_v1"


def _run(cmd: list[str], *, dry_run: bool = False) -> None:
    print("RUN:", " ".join(shlex.quote(part) for part in cmd))
    if dry_run:
        return
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def _run_module(module: str, *args: str, dry_run: bool = False) -> None:
    _run([sys.executable, "-m", module, *args], dry_run=dry_run)


def _copy(src: Path, dst: Path, *, dry_run: bool = False) -> None:
    print(f"COPY: {src} -> {dst}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage-2 release orchestration (reproduce + verify)."
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    reproduce = subparsers.add_parser("reproduce", help="Regenerate release artifacts from existing runs.")
    reproduce.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    reproduce.add_argument("--skip-flicker", action="store_true", help="Skip flicker aggregation and paired output.")
    reproduce.add_argument(
        "--warmup-delay10-root",
        type=str,
        default=ReleaseRoots.warmup_delay10_root,
        help="Root for delay=10 warmup runs.",
    )
    reproduce.add_argument(
        "--warmup-sweep-root",
        type=str,
        default=ReleaseRoots.warmup_sweep_root,
        help="Root for warmup sweep delay folders.",
    )
    reproduce.add_argument(
        "--nonwarmup-root",
        type=str,
        default=ReleaseRoots.nonwarmup_root,
        help="Root for non-warmup causal runs.",
    )
    reproduce.add_argument(
        "--flicker-root",
        type=str,
        default=ReleaseRoots.flicker_root,
        help="Root for flicker robustness runs.",
    )

    verify = subparsers.add_parser("verify", help="Verify release guard conditions from docs artifacts.")
    verify.add_argument("--min-warmup-pairs", type=int, default=30)
    verify.add_argument("--min-nonwarmup-pairs", type=int, default=15)
    verify.add_argument("--min-flicker-pairs", type=int, default=5)
    verify.add_argument("--min-groupdiff-seeds", type=int, default=30)
    return parser.parse_args()


def run_reproduce(args: argparse.Namespace) -> None:
    warmup_delay10_root = require_path(args.warmup_delay10_root, kind="dir")
    warmup_sweep_root = require_path(args.warmup_sweep_root, kind="dir")
    nonwarmup_root = require_path(args.nonwarmup_root, kind="dir")
    flicker_root = Path(args.flicker_root)

    _run([sys.executable, "-V"], dry_run=args.dry_run)
    _run([sys.executable, "-m", "compileall", "internal_time_rl"], dry_run=args.dry_run)

    # Non-warmup causal evidence
    _run_module(
        "internal_time_rl.analysis.aggregate_runs",
        "--root",
        str(nonwarmup_root),
        "--metric",
        "episode/return_mean_20",
        "--last-n",
        "10",
        "--points",
        "200",
        dry_run=args.dry_run,
    )
    _run_module(
        "internal_time_rl.analysis.paired_effects",
        "--input",
        str(nonwarmup_root / "aggregate" / "per_run_summary.csv"),
        "--anchor",
        "learned_tau_delay10_selfmodel",
        "--comparators",
        "learned_tau_delay10",
        "learned_tau_delay10_selfmodel_noerr",
        "--metrics",
        "final_mean",
        "auc",
        "--min-n-pairs",
        "15",
        "--out-csv",
        "docs/stage2_delay_paired_effects_2026-02-16.csv",
        "--out-per-seed",
        "docs/stage2_delay_paired_effects_seeds_2026-02-16.csv",
        "--out-tex",
        "docs/stage2_delay_paired_effects_2026-02-16.tex",
        dry_run=args.dry_run,
    )

    # Warmup delay=10 main result
    _run_module(
        "internal_time_rl.analysis.aggregate_runs",
        "--root",
        str(warmup_delay10_root),
        "--metric",
        "episode/return_mean_20",
        "--last-n",
        "10",
        "--points",
        "200",
        dry_run=args.dry_run,
    )
    _run_module(
        "internal_time_rl.analysis.paired_effects",
        "--input",
        str(warmup_delay10_root / "aggregate" / "per_run_summary.csv"),
        "--anchor",
        "learned_tau_delay10_selfmodel",
        "--comparators",
        "learned_tau_delay10",
        "learned_tau_delay10_selfmodel_noerr",
        "--metrics",
        "final_mean",
        "auc",
        "--min-n-pairs",
        "30",
        "--out-csv",
        "docs/stage2_delay_warmup_both20k_paired_2026-02-16.csv",
        "--out-per-seed",
        "docs/stage2_delay_warmup_both20k_paired_seeds_2026-02-16.csv",
        "--out-tex",
        "docs/stage2_delay_warmup_both20k_paired_2026-02-16.tex",
        dry_run=args.dry_run,
    )

    # Import delay10 aggregate into warmup sweep root.
    _copy(
        warmup_delay10_root / "aggregate" / "condition_summary.csv",
        warmup_sweep_root / "delay10" / "aggregate" / "condition_summary.csv",
        dry_run=args.dry_run,
    )
    _copy(
        warmup_delay10_root / "aggregate" / "per_run_summary.csv",
        warmup_sweep_root / "delay10" / "aggregate" / "per_run_summary.csv",
        dry_run=args.dry_run,
    )

    # Warmup sweep summary and plots.
    _run_module(
        "internal_time_rl.analysis.summarize_stage2_delay_sweep",
        "--base-root",
        str(warmup_sweep_root),
        "--delays",
        "0",
        "10",
        "20",
        "--baseline-condition",
        "learned_tau_delay10",
        "--selfmodel-condition",
        "learned_tau_delay10_selfmodel",
        "--out-csv",
        "docs/stage2_delay_sweep_warmup_both20k_summary_2026-02-16.csv",
        "--out-tex",
        "docs/stage2_delay_sweep_warmup_both20k_summary_2026-02-16.tex",
        dry_run=args.dry_run,
    )
    _run_module(
        "internal_time_rl.analysis.plot_stage2_delay_sweep",
        "--input",
        "docs/stage2_delay_sweep_warmup_both20k_summary_2026-02-16.csv",
        "--out-final",
        "docs/figures/fig_stage2_delay_sweep_warmup_both20k_final.png",
        "--out-delta",
        "docs/figures/fig_stage2_delay_sweep_warmup_both20k_paired_delta.png",
        dry_run=args.dry_run,
    )

    _run_module(
        "internal_time_rl.analysis.stage2_delay_group_difference",
        "--base-root",
        str(warmup_sweep_root),
        "--delays",
        "0",
        "10",
        "20",
        "--low-delays",
        "0",
        "--high-delays",
        "10",
        "20",
        "--baseline-condition",
        "learned_tau_delay10",
        "--selfmodel-condition",
        "learned_tau_delay10_selfmodel",
        "--metrics",
        "final_mean",
        "auc",
        "--out-csv",
        "docs/stage2_delay_warmup_groupdiff_0_vs_10_20_2026-02-16.csv",
        "--out-tex",
        "docs/stage2_delay_warmup_groupdiff_0_vs_10_20_2026-02-16.tex",
        "--out-per-seed",
        "docs/stage2_delay_warmup_groupdiff_0_vs_10_20_per_seed_2026-02-16.csv",
        dry_run=args.dry_run,
    )

    # Flicker robustness.
    if args.skip_flicker:
        print("SKIP: flicker aggregation")
    elif not flicker_root.exists():
        print(f"NOTE: flicker root not found, skipping: {flicker_root}")
    else:
        _run_module(
            "internal_time_rl.analysis.aggregate_runs",
            "--root",
            str(flicker_root),
            "--metric",
            "episode/return_mean_20",
            "--last-n",
            "10",
            "--points",
            "200",
            dry_run=args.dry_run,
        )
        _run_module(
            "internal_time_rl.analysis.paired_effects",
            "--input",
            str(flicker_root / "aggregate" / "per_run_summary.csv"),
            "--anchor",
            "learned_tau_delay10_selfmodel",
            "--comparators",
            "learned_tau_delay10",
            "--metrics",
            "final_mean",
            "auc",
            "--min-n-pairs",
            "5",
            "--out-csv",
            "docs/stage2_delay_warmup_flicker01_paired_2026-02-16.csv",
            "--out-per-seed",
            "docs/stage2_delay_warmup_flicker01_paired_seeds_2026-02-16.csv",
            "--out-tex",
            "docs/stage2_delay_warmup_flicker01_paired_2026-02-16.tex",
            dry_run=args.dry_run,
        )
        _copy(
            flicker_root / "aggregate" / "learning_curves.png",
            Path("docs/figures/fig_stage2_delay_warmup_flicker01_learning_curves.png"),
            dry_run=args.dry_run,
        )
        _copy(
            flicker_root / "aggregate" / "final_performance.png",
            Path("docs/figures/fig_stage2_delay_warmup_flicker01_final.png"),
            dry_run=args.dry_run,
        )

    _run_module(
        "internal_time_rl.analysis.export_stage2_main_table",
        "--out-csv",
        "docs/stage2_main_table_2026-02-16.csv",
        "--out-tex",
        "docs/stage2_main_table_2026-02-16.tex",
        dry_run=args.dry_run,
    )

    print("Reproduce completed.")


def _read_single(df: pd.DataFrame, query: pd.Series, message: str) -> pd.Series:
    rows = df.loc[query]
    if rows.empty:
        raise SystemExit(message)
    return rows.iloc[0]


def run_verify(args: argparse.Namespace) -> None:
    _run([sys.executable, "-V"])
    _run([sys.executable, "-m", "compileall", "internal_time_rl"])

    summary_doc = require_path("docs/stage2_results_2026-02-16.md", kind="file")
    head = "\n".join(summary_doc.read_text(encoding="utf-8").splitlines()[:60])
    missing = [key for key in ("Fixed:", "Trend:", "Limitation:") if key not in head]
    if missing:
        raise SystemExit(f"Missing release summary keys in document head: {missing}")

    warmup = pd.read_csv(require_path("docs/stage2_delay_warmup_both20k_paired_2026-02-16.csv"))
    warmup_row = _read_single(
        warmup,
        (warmup["comparator"] == "learned_tau_delay10") & (warmup["metric"] == "final_mean"),
        "Warmup fixed row not found in docs/stage2_delay_warmup_both20k_paired_2026-02-16.csv",
    )
    if int(warmup_row["n_pairs"]) < int(args.min_warmup_pairs):
        raise SystemExit(f"warmup n_pairs too small: {int(warmup_row['n_pairs'])}")
    if float(warmup_row["ci95_low"]) <= 0.0:
        raise SystemExit(f"warmup fixed claim failed: ci95_low={float(warmup_row['ci95_low']):.6f}")

    nonwarmup = pd.read_csv(require_path("docs/stage2_delay_paired_effects_2026-02-16.csv"))
    nonwarmup_row = _read_single(
        nonwarmup,
        (nonwarmup["comparator"] == "learned_tau_delay10_selfmodel_noerr")
        & (nonwarmup["metric"] == "final_mean"),
        "Non-warmup causal row not found in docs/stage2_delay_paired_effects_2026-02-16.csv",
    )
    if int(nonwarmup_row["n_pairs"]) < int(args.min_nonwarmup_pairs):
        raise SystemExit(f"non-warmup n_pairs too small: {int(nonwarmup_row['n_pairs'])}")
    if float(nonwarmup_row["ci95_low"]) <= 0.0:
        raise SystemExit(
            f"non-warmup causal claim failed: ci95_low={float(nonwarmup_row['ci95_low']):.6f}"
        )

    groupdiff = pd.read_csv(require_path("docs/stage2_delay_warmup_groupdiff_0_vs_10_20_2026-02-16.csv"))
    group_row = _read_single(
        groupdiff,
        groupdiff["metric"] == "final_mean",
        "Groupdiff final_mean row not found in docs/stage2_delay_warmup_groupdiff_0_vs_10_20_2026-02-16.csv",
    )
    if int(group_row["n_seeds"]) < int(args.min_groupdiff_seeds):
        raise SystemExit(f"groupdiff n_seeds too small: {int(group_row['n_seeds'])}")

    flicker = pd.read_csv(require_path("docs/stage2_delay_warmup_flicker01_paired_2026-02-16.csv"))
    flicker_row = _read_single(
        flicker,
        (flicker["comparator"] == "learned_tau_delay10") & (flicker["metric"] == "auc"),
        "Flicker AUC row not found in docs/stage2_delay_warmup_flicker01_paired_2026-02-16.csv",
    )
    if int(flicker_row["n_pairs"]) < int(args.min_flicker_pairs):
        raise SystemExit(f"flicker n_pairs too small: {int(flicker_row['n_pairs'])}")
    if float(flicker_row["ci95_high"]) >= 0.0:
        raise SystemExit(
            f"flicker limitation check failed: ci95_high={float(flicker_row['ci95_high']):.6f}"
        )

    print("Verify OK.")


def main() -> None:
    args = parse_args()
    if args.cmd == "reproduce":
        run_reproduce(args)
        return
    if args.cmd == "verify":
        run_verify(args)
        return
    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
