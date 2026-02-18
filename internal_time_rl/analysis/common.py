from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def bootstrap_ci_mean(
    values: np.ndarray | list[float],
    samples: int,
    seed: int,
    ci_low: float = 2.5,
    ci_high: float = 97.5,
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    n = arr.size
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        v = float(arr[0])
        return v, v

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(samples, n))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, ci_low)), float(np.percentile(means, ci_high))


def ensure_columns(df: pd.DataFrame, cols: Iterable[str], context: str = "dataframe") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {context}: {missing}")


def require_path(path_str: str | Path, *, kind: str = "file") -> Path:
    path = Path(path_str)
    if kind == "file" and not path.is_file():
        raise SystemExit(f"Missing file: {path}")
    if kind == "dir" and not path.is_dir():
        raise SystemExit(f"Missing directory: {path}")
    return path


def save_tex_with_numeric_format(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    non_numeric_cols: Iterable[str] | None = None,
    digits: int = 3,
) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    formatted = df.copy()
    non_numeric = set(non_numeric_cols or [])
    for col in formatted.columns:
        if col in non_numeric:
            continue
        series = pd.to_numeric(formatted[col], errors="coerce")
        if series.notna().any():
            formatted[col] = series.map(lambda x: f"{x:.{digits}f}" if pd.notna(x) else "nan")

    formatted.to_latex(out, index=False, escape=False)
    return out

