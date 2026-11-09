"""LaTeX table of the CROMA replica: fitted nM and held-out hM measures per model.

Reads the selected points of the low-anchor fit (selected_<mode>.csv, run
means over seeds) and the experimental values (croma_waiting.csv), and
writes, per condition and measure, the range over the two blank runs.

  python scripts/croma_results_table.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS = {"nM": ["1C060", "2C020"], "hM": ["2C070", "2C120"]}
MODES = [("BASE_MODEL", "Static"), ("V", "Fixed value"), ("PVE", "EVP")]
MEASURES = [
    ("depth_med", "Crowd depth [m]", "{:.1f}"),
    ("frac_front3m", "Front fraction", "{:.2f}"),
    ("discharge_60s", "Discharge in 60\\,s", "{:.0f}"),
    ("rho_entry_rank", "Entry order vs.\\ rank $\\rho$", "{:.2f}"),
    ("rho_rank_area", "Rank vs.\\ area $\\rho$", "{:.2f}"),
]
FITTED = {"depth_med", "frac_front3m", "discharge_60s"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fit-dir", type=Path, default=PROJECT_ROOT / "build" / "croma_anchor_fit")
    parser.add_argument("--experiment", type=Path, default=PROJECT_ROOT / "build" / "croma_waiting.csv")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT.parent / "motivation-for-springer" / "figures" / "croma_heldout_table.tex")
    return parser.parse_args()


def span(values: pd.Series, fmt: str) -> str:
    lo, hi = (fmt.format(v).replace("-", "$-$") for v in (values.min(), values.max()))
    if lo == hi:
        return lo
    return f"{lo} to {hi}" if lo.startswith("$-$") else f"{lo}--{hi}"


def main() -> None:
    args = parse_args()
    exp = pd.read_csv(args.experiment).set_index("run")
    sel = {mode: pd.read_csv(args.fit_dir / f"selected_{mode}.csv").set_index("run") for mode, _ in MODES}
    lines = [
        "\\begin{tabular}{@{}llcccc@{}}",
        "\\toprule",
        " & & Static & Fixed value & EVP & Experiment \\\\",
        "\\midrule",
    ]
    for cond, label in [("nM", "nM (fitted)"), ("hM", "hM (predicted)")]:
        lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{label}}}}} \\\\")
        for key, name, fmt in MEASURES:
            mark = "" if key in FITTED else "$^{\\dagger}$"
            cells = [span(sel[mode].loc[RUNS[cond], key], fmt) for mode, _ in MODES]
            cells.append(span(exp.loc[RUNS[cond], key], fmt))
            lines.append(f" & {name}{mark} & " + " & ".join(cells) + " \\\\")
        lines.append("\\addlinespace")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
