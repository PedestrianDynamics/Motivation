"""Aggregate per-seed scenario outputs into band plots and tests.

Reads the per-seed detail CSVs written by the analysis scripts
(``final_rank_details_*``, ``voronoi_density_details_*``,
``coordination_number_details_*``) with tags shaped like
``{scenario}_seed{seed}``, computes one summary scalar per run,
and writes:

* band plots (median + IQR) per observable,
* ``seed_summary.csv`` with one row per (seed, model, scenario, scalar),
* ``tests.csv`` with paired Wilcoxon tests and effect sizes
  (Cliff's delta) comparing PVE vs. BASE_MODEL.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Pattern: {stem}_{scenario}_seed{seed}.csv -> (scenario, seed)
TAG_RE = re.compile(r"^(?P<scenario>.+)_seed(?P<seed>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate per-seed scenario CSVs into bands, scalars, and tests."
    )
    parser.add_argument(
        "--scenarios-dir",
        default=str(PROJECT_ROOT / "files" / "coordination_scenarios"),
        help="Directory holding per-scenario folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "files" / "coordination_scenarios" / "_aggregated"),
        help="Directory for aggregated CSVs and figures.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["BASE_MODEL", "PVE"],
        help="Models to include in the band plots.",
    )
    return parser.parse_args()


def parse_tag(tag: str) -> Tuple[str, int] | None:
    match = TAG_RE.match(tag)
    if not match:
        return None
    return match.group("scenario"), int(match.group("seed"))


def glob_tagged(results_dir: Path, stem: str) -> List[Tuple[str, int, Path]]:
    if not results_dir.exists():
        return []
    found: List[Tuple[str, int, Path]] = []
    for path in results_dir.glob(f"{stem}_*.csv"):
        tag = path.stem[len(stem) + 1 :]
        parsed = parse_tag(tag)
        if parsed is None:
            continue
        scenario, seed = parsed
        found.append((scenario, seed, path))
    return found


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    rx = _rankdata(xs)
    ry = _rankdata(ys)
    return _pearson(rx, ry)


def _rankdata(values: Sequence[float]) -> List[float]:
    ordered = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and values[ordered[j + 1]] == values[ordered[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[ordered[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs)
    dy = sum((y - my) ** 2 for y in ys)
    if dx == 0.0 or dy == 0.0:
        return 0.0
    return num / (dx * dy) ** 0.5


def ols_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0


def tail_ratio(ranks: Sequence[int], areas: Sequence[float]) -> float:
    if not ranks:
        return 0.0
    n = len(ranks)
    q = max(1, n // 4)
    paired = sorted(zip(ranks, areas), key=lambda t: t[0])
    bottom = [a for _, a in paired[:q]]
    top = [a for _, a in paired[-q:]]
    bm = sum(bottom) / len(bottom) if bottom else 0.0
    return (sum(top) / len(top)) / bm if bm else 0.0


# ---------------- Rank-area ----------------

def scalars_from_rank_area(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    by_model: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append((int(row["final_rank"]), float(row["area"])))

    out: Dict[str, Dict[str, float]] = {}
    for model, pairs in by_model.items():
        ranks = [r for r, _ in pairs]
        areas = [a for _, a in pairs]
        out[model] = {
            "spearman_rho": spearman_rho(ranks, areas),
            "slope": ols_slope(ranks, areas),
            "tail_ratio": tail_ratio(ranks, areas),
        }
    return out


# ---------------- Voronoi density ----------------

def scalars_from_voronoi(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    by_model: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(
            (float(row["time"]), float(row["voronoi_density"]))
        )

    out: Dict[str, Dict[str, float]] = {}
    for model, pairs in by_model.items():
        densities = [d for _, d in pairs]
        if not densities:
            continue
        mean = sum(densities) / len(densities)
        var = sum((d - mean) ** 2 for d in densities) / len(densities)
        threshold = 3.0
        frac_above = sum(1 for d in densities if d > threshold) / len(densities)
        out[model] = {
            "mean_density": mean,
            "density_std": var ** 0.5,
            "frac_time_above_3": frac_above,
        }
    return out


# ---------------- Coordination number ----------------

def scalars_from_coordination(
    rows: List[Dict[str, str]],
) -> Dict[str, Dict[str, float]]:
    by_model: Dict[str, List[int]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(int(row["coordination_number"]))

    out: Dict[str, Dict[str, float]] = {}
    for model, values in by_model.items():
        if not values:
            continue
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        mode = _mode(values)
        out[model] = {
            "mean_nn": mean,
            "nn_std": var ** 0.5,
            "mode_nn": float(mode),
        }
    return out


def _mode(values: Sequence[int]) -> int:
    counts: Dict[int, int] = defaultdict(int)
    for v in values:
        counts[v] += 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


# ---------------- Statistical tests ----------------

def wilcoxon_signed_rank(diffs: Sequence[float]) -> Tuple[float, float]:
    """Return (W, two-sided p-value) via normal approximation."""
    nonzero = [d for d in diffs if d != 0.0]
    n = len(nonzero)
    if n < 6:
        return 0.0, float("nan")
    abs_vals = [abs(d) for d in nonzero]
    ranks = _rankdata(abs_vals)
    w_plus = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, nonzero) if d < 0)
    w = min(w_plus, w_minus)
    mean = n * (n + 1) / 4.0
    sd = (n * (n + 1) * (2 * n + 1) / 24.0) ** 0.5
    if sd == 0:
        return w, float("nan")
    z = (w - mean) / sd
    p = 2.0 * _normal_sf(abs(z))
    return w, p


def _normal_sf(z: float) -> float:
    # Abramowitz-Stegun 26.2.17 approximation
    import math

    t = 1.0 / (1.0 + 0.2316419 * z)
    d = math.exp(-(z * z) / 2.0) / math.sqrt(2.0 * math.pi)
    poly = (
        0.319381530 * t
        - 0.356563782 * t**2
        + 1.781477937 * t**3
        - 1.821255978 * t**4
        + 1.330274429 * t**5
    )
    return d * poly


def cliffs_delta(xs: Sequence[float], ys: Sequence[float]) -> float:
    if not xs or not ys:
        return 0.0
    greater = sum(1 for x in xs for y in ys if x > y)
    less = sum(1 for x in xs for y in ys if x < y)
    return (greater - less) / (len(xs) * len(ys))


# ---------------- Aggregation ----------------

def collect_summary(
    scenarios_dir: Path,
) -> Tuple[
    List[Dict[str, object]],
    Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]],
]:
    """Walk scenario dirs, compute per-seed scalars, return rows + raw curves."""
    summary_rows: List[Dict[str, object]] = []
    # Raw curves for band plots: observable -> scenario -> model -> {axis_key -> list of lists}
    curves: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )

    for scenario_path in sorted(scenarios_dir.iterdir()):
        if not scenario_path.is_dir() or scenario_path.name.startswith("_"):
            continue

        _collect_rank_area(scenario_path, summary_rows, curves)
        _collect_voronoi(scenario_path, summary_rows, curves)
        _collect_coordination(scenario_path, summary_rows, curves)

    return summary_rows, curves


def _collect_rank_area(scenario_path, summary_rows, curves) -> None:
    results_dir = scenario_path / "final_rank_results"
    for scenario, seed, csv_path in glob_tagged(results_dir, "final_rank_details"):
        rows = read_csv(csv_path)
        for model, scalars in scalars_from_rank_area(rows).items():
            for name, value in scalars.items():
                summary_rows.append(
                    {
                        "observable": "rank_area",
                        "scenario": scenario,
                        "seed": seed,
                        "model": model,
                        "scalar": name,
                        "value": value,
                    }
                )
        by_model: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        for row in rows:
            by_model[row["model"]].append(
                (int(row["final_rank"]), float(row["area"]))
            )
        for model, pairs in by_model.items():
            ranks = [r for r, _ in pairs]
            areas = [a for _, a in pairs]
            curves["rank_area"][scenario][model]["rank"].append(ranks)
            curves["rank_area"][scenario][model]["value"].append(areas)


def _collect_voronoi(scenario_path, summary_rows, curves) -> None:
    results_dir = scenario_path / "voronoi_density_results"
    for scenario, seed, csv_path in glob_tagged(results_dir, "voronoi_density_details"):
        rows = read_csv(csv_path)
        for model, scalars in scalars_from_voronoi(rows).items():
            for name, value in scalars.items():
                summary_rows.append(
                    {
                        "observable": "voronoi_density",
                        "scenario": scenario,
                        "seed": seed,
                        "model": model,
                        "scalar": name,
                        "value": value,
                    }
                )
        by_model: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        for row in rows:
            by_model[row["model"]].append(
                (float(row["time"]), float(row["voronoi_density"]))
            )
        for model, pairs in by_model.items():
            times = [t for t, _ in pairs]
            dens = [d for _, d in pairs]
            curves["voronoi_density"][scenario][model]["rank"].append(times)
            curves["voronoi_density"][scenario][model]["value"].append(dens)


def _collect_coordination(scenario_path, summary_rows, curves) -> None:
    results_dir = scenario_path / "coordination_number_results"
    for scenario, seed, csv_path in glob_tagged(
        results_dir, "coordination_number_details"
    ):
        rows = read_csv(csv_path)
        for model, scalars in scalars_from_coordination(rows).items():
            for name, value in scalars.items():
                summary_rows.append(
                    {
                        "observable": "coordination_number",
                        "scenario": scenario,
                        "seed": seed,
                        "model": model,
                        "scalar": name,
                        "value": value,
                    }
                )
        # Per-seed histogram as a "curve" over coordination number buckets
        by_model: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for row in rows:
            by_model[row["model"]][int(row["coordination_number"])] += 1
        for model, bucket in by_model.items():
            total = sum(bucket.values()) or 1
            keys = sorted(bucket.keys())
            probs = [bucket[k] / total for k in keys]
            curves["coordination_number"][scenario][model]["rank"].append(
                [float(k) for k in keys]
            )
            curves["coordination_number"][scenario][model]["value"].append(probs)


# ---------------- Paired tests ----------------

def compute_tests(
    summary_rows: List[Dict[str, object]],
    models: Sequence[str] = ("BASE_MODEL", "PVE"),
) -> List[Dict[str, object]]:
    model_a, model_b = models[0], models[1]
    grouped: Dict[Tuple[str, str, str], Dict[str, Dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in summary_rows:
        key = (str(row["observable"]), str(row["scenario"]), str(row["scalar"]))
        seed_value = int(str(row["seed"]))
        value = float(str(row["value"]))
        grouped[key][str(row["model"])][seed_value] = value

    rows: List[Dict[str, object]] = []
    for (observable, scenario, scalar), by_model in grouped.items():
        a = by_model.get(model_a, {})
        b = by_model.get(model_b, {})
        shared = sorted(set(a.keys()) & set(b.keys()))
        if len(shared) < 3:
            continue
        values_a = [a[s] for s in shared]
        values_b = [b[s] for s in shared]
        diffs = [y - x for x, y in zip(values_a, values_b)]
        _, p_value = wilcoxon_signed_rank(diffs)
        delta = cliffs_delta(values_b, values_a)
        rows.append(
            {
                "observable": observable,
                "scenario": scenario,
                "scalar": scalar,
                "model_A": model_a,
                "model_B": model_b,
                "n_pairs": len(shared),
                "median_A": median(values_a),
                "median_B": median(values_b),
                "median_diff": median(diffs),
                "wilcoxon_p": p_value,
                "cliffs_delta": delta,
                "paired": True,
            }
        )
    return rows


# ---------------- Band plots ----------------

def plot_bands(
    curves,
    output_dir: Path,
    models: Sequence[str],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib not available; skipping band plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for observable, by_scenario in curves.items():
        for scenario, by_model in by_scenario.items():
            fig, ax = plt.subplots(figsize=(7, 4))
            for model in models:
                seeds_data = by_model.get(model)
                if not seeds_data:
                    continue
                rank_lists = seeds_data["rank"]
                value_lists = seeds_data["value"]
                xs, med, lo, hi = _median_iqr_band(rank_lists, value_lists)
                if not xs:
                    continue
                line_style = "o-" if observable == "coordination_number" else "-"
                ax.plot(
                    xs,
                    med,
                    line_style,
                    label=f"{model} (median)",
                    linewidth=2,
                    markersize=6,
                )
                ax.fill_between(xs, lo, hi, alpha=0.2, label=f"{model} IQR")
            ax.set_xlabel(_xlabel_for(observable))
            ax.set_ylabel(_ylabel_for(observable))
            ax.grid(alpha=0.2)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / f"{observable}_band_{scenario}.png", dpi=200)
            plt.close(fig)


def _median_iqr_band(
    rank_lists: List[List[float]], value_lists: List[List[float]]
) -> Tuple[List[float], List[float], List[float], List[float]]:
    if not rank_lists:
        return [], [], [], []
    # Resample each run onto a shared grid (union of observed x values)
    shared_x = sorted({float(x) for xs in rank_lists for x in xs})
    if not shared_x:
        return [], [], [], []
    by_x: Dict[float, List[float]] = defaultdict(list)
    for xs, ys in zip(rank_lists, value_lists):
        for x, y in zip(xs, ys):
            by_x[float(x)].append(float(y))
    med: List[float] = []
    lo: List[float] = []
    hi: List[float] = []
    kept_x: List[float] = []
    for x in shared_x:
        values = sorted(by_x[x])
        if len(values) < 2:
            continue
        kept_x.append(x)
        med.append(_percentile(values, 50))
        lo.append(_percentile(values, 25))
        hi.append(_percentile(values, 75))
    return kept_x, med, lo, hi


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (q / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def _xlabel_for(observable: str) -> str:
    return {
        "rank_area": "final rank",
        "voronoi_density": "time [s]",
        "coordination_number": "coordination number",
    }.get(observable, "x")


def _ylabel_for(observable: str) -> str:
    return {
        "rank_area": r"Voronoi area [$\mathrm{m}^2$]",
        "voronoi_density": r"Voronoi density [$\mathrm{m}^{-2}$]",
        "coordination_number": "probability",
    }.get(observable, "y")


# ---------------- IO helpers ----------------

def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    scenarios_dir = Path(args.scenarios_dir)
    output_dir = Path(args.output_dir)

    summary_rows, curves = collect_summary(scenarios_dir)
    write_csv(
        output_dir / "seed_summary.csv",
        summary_rows,
        ["observable", "scenario", "seed", "model", "scalar", "value"],
    )

    test_rows = compute_tests(summary_rows, args.models)
    write_csv(
        output_dir / "tests.csv",
        test_rows,
        [
            "observable",
            "scenario",
            "scalar",
            "model_A",
            "model_B",
            "n_pairs",
            "median_A",
            "median_B",
            "median_diff",
            "wilcoxon_p",
            "cliffs_delta",
            "paired",
        ],
    )

    plot_bands(curves, output_dir, args.models)
    print(f"Wrote aggregated outputs to {output_dir}")


if __name__ == "__main__":
    main()
