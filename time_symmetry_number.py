"""Time only the symmetry-number calculation for every molecule in kinbot.csv.

StereoMolGraph construction (and RDKit parsing) is intentionally excluded from
the timing: all graphs are built up front, then a warm-up pass is run, and only
the `symmetry_number(graph)` call is measured per molecule.

Usage:
    pixi run -e smg python experiments/SI_SymmetryNumbers/time_symmetry_number.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from rdkit import Chem
from stereomolgraph import StereoMolGraph
from stereomolgraph.algorithms.symmetry import symmetry_number

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "data" / "kinbot.csv"


def build_graphs() -> list[tuple[str, object]]:
    """Parse kinbot.csv and build one StereoMolGraph per valid molecule.

    Returns a list of (inchi, graph) pairs; unparseable InChIs are skipped.
    Graph construction is *not* timed.
    """
    df = pd.read_csv(CSV_PATH, escapechar="\\")

    graphs: list[tuple[str, object]] = []
    for inchi in df["inchi"]:
        rdmol = Chem.MolFromInchi(
            inchi, sanitize=True, removeHs=False, treatWarningAsError=True
        )
        if rdmol is not None:
            rdmol = Chem.AddHs(rdmol)
        if rdmol is None:
            continue

        graph = StereoMolGraph.from_rdmol(
            rdmol, stereo_complete=True, resonance=True, lone_pair_stereo=True
        )
        graphs.append((inchi, graph))

    return graphs


def time_symmetry_number(
    graphs: list[tuple[str, object]], repeats: int
) -> pd.DataFrame:
    """Measure the best-of-`repeats` wall time for `symmetry_number(graph)`.

    A warm-up pass over all graphs runs first so lazy one-time costs are not
    included in the reported timings.
    """
    # Warm-up (not timed).
    for _, graph in graphs:
        symmetry_number(graph)

    rows = []
    for inchi, graph in graphs:
        best_ns = float("inf")
        result = None
        for _ in range(repeats):
            start = time.perf_counter_ns()
            result = symmetry_number(graph)
            elapsed = time.perf_counter_ns() - start
            best_ns = min(best_ns, elapsed)
        rows.append(
            {
                "InChI": inchi,
                "Symmetry Number": result,
                "Time (ms)": best_ns / 1e6,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Time symmetry_number() per molecule in kinbot.csv."
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed repetitions per molecule (best is kept). Default: 3",
    )
    args = parser.parse_args()

    print(f"Loading and building graphs from {CSV_PATH} ...")
    graphs = build_graphs()
    print(f"Built {len(graphs)} graphs (StereoMolGraph construction not timed).\n")

    print(f"Timing symmetry_number() per molecule (best of {args.repeats} runs) ...")
    df = time_symmetry_number(graphs, args.repeats)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print()

    times_ms = df["Time (ms)"]
    print("─" * 56)
    print(f"  Molecules timed      : {len(df)}")
    print(f"  Total time           : {times_ms.sum():.3f} ms")
    print(f"  Mean   per molecule  : {times_ms.mean():.4f} ms")
    print(f"  Median per molecule  : {times_ms.median():.4f} ms")
    print(f"  Min    per molecule  : {times_ms.min():.4f} ms")
    print(f"  Max    per molecule  : {times_ms.max():.4f} ms")
    print(f"  Std    per molecule  : {times_ms.std():.4f} ms")
    print("─" * 56)


if __name__ == "__main__":
    main()
