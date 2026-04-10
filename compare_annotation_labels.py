"""
Compare "Start page" labels across multiple annotation workbooks.

For the first K dossiers (order from the first workbook), reports (stem, page) keys where
the **binary** label (document start vs not) disagrees between files that annotate that row.
Empty/NaN cells are treated as not-start, same as training.

Fleiss' κ is computed on (stem, page) rows where **every** listed workbook has that row
(binary categories: not-start vs start). 

Run:
  python compare_annotation_labels.py
  python compare_annotation_labels.py --first-dossiers 5 --annotations "annotation 1.xlsx" ...
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa

from annotation_io import (
    binary_start_label,
    find_start_column,
    read_annotation_sheet,
    stem_order_first_appearance,
)

SCRIPT_DIR = Path(__file__).resolve().parent


def fleiss_binary_table(
    key_to_bins: dict[tuple[str, int], dict[str, int]],
    rater_names: list[str],
) -> np.ndarray | None:
    """Rows = subjects with a rating from every rater; columns = [count 0, count 1]."""
    n = len(rater_names)
    rows: list[list[int]] = []
    for bins in key_to_bins.values():
        if len(bins) != n or set(bins.keys()) != set(rater_names):
            continue
        labels = [bins[name] for name in rater_names]
        rows.append([labels.count(0), labels.count(1)])
    if not rows:
        return None
    return np.asarray(rows, dtype=float)


def main() -> None:
    p = argparse.ArgumentParser(description="Compare Start page labels across xlsx files.")
    p.add_argument("--first-dossiers", type=int, default=5, help="Number of dossiers to compare.")
    p.add_argument(
        "--annotations",
        type=Path,
        nargs="*",
        default=[
            SCRIPT_DIR / "annotation 1.xlsx",
            SCRIPT_DIR / "annotation 2.xlsx",
            SCRIPT_DIR / "annotation 3.xlsx",
            SCRIPT_DIR / "annotation 4.xlsx",
        ],
        help="Annotation workbooks (same order as training merge tie-break).",
    )
    args = p.parse_args()

    paths = [x.expanduser().resolve() for x in args.annotations]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    dfs = {path.name: read_annotation_sheet(path) for path in paths}
    base_name = paths[0].name
    first_stems = stem_order_first_appearance(dfs[base_name])[: args.first_dossiers]
    first_set = set(first_stems)

    # (stem, page) -> { file: binary }
    key_to_bins: dict[tuple[str, int], dict[str, int]] = defaultdict(dict)

    for path in paths:
        df = dfs[path.name]
        start_col = find_start_column(df)
        for _, row in df.iterrows():
            if pd.isna(row.get("image path")):
                continue
            stem = Path(str(row["image path"]).strip()).stem
            if stem not in first_set:
                continue
            page = int(row["page number"])
            key_to_bins[(stem, page)][path.name] = binary_start_label(row[start_col])

    order = {s: i for i, s in enumerate(first_stems)}
    conflicts = []
    for key in sorted(key_to_bins.keys(), key=lambda k: (order.get(k[0], 10**9), k[1])):
        bins = key_to_bins[key]
        if len(set(bins.values())) > 1:
            conflicts.append((key, bins))

    rater_names = [p.name for p in paths]
    table = fleiss_binary_table(key_to_bins, rater_names)
    print(f"Compared first {args.first_dossiers} dossiers (stem order from {base_name}):")
    print(first_stems)
    if table is None:
        print("\nFleiss' kappa: skipped (no (stem, page) rows present in all workbooks).")
    elif len(paths) < 2:
        print("\nFleiss' kappa: skipped (need at least two annotation files).")
    else:
        k = fleiss_kappa(table)
        print(
            f"\nFleiss' kappa (binary Start page, {int(table[0].sum())} raters, "
            f"{table.shape[0]} subjects with complete ratings): {k:.4f}"
        )
    print(f"\nBinary disagreements (yes vs no, across files that contain the row): {len(conflicts)}")
    for (stem, page), bins in conflicts[:80]:
        print(f"  {stem} page {page}: {bins}")
    if len(conflicts) > 80:
        print(f"  ... and {len(conflicts) - 80} more")


if __name__ == "__main__":
    main()
