from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DATA_PATH = Path("data/processed/cleaned_data.csv")
OUTPUT_DIR = Path("outputs/metrics")
OUTPUT_CSV = OUTPUT_DIR / "categorical_value_audit.csv"
OUTPUT_TXT = OUTPUT_DIR / "categorical_audit_summary.txt"
LOW_CARDINALITY_THRESHOLD = 20


def detect_categorical_columns(df: pd.DataFrame) -> list[str]:
    categorical = set(
        df.select_dtypes(include=["object", "string", "category"]).columns
    )

    numeric_like = df.select_dtypes(include=["number", "bool"]).columns
    low_card_numeric = [
        col
        for col in numeric_like
        if df[col].nunique(dropna=True) <= LOW_CARDINALITY_THRESHOLD
    ]

    categorical.update(low_card_numeric)
    return sorted(categorical)


def summarize_column(series: pd.Series) -> dict[str, object]:
    non_null = series.dropna()
    type_set = {type(value) for value in non_null}
    mixed_type = len(type_set) > 1

    notes: list[str] = []

    if mixed_type:
        example_types = ", ".join(sorted(t.__name__ for t in type_set))
        notes.append(f"Mixed python types ({example_types})")

    string_values = [value for value in non_null if isinstance(value, str)]
    case_inconsistent = False
    if string_values:
        unique_strings = set(string_values)
        casefolded = Counter(value.casefold() for value in unique_strings)
        case_inconsistent = any(count > 1 for count in casefolded.values())
        if case_inconsistent:
            notes.append("Case variants detected (e.g., Male vs male)")

    numeric_encoding = series.dtype.kind in "ifb" and isinstance(series.dtype, np.dtype)
    if numeric_encoding and str(series.dtype) != "bool":
        notes.append("Numeric dtype flagged as categorical (check encoding)")

    string_numeric = False
    if series.dtype == object and string_values:
        string_numeric = all(value.replace(".", "", 1).isdigit() for value in string_values)
        if string_numeric:
            notes.append("String values look numeric (possible code list)")

    return {
        "column": series.name,
        "dtype": str(series.dtype),
        "cardinality": series.nunique(dropna=True),
        "missing_pct": round(series.isna().mean() * 100, 2),
        "notes": "; ".join(notes) if notes else "",
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH.resolve()}")

    df = pd.read_csv(DATA_PATH)
    categorical_columns = detect_categorical_columns(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audit_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for column in categorical_columns:
        series = df[column]
        dtype = str(series.dtype)

        counts = series.value_counts(dropna=False)
        values = ["<NA>" if pd.isna(value) else value for value in counts.index]

        audit_frames.append(
            pd.DataFrame(
                {
                    "column_name": column,
                    "raw_value": values,
                    "count": counts.values,
                    "dtype": dtype,
                }
            )
        )

        summary_rows.append(summarize_column(series))

    audit_df = pd.concat(audit_frames, ignore_index=True)
    audit_df.to_csv(OUTPUT_CSV, index=False)

    summary_df = pd.DataFrame(summary_rows)

    flagged = summary_df[summary_df["notes"] != ""]
    top_flagged = flagged.sort_values("cardinality", ascending=False)

    lines: list[str] = []
    lines.append("Categorical Value Audit Summary")
    lines.append("=" * 40)
    lines.append(f"Total rows analyzed: {len(df):,}")
    lines.append(f"Categorical columns inspected: {len(categorical_columns)}")
    lines.append("")

    if flagged.empty:
        lines.append("No immediate inconsistencies detected based on heuristics.")
    else:
        lines.append("Columns requiring attention:")
        for _, row in top_flagged.iterrows():
            lines.append(
                f"- {row['column']} (dtype={row['dtype']}, cardinality={row['cardinality']}): {row['notes']}"
            )
    lines.append("")
    lines.append("Full per-value details stored in categorical_value_audit.csv.")

    OUTPUT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote categorical audit CSV to {OUTPUT_CSV}")
    print(f"Wrote summary text to {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
