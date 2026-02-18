import json
import ast
import re
import argparse
from pathlib import Path

import pandas as pd


def parse_counts_cell(x):
    if x is None:
        return {}
    s = str(x).strip()
    if s == "" or s.lower() in {"null", "none", "nan"}:
        return {}

    # Try strict JSON
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    # Quote unquoted keys (ClickHouse-style {a:1,b:2})
    s2 = re.sub(r'([{,]\s*)([A-Za-z0-9_\-\.]+)\s*:', r'\1"\2":', s)

    try:
        obj = json.loads(s2)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    # Fallback to Python literal
    try:
        obj = ast.literal_eval(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    try:
        obj = ast.literal_eval(s2)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser("Bulk Edit Users")
    parser.add_argument(
        "input_csv",
        type=str,
        help="Path to input CSV containing event_type_counts column"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output CSV path (default: <input>_unpacked.csv)"
    )

    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"File not found: {input_csv}")

    if args.output:
        output_csv = Path(args.output)
    else:
        output_csv = input_csv.with_name(input_csv.stem + "_unpacked.csv")

    df = pd.read_csv(input_csv)

    if "event_type_counts" not in df.columns:
        raise ValueError(
            f"Column 'event_type_counts' not found. Found columns: {list(df.columns)}"
        )

    parsed = df["event_type_counts"].apply(parse_counts_cell)
    expanded = pd.json_normalize(parsed)

    expanded = expanded.apply(pd.to_numeric, errors="coerce").fillna(0).astype("int64")

    out = pd.concat([df.drop(columns=["event_type_counts"]), expanded], axis=1)

    out.to_csv(output_csv, index=False)
    print(f"Wrote: {output_csv}")


if __name__ == "__main__":
    main()
