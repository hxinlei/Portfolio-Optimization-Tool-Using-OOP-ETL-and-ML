from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

def save_df(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)

def load_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)

def save_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
