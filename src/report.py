from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from src.config import ARTIFACT_DIR
from src.storage import load_df, load_json

def explain(summary: dict):
    print("=== PROJECT REPORT ===")
    print("Tickers:", ", ".join(summary["tickers"]))
    print(f"Investment Amount: ${summary['investment']:,.2f}")
    print("\nPortfolio (Historical, Max Sharpe):")
    for k, v in summary["weights"].items():
        print(f"  {k}: {v:.2%}")
    print(f"Expected Return: {summary['exp_return']:.2%}")
    print(f"Volatility:      {summary['volatility']:.2%}")
    print(f"Sharpe:          {summary['sharpe']:.2f}")
    print(f"Discrete Allocation: {summary['discrete_allocation']}")
    print(f"Cash Leftover:   ${summary['cash_leftover']:.2f}")
    print("\nML Evaluation:")
    print(f"MAE:  {summary['mae']:.6f}")
    print(f"RMSE: {summary['rmse']:.6f}")
    if summary.get("qa_answer"):
        print("\nQ&A:", summary["qa_answer"])

def main(key: str):
    root = ARTIFACT_DIR / key
    summary = load_json(root / "summary.json")
    explain(summary)

    # Quick visual check
    close = load_df(root / "close.csv")
    close.plot(title="Price Trend", figsize=(10, 4))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--key", default="demo")
    args = p.parse_args()
    main(args.key)