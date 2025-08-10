# src/pipeline.py (FULL with adaptive ML)
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from transformers import pipeline as hf_pipeline
import numpy as np

from src.config import RunConfig, ARTIFACT_DIR
from src.data_extraction import YahooExtractor
from src.features import compute_returns, engineer_features
from src.models import Backtester, MLResults
from src.optimizer import PortfolioOptimizer
from src.storage import save_df, save_json

plt.style.use("fivethirtyeight")

FINANCE_CONTEXT = """
A stock represents a share in the ownership of a company and constitutes a claim on part of the company’s assets and earnings.
Bonds are fixed income instruments that represent loans made by an investor to a borrower, typically corporations or governments.
Inflation is the rate at which the general level of prices for goods and services is rising and subsequently eroding purchasing power.
A mutual fund pools money from many investors to purchase a diversified portfolio of stocks, bonds, or other securities.
The stock market is a marketplace where investors can buy and sell shares of publicly traded companies.
"""

class FinancePipeline:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.extractor = YahooExtractor()
        self.optimizer = PortfolioOptimizer(risk_free_rate=cfg.risk_free_rate)
        self.qa = hf_pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

    def root(self) -> Path:
        p = ARTIFACT_DIR / self.cfg.cache_key
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ---------- helpers to save visuals ----------
    def _save_price_history_plot(self, close: pd.DataFrame) -> Path:
        fig = go.Figure()
        for t in close.columns:
            fig.add_trace(go.Scatter(x=close.index, y=close[t], mode="lines", name=t))
        fig.update_layout(title="Stock Price History", xaxis_title="Date", yaxis_title="Price")
        path = self.root() / "price_history.html"
        fig.write_html(str(path))
        return path

    def _save_return_hist(self, returns: pd.DataFrame) -> Path:
        path = self.root() / "return_distribution.png"
        returns.hist(bins=50, alpha=0.6)
        plt.title("Return Distribution")
        plt.xlabel("Daily Return"); plt.ylabel("Frequency")
        plt.tight_layout(); plt.savefig(path); plt.close()
        return path

    def _save_pred_vs_actual(self, mlres: MLResults) -> Path:
        path = self.root() / "pred_vs_actual.html"
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mlres.predictions.index, y=mlres.predictions["Return"], name="Actual"))
        fig.add_trace(go.Scatter(x=mlres.predictions.index, y=mlres.predictions["Predictions"], name="Predicted"))
        fig.update_layout(title="Predicted vs Actual Returns", xaxis_title="Date", yaxis_title="Return")
        fig.write_html(str(path))
        return path

    def _save_alloc_comparison(self, hist_weights: dict, tickers: list[str], mlres: MLResults) -> Path:
        # naive ML weights from average predicted return
        avg_pred = mlres.predictions["Predictions"].mean() if not mlres.predictions.empty else 0.0
        mu_ml = pd.Series([avg_pred] * len(tickers), index=tickers)
        if mu_ml.sum() != 0:
            ml_weights = (mu_ml / mu_ml.sum()).to_dict()
        else:
            ml_weights = {t: 0.0 for t in tickers}

        df = pd.DataFrame({
            "Historical": pd.Series(hist_weights),
            "ML-Predicted": pd.Series(ml_weights),
        }).fillna(0.0)

        path = self.root() / "allocation_comparison.png"
        ax = df.plot(kind="bar", title="Comparison: Historical vs ML-Predicted Allocation")
        ax.set_ylabel("Weight")
        plt.tight_layout(); plt.savefig(path); plt.close()
        return path
    # --------------------------------------------

    def run(self, user_question: str | None = None) -> dict:
        # ---- Extract ----
        data = self.extractor.download(self.cfg.tickers, self.cfg.start, self.cfg.end)
        close = data.close

        price_hist_html = self._save_price_history_plot(close)

        # ---- Transform ----
        rets = compute_returns(close)
        ret_hist_png = self._save_return_hist(rets)

        # ---- Optimize (historical) ----
        port = self.optimizer.optimize(close, self.cfg.investment_amount)

        # ---- ML: features + adaptive backtest (Patch 2) ----
        feats, predictors = engineer_features(self.cfg.tickers, close, data.volume)
        rows = len(feats)

        pred_vs_actual_html = None
        # default empty MLResults if we skip ML
        mlres = MLResults(predictions=pd.DataFrame(columns=["Return", "Predictions"]),
                          mae=float("nan"), rmse=float("nan"))

        if rows >= 300:
            # choose sizes that fit the data
            train_size = min(1000, max(200, int(rows * 0.6)))
            step = min(250, max(50, int(rows * 0.2)))

            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(
                n_estimators=150, min_samples_split=50, random_state=1, n_jobs=-1
            )
            bt = Backtester(model=rf, train_size=train_size, step=step)
            mlres = bt.run(feats, predictors)

            if not mlres.predictions.empty:
                pred_vs_actual_html = self._save_pred_vs_actual(mlres)

        # Always save comparison (uses naive ML weights if ML skipped)
        alloc_cmp_png = self._save_alloc_comparison(port.weights, self.cfg.tickers, mlres)

        # ---- Optional Q&A ----
        answer = None
        if user_question:
            res = self.qa(question=user_question, context=FINANCE_CONTEXT)
            answer = res.get("answer", "")

        # ---- Save artifacts & summary ----
        close_csv   = self.root() / "close.csv"
        returns_csv = self.root() / "returns.csv"
        ml_csv      = self.root() / "ml_predictions.csv"
        save_df(close, close_csv)
        save_df(rets, returns_csv)
        save_df(mlres.predictions, ml_csv)

        # guard NaNs for summary
        mae_val = 0.0 if (mlres.mae is None or (isinstance(mlres.mae, float) and np.isnan(mlres.mae))) else mlres.mae
        rmse_val = 0.0 if (mlres.rmse is None or (isinstance(mlres.rmse, float) and np.isnan(mlres.rmse))) else mlres.rmse

        summary_path = self.root() / "summary.json"
        save_json({
            "tickers": self.cfg.tickers,
            "investment": self.cfg.investment_amount,
            "risk_free_rate": self.cfg.risk_free_rate,
            "weights": port.weights,
            "exp_return": port.exp_return,
            "volatility": port.volatility,
            "sharpe": port.sharpe,
            "discrete_allocation": port.discrete_allocation,
            "cash_leftover": port.cash_leftover,
            "mae": mae_val,
            "rmse": rmse_val,
            "qa_answer": answer,
            "failed_tickers": getattr(data, "failed", []),
        }, summary_path)

        return {
            "close": str(close_csv),
            "returns": str(returns_csv),
            "ml_predictions": str(ml_csv),
            "summary": str(summary_path),
            "price_history_html": str(price_hist_html),
            "return_hist_png": str(ret_hist_png),
            "pred_vs_actual_html": str(pred_vs_actual_html) if pred_vs_actual_html else None,
            "allocation_compare_png": str(alloc_cmp_png),
        }
