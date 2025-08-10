from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pypfopt import expected_returns, risk_models, objective_functions
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

@dataclass
class PortfolioResult:
    weights: dict
    exp_return: float
    volatility: float
    sharpe: float
    discrete_allocation: dict
    cash_leftover: float

def _make_psd(S: pd.DataFrame, max_tries: int = 5) -> pd.DataFrame:
    """Symmetrize + add diagonal jitter until S is PSD."""
    S = (S + S.T) / 2
    jitter = 1e-8
    for _ in range(max_tries):
        eigs = np.linalg.eigvalsh(S.values)
        if np.all(eigs >= -1e-12):
            return S
        S = S + np.eye(S.shape[0]) * jitter
        jitter *= 10
    return S  # return best effort

class PortfolioOptimizer:
    def __init__(self, risk_free_rate: float = 0.0):
        self.risk_free_rate = risk_free_rate

    def _fit_ef(self, mu: pd.Series, S: pd.DataFrame, solver: str | None = None) -> EfficientFrontier:
        if solver:
            return EfficientFrontier(mu, S, solver=solver)
        return EfficientFrontier(mu, S)

    def optimize(self, prices: pd.DataFrame, investment_amount: float) -> PortfolioResult:
        # Expected returns & PSD covariance
        mu = expected_returns.mean_historical_return(prices, frequency=252)
        # Shrinkage tends to be PSD and numerically stable
        S = risk_models.CovarianceShrinkage(prices, frequency=252).ledoit_wolf()
        S = _make_psd(S)

        # Try sequence of increasingly conservative solves
        ef = None
        solved = False
        # 1) Default solver
        try:
            ef = self._fit_ef(mu, S)
            ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            solved = True
        except Exception:
            pass

        # 2) Fallback solver
        if not solved:
            try:
                ef = self._fit_ef(mu, S, solver="SCS")
                ef.max_sharpe(risk_free_rate=self.risk_free_rate)
                solved = True
            except Exception:
                pass

        # 3) Add mild L2 regularization (improves conditioning)
        if not solved:
            try:
                ef = self._fit_ef(mu, S, solver="SCS")
                ef.add_objective(objective_functions.L2_reg, gamma=1e-3)
                ef.max_sharpe(risk_free_rate=self.risk_free_rate)
                solved = True
            except Exception:
                pass

        # 4) Last resort: Max Quadratic Utility (more stable than max_sharpe)
        if not solved:
            ef = self._fit_ef(mu, S, solver="SCS")
            ef.add_objective(objective_functions.L2_reg, gamma=1e-3)
            ef.max_quadratic_utility(risk_aversion=1)
            solved = True

        weights = ef.clean_weights()
        exp_ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=self.risk_free_rate)

        latest = get_latest_prices(prices)
        # If LP allocation fails on edge cases, fallback to greedy
        try:
            da = DiscreteAllocation(weights, latest, total_portfolio_value=investment_amount)
            allocation, leftover = da.lp_portfolio()
        except Exception:
            da = DiscreteAllocation(weights, latest, total_portfolio_value=investment_amount)
            allocation, leftover = da.greedy_portfolio()  # simpler but robust

        return PortfolioResult(
            weights=weights,
            exp_return=float(exp_ret),
            volatility=float(vol),
            sharpe=float(sharpe),
            discrete_allocation=allocation,
            cash_leftover=float(leftover),
        )
