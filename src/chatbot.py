"""Finance chatbot for the portfolio tool.

What this module adds on top of the original one-shot BERT Q&A:
  1. The model is loaded once and reused (get_qa_model), instead of on every pipeline run.
  2. Each question is answered from the few most relevant facts (retrieve), which keeps
     the text BERT has to read short and the response fast.
  3. Questions about "my portfolio" are answered from the saved results of a run
     (portfolio_facts), not only from general finance definitions.
  4. Low-confidence or off-topic questions get a clear "I don't know" reply.
  5. Chat history is saved per run and reloaded next session (load_history / save_history).
  6. Every answer reports its response time in milliseconds.
"""
from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from pathlib import Path

from src.config import ARTIFACT_DIR

MODEL_NAME = "bert-large-uncased-whole-word-masking-finetuned-squad"
MIN_SCORE = 0.10          # below this confidence the bot says it does not know
TOP_K = 3                 # number of facts passed to BERT per question
MAX_HISTORY = 200         # messages kept on disk per run
FALLBACK = "I don't have that in my notes yet. Try asking about a finance term or about this portfolio's results."

KNOWLEDGE = [
    "A stock represents a share in the ownership of a company and constitutes a claim on part of the company's assets and earnings.",
    "Bonds are fixed income instruments that represent loans made by an investor to a borrower, typically corporations or governments.",
    "Inflation is the rate at which the general level of prices for goods and services is rising and subsequently eroding purchasing power.",
    "A mutual fund pools money from many investors to purchase a diversified portfolio of stocks, bonds, or other securities.",
    "The stock market is a marketplace where investors can buy and sell shares of publicly traded companies.",
    "An ETF, or exchange-traded fund, is a basket of securities that trades on an exchange like a single stock.",
    "A dividend is a cash payment a company makes to its shareholders out of its profits.",
    "Diversification means spreading money across many assets so that a loss in one has a smaller effect on the whole portfolio.",
    "Volatility is the standard deviation of returns and is the most common measure of investment risk.",
    "Expected return is the average return an investor anticipates from an asset, usually estimated from its historical returns.",
    "The Sharpe ratio measures the return earned above the risk-free rate for each unit of volatility.",
    "The risk-free rate is the return on an investment with no default risk, usually a short-term government Treasury bill.",
    "The efficient frontier is the set of portfolios that offer the highest expected return for each level of risk.",
    "Modern Portfolio Theory is a framework by Harry Markowitz for choosing asset weights that balance expected return against risk.",
    "Covariance measures how the returns of two assets move together.",
    "Correlation is covariance scaled to a range between minus one and plus one.",
    "Rebalancing is the process of buying and selling assets to bring a portfolio back to its target weights.",
    "Market capitalization is the total value of a company's shares, equal to share price multiplied by shares outstanding.",
    "An index is a group of securities, such as the S&P 500, used to track the performance of a market.",
    "A bull market is a period of rising prices, usually defined as a gain of 20 percent or more from a recent low.",
    "A bear market is a period of falling prices, usually defined as a decline of 20 percent or more from a recent high.",
    "Liquidity is how quickly an asset can be sold for cash without moving its price.",
    "Compound interest is interest earned on both the original principal and the interest already added to it.",
    "The price-to-earnings ratio is a company's share price divided by its earnings per share.",
    "Beta measures how much a stock tends to move when the overall market moves by one percent.",
]

_STOP = set("a an and are as at be by can do does for from how i in is it its me of on or that the to was what when "
            "where which who why with you your about tell explain define mean means".split())
_PERSONAL = {"my", "mine", "our", "this"}
_ALIASES = {"risky": "risk", "riskiness": "risk", "volatile": "volatility", "biggest": "largest", "top": "largest"}


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9&]+", text.lower().replace("-", " "))
    out = []
    for w in words:
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]                       # crude plural stripping: bonds -> bond
        out.append(_ALIASES.get(w, w))
    return out


def portfolio_facts(summary: dict | None) -> list[str]:
    """Turn a run's summary.json into plain sentences the model can quote."""
    if not summary:
        return []
    facts = []
    if summary.get("exp_return") is not None:
        facts.append(f"My portfolio has an expected annual return of {summary['exp_return']:.1%}.")
    if summary.get("volatility") is not None:
        facts.append(f"My portfolio has an annual volatility, or risk, of {summary['volatility']:.1%}.")
    if summary.get("sharpe") is not None:
        facts.append(f"My portfolio has a Sharpe ratio of {summary['sharpe']:.2f}.")
    weights = {t: w for t, w in (summary.get("weights") or {}).items() if w and w > 0}
    if weights:
        top = max(weights, key=weights.get)
        facts.append(f"My largest holding, the biggest position, is {top} with a weight of {weights[top]:.1%}.")
        facts.append("My portfolio holds these stocks: " + ", ".join(sorted(weights)) + ".")
    alloc = summary.get("discrete_allocation") or {}
    if alloc:
        buys = ", ".join(f"{n} shares of {t}" for t, n in alloc.items())
        facts.append(f"My whole-share allocation is to buy {buys}.")
    if summary.get("cash_leftover") is not None:
        facts.append(f"My cash left over after buying whole shares is ${summary['cash_leftover']:,.2f}.")
    if summary.get("investment") is not None:
        facts.append(f"My investment amount is ${summary['investment']:,.0f}.")
    return facts


def retrieve(question: str, facts: list[str], top_k: int = TOP_K) -> list[str]:
    """Rank facts by weighted keyword overlap with the question and return the best few.

    Rare words count for more than common ones (a word found in 1 fact outweighs a word
    found in 10), which is the idea behind TF-IDF without needing another library.
    A match in the first few words of a fact counts 1.5x, because each fact opens with
    the term it defines ("The risk-free rate is ...").
    """
    q = [t for t in _tokens(question) if t not in _STOP and t not in _PERSONAL]
    if not q:
        return []
    ordered = [_tokens(f) for f in facts]
    fact_tokens = [set(toks) for toks in ordered]
    heads = [set(toks[:6]) for toks in ordered]
    df = {t: sum(t in ft for ft in fact_tokens) for t in set(q)}
    scored = []
    for i, ft in enumerate(fact_tokens):
        score = sum((1.5 if t in heads[i] else 1.0) / df[t] for t in set(q) if t in ft)
        if score > 0:
            scored.append((-score, i))
    return [facts[i] for _, i in sorted(scored)[:top_k]]


@lru_cache(maxsize=1)
def get_qa_model():
    """Load BERT once per process. The original code reloaded it on every pipeline run."""
    from transformers import pipeline as hf_pipeline
    return hf_pipeline("question-answering", model=MODEL_NAME)


def answer(question: str, summary: dict | None = None, qa=None, previous_question: str | None = None) -> dict:
    """Answer one question. Returns {"answer", "score", "latency_ms", "sources"}."""
    start = time.perf_counter()
    personal = bool(_PERSONAL & set(_tokens(question)))
    own = portfolio_facts(summary)

    sources = retrieve(question, own) if (personal and own) else []
    if not sources:
        sources = retrieve(question, KNOWLEDGE + own)
    if not sources and previous_question:      # follow-up such as "and why does it matter?"
        sources = retrieve(previous_question + " " + question, KNOWLEDGE + own)

    text, score = FALLBACK, 0.0
    if sources:
        qa = qa or get_qa_model()
        res = qa(question=question, context=" ".join(sources))
        score = float(res.get("score", 0.0))
        if res.get("answer", "").strip() and score >= MIN_SCORE:
            text = res["answer"].strip()
    return {
        "answer": text,
        "score": round(score, 4),
        "latency_ms": round((time.perf_counter() - start) * 1000),
        "sources": sources,
    }


def _history_path(cache_key: str) -> Path:
    return ARTIFACT_DIR / cache_key / "chat_history.json"


def load_history(cache_key: str) -> list[dict]:
    path = _history_path(cache_key)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def save_history(cache_key: str, history: list[dict]) -> None:
    path = _history_path(cache_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history[-MAX_HISTORY:], indent=2))
