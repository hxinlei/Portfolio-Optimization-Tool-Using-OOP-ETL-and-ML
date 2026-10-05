"""Regression suite for the finance chatbot: 25 fixed questions.

Run from the repository root:   python -m pytest tests -v

What it checks
  1. Retrieval: every question pulls the right fact as its top source (no model needed, runs in under a second).
  2. Consistency: every answer matches the saved baseline in tests/baseline_answers.json.
     The first run records the baseline; later runs fail if any answer has changed.
  3. Repeatability: asking the same question twice gives the same answer.
  4. Speed: the median response time after warm-up is printed and must stay under 3 seconds.
"""
import json
import statistics
from pathlib import Path

import pytest

from src import chatbot

BASELINE = Path(__file__).parent / "baseline_answers.json"

SAMPLE_RUN = {
    "exp_return": 0.241, "volatility": 0.213, "sharpe": 1.13, "investment": 15000.0,
    "weights": {"AAPL": 0.30, "MSFT": 0.52, "GOOGL": 0.18},
    "discrete_allocation": {"AAPL": 19, "MSFT": 18, "GOOGL": 15}, "cash_leftover": 43.2,
}

# (question, a word that must appear in the top fact retrieved for it)
CASES = [
    ("What is a stock?", "ownership"),
    ("What are bonds?", "fixed income"),
    ("What is inflation?", "prices"),
    ("What does a mutual fund do?", "pools"),
    ("What is the stock market?", "marketplace"),
    ("What is an ETF?", "exchange-traded"),
    ("What is a dividend?", "cash payment"),
    ("What does diversification mean?", "spreading"),
    ("What is volatility?", "standard deviation"),
    ("How is expected return estimated?", "historical"),
    ("What does the Sharpe ratio measure?", "risk-free rate"),
    ("What is the risk-free rate?", "Treasury"),
    ("What is the efficient frontier?", "highest expected return"),
    ("Who developed Modern Portfolio Theory?", "Markowitz"),
    ("What does covariance measure?", "move together"),
    ("What is correlation?", "minus one"),
    ("What is rebalancing?", "target weights"),
    ("How is market capitalization calculated?", "share price multiplied"),
    ("What is a bear market?", "falling"),
    ("What does beta measure?", "one percent"),
    ("What is the Sharpe ratio of my portfolio?", "1.13"),
    ("What is my expected return?", "24.1%"),
    ("How risky is my portfolio?", "21.3%"),
    ("What is my largest holding?", "MSFT"),
    ("How much cash is left over in my portfolio?", "$43.20"),
]
QUESTIONS = [q for q, _ in CASES]


def test_suite_has_25_questions():
    assert len(CASES) == 25 and len(set(QUESTIONS)) == 25


@pytest.mark.parametrize("question,expected", CASES)
def test_retrieval_finds_the_right_fact(question, expected):
    personal = bool(chatbot._PERSONAL & set(chatbot._tokens(question)))
    facts = chatbot.portfolio_facts(SAMPLE_RUN) if personal else chatbot.KNOWLEDGE
    top = chatbot.retrieve(question, facts)
    assert top, f"nothing retrieved for: {question}"
    assert expected in top[0], f"top fact for '{question}' was: {top[0]}"


def test_off_topic_question_gets_fallback_without_calling_the_model():
    def never_called(**_):
        raise AssertionError("model should not run for an off-topic question")
    out = chatbot.answer("Who won the football game?", qa=never_called)
    assert out["answer"] == chatbot.FALLBACK


def test_history_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(chatbot, "ARTIFACT_DIR", tmp_path)
    msgs = [{"role": "user", "content": "What is a stock?"}, {"role": "assistant", "content": "a share", "latency_ms": 420}]
    chatbot.save_history("demo", msgs)
    assert chatbot.load_history("demo") == msgs
    assert chatbot.load_history("never-saved") == []


@pytest.fixture(scope="module")
def qa():
    pytest.importorskip("transformers")
    return chatbot.get_qa_model()


@pytest.fixture(scope="module")
def answers(qa):
    return {q: chatbot.answer(q, summary=SAMPLE_RUN, qa=qa) for q in QUESTIONS}


def test_answers_match_baseline(answers):
    current = {q: a["answer"] for q, a in answers.items()}
    if not BASELINE.exists():
        BASELINE.write_text(json.dumps(current, indent=2))
        pytest.skip("Baseline recorded. Review tests/baseline_answers.json, commit it, and run again.")
    baseline = json.loads(BASELINE.read_text())
    changed = {q: (baseline.get(q), a) for q, a in current.items() if baseline.get(q) != a}
    assert not changed, f"{len(changed)} of {len(current)} answers changed: {changed}"


def test_same_question_twice_gives_same_answer(qa, answers):
    for q in QUESTIONS:
        assert chatbot.answer(q, summary=SAMPLE_RUN, qa=qa)["answer"] == answers[q]["answer"]


def test_median_response_time(qa, answers):
    times = [chatbot.answer(q, summary=SAMPLE_RUN, qa=qa)["latency_ms"] for q in QUESTIONS]
    median = statistics.median(times)
    print(f"\nMedian response time: {median / 1000:.2f} s (slowest {max(times) / 1000:.2f} s)")
    assert median < 3000
