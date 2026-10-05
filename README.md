# Portfolio Optimization Tool with Machine Learning

## Quick Start
Get the app running in three steps:

```bash
git clone https://github.com/hxinlei/Portfolio-Optimization-Tool-with-Machine-Learning-.git
cd Portfolio-Optimization-Tool-with-Machine-Learning-
pip install -r requirements.txt
streamlit run src/ui_streamlit.py

# Press Control + C in the terminal to stop the app
```

This project is an object-oriented pipeline for financial data analysis, portfolio optimization, and simple machine learning predictions, with a Streamlit-based interface and a built-in finance chatbot.

## Features
- **Data Extraction**: Downloads historical stock price data from Yahoo Finance with batching and retries.
- **Feature Engineering**: Calculates returns, moving average ratios, and trend-based features.
- **Portfolio Optimization**: Uses PyPortfolioOpt to compute a Max-Sharpe portfolio with fallback strategies and discrete allocation.
- **Machine Learning**: Trains a Random Forest model with walk-forward validation to predict short-term returns.
- **Visualization**: Interactive charts and allocation comparisons via Streamlit; artifacts saved for reproducibility.
- **Finance Chatbot**: Answers questions about finance terms and about the results of your own runs (for example, "What is my Sharpe ratio?"), with saved chat history and a response time shown under each answer.
- **Tests**: A 25-question regression suite checks the chatbot's accuracy, consistency and speed.

## Structure
```
src/
  config.py           # Settings and paths
  data_extraction.py  # Data download logic
  features.py         # Feature engineering
  models.py           # Backtesting and ML results
  optimizer.py        # Portfolio optimization
  pipeline.py         # Main pipeline orchestration
  chatbot.py          # Finance chatbot (retrieval + BERT question answering)
  report.py           # CLI reporting tool
  storage.py          # Save/load helpers
  ui_streamlit.py     # Streamlit dashboard
tests/
  test_chatbot_regression.py  # 25-question chatbot test suite
  baseline_answers.json       # Saved answers the tests compare against
artifacts/            # Output from runs
requirements.txt
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/hxinlei/Portfolio-Optimization-Tool-with-Machine-Learning-.git
   cd Portfolio-Optimization-Tool-with-Machine-Learning-
   ```
2. Python 3.10+ recommended.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Run the Streamlit App
```bash
streamlit run src/ui_streamlit.py

# Press Control + C in the terminal to stop the app
```
Notes:
- While Streamlit is running, that terminal is "busy" (you cannot type other commands there). Open a new terminal tab/window for other commands, or press Control + C to stop Streamlit and return to the prompt.
- On first run, Streamlit may prompt for an email and show telemetry info. You can simply press Enter to skip.

Enter tickers, date range, and investment amount in the sidebar, then run the pipeline.

### Ask the Finance Chatbot
Scroll to **Finance Chatbot** at the bottom of the app, choose the run to chat about, and type a question.

How it works:
1. **Portfolio facts**: The chatbot turns the run's saved results (expected return, volatility, Sharpe ratio, weights, share allocation, leftover cash) into short sentences.
2. **Retrieval**: For each question, it picks the 3 most relevant sentences from those facts and a 25-entry finance glossary, using keyword scoring that gives rare words more weight (the idea behind TF-IDF).
3. **Answering**: A BERT question-answering model reads only those 3 sentences and extracts the answer. The model loads once on the first question and is reused after that.
4. **Fallback**: Off-topic or low-confidence questions get a clear "I don't know" reply instead of a guess.

Questions that mention "my", "our" or "this" are answered from your portfolio first. Chat history is saved per run in `artifacts/<cache_key>/chat_history.json` and reloads when you reopen the app.

Example questions:
- What is my Sharpe ratio?
- What is my largest holding?
- How much cash is left over in my portfolio?
- What does the efficient frontier mean?

### Generate a CLI Report
```bash
python -m src.report --key <cache_key>
```

## Testing
Run the chatbot test suite from the repository root:
```bash
python -m pytest tests -v -s
```
The suite asks 25 fixed questions (20 finance terms and 5 portfolio questions) and checks that:
- the right fact is retrieved for every question,
- every answer matches the saved baseline in `tests/baseline_answers.json`,
- asking the same question twice gives the same answer,
- the median response time stays under 3 seconds (printed at the end).

If `tests/baseline_answers.json` does not exist yet, the first run creates it and skips the comparison. Review the answers in that file, commit it, and run the tests again.

## Cache Key
The cache key is a short name that identifies a run and its saved outputs.
- In the Streamlit sidebar, set the **Run name / Cache key** before running. If you leave it blank, a default (e.g., `demo`) may be used.
- Artifacts are stored under `artifacts/<cache_key>/` so you can reload or compare runs later.
- The CLI uses the same key via `--key`, for example:
  ```bash
  python -m src.report --key demo
  ```

## Artifacts
Saved under `artifacts/<cache_key>/`:
- CSVs for prices, returns, and predictions
- JSON summary of portfolio metrics (MAE and RMSE are `null` when there was not enough data to train the model)
- Plots (PNG/HTML) for prices, returns, predictions, and allocations
- Chat history for that run

## Limitations
- No allocation constraints beyond defaults.
- Machine learning allocation comparison is a simple baseline.
- Subject to Yahoo Finance rate limits.
- The chatbot extracts short answers from its facts rather than writing full explanations, so it can answer "what" questions (such as "What is my Sharpe ratio?") but not "why" questions.
- The BERT model is about 1.3 GB and downloads the first time the chatbot is used.
