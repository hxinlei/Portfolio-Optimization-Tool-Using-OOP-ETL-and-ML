# Portfolio Optimization Tool (OOP + ETL + ML)

## Quick Start
Get the app running in three steps:

```bash
git clone https://github.com/hxinlei/Portfolio-Optimization-Tool-Using-OOP-ETL-and-ML.git
cd Portfolio-Optimization-Tool-Using-OOP-ETL-and-ML
pip install -r requirements.txt
streamlit run src/ui_streamlit.py

# Press Control + C in the terminal to stop the app
```

This project is an object-oriented pipeline for financial data analysis, portfolio optimization, and simple machine learning predictions, with a Streamlit-based interface.

## Features
- **Data Extraction**: Downloads historical stock price data from Yahoo Finance with batching and retries.
- **Feature Engineering**: Calculates returns, moving average ratios, and trend-based features.
- **Portfolio Optimization**: Uses PyPortfolioOpt to compute a Max-Sharpe portfolio with fallback strategies and discrete allocation.
- **Machine Learning**: Trains a Random Forest model with walk-forward validation to predict short-term returns.
- **Visualization**: Interactive charts and allocation comparisons via Streamlit; artifacts saved for reproducibility.
- **Q&A**: Optional finance question answering using a pretrained BERT model.

## Structure
```
src/
  config.py           # Settings and paths
  data_extraction.py  # Data download logic
  features.py         # Feature engineering
  models.py           # Backtesting and ML results
  optimizer.py        # Portfolio optimization
  pipeline.py         # Main pipeline orchestration
  report.py           # CLI reporting tool
  storage.py          # Save/load helpers
  ui_streamlit.py     # Streamlit dashboard
artifacts/            # Output from runs
requirements.txt
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/hxinlei/Portfolio-Optimization-Tool-Using-OOP-ETL-and-ML.git
   cd Portfolio-Optimization-Tool-Using-OOP-ETL-and-ML
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

### Generate a CLI Report
```bash
python -m src.report --key <cache_key>
```

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
- JSON summary of portfolio metrics
- Plots (PNG/HTML) for prices, returns, predictions, and allocations

## Limitations
- No allocation constraints beyond defaults.
- Machine learning allocation comparison is a simple baseline.
- Subject to Yahoo Finance rate limits.
