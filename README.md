# Portfolio Optimization Tool Using OOP, ETL, and ML

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
1. Python 3.10+ recommended.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Run the Streamlit App
```bash
streamlit run src/ui_streamlit.py
```
Enter tickers, date range, and investment amount in the sidebar, then run the pipeline.

### Generate a CLI Report
```bash
python -m src.report --key <cache_key>
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
