from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Tuple, List, Optional

import pandas as pd
import yfinance as yf

@dataclass
class MarketData:
    close: pd.DataFrame
    volume: pd.DataFrame
    failed: List[str]  # tickers that failed completely

class YahooExtractor:
    def __init__(self, max_retries: int = 3, pause_sec: float = 2.0, batch_size: int = 6):
        self.max_retries = max_retries
        self.pause_sec = pause_sec
        self.batch_size = batch_size

    def _download_batch(self, batch: List[str], start: str, end: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = yf.download(
            tickers=batch,
            start=start,
            end=end,
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=False,  # gentler for rate limits
            interval="1d",
        )

        if df.empty:
            # Return empty frames with the right columns so callers can handle
            return pd.DataFrame(), pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            avail = set(df.columns.get_level_values(0))
            close = pd.DataFrame({t: df[t]["Close"] for t in batch if t in avail and "Close" in df[t].columns})
            vol   = pd.DataFrame({t: df[t]["Volume"] for t in batch if t in avail and "Volume" in df[t].columns})
        else:
            # Single-ticker case sometimes returns a flat frame
            t = batch[0]
            if "Close" not in df or "Volume" not in df:
                return pd.DataFrame(), pd.DataFrame()
            close = df["Close"].to_frame(name=t)
            vol   = df["Volume"].to_frame(name=t)

        return close, vol

    def _download_single(self, ticker: str, start: str, end: Optional[str]) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        attempt = 0
        last_err = None
        while attempt < self.max_retries:
            try:
                df = yf.download(
                    tickers=ticker,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                    interval="1d",
                )
                if df.empty or "Close" not in df or "Volume" not in df:
                    raise RuntimeError("empty frame")
                return df["Close"].to_frame(name=ticker), df["Volume"].to_frame(name=ticker)
            except Exception as e:
                last_err = e
                attempt += 1
                time.sleep(self.pause_sec * attempt)  # backoff
        return None

    def download(self, tickers: List[str], start: str, end: Optional[str] = None) -> MarketData:
        tickers = [t.strip().upper() for t in tickers if t.strip()]
        if not tickers:
            raise ValueError("No valid tickers provided.")

        # Split the annotation across variables (fixes your SyntaxError)
        all_close: List[pd.DataFrame] = []
        all_vol: List[pd.DataFrame] = []
        failed: List[str] = []

        # 1) Try in batches
        for i in range(0, len(tickers), self.batch_size):
            batch = tickers[i:i + self.batch_size]
            attempt, last_err = 0, None
            while attempt < self.max_retries:
                try:
                    c, v = self._download_batch(batch, start, end)
                    if not c.empty:
                        all_close.append(c)
                        all_vol.append(v)
                        break
                    else:
                        raise RuntimeError("batch empty")
                except Exception as e:
                    last_err = e
                    attempt += 1
                    time.sleep(self.pause_sec * attempt)
            if attempt == self.max_retries:
                # 2) Fall back per-ticker for this batch
                for t in batch:
                    res = self._download_single(t, start, end)
                    if res is None:
                        failed.append(t)
                    else:
                        c, v = res
                        all_close.append(c)
                        all_vol.append(v)

        if not all_close:
            # Nothing worked at all
            raise RuntimeError("No data downloaded (likely rate limit or invalid tickers).")

        close = pd.concat(all_close, axis=1).sort_index().dropna(how="all")
        vol   = pd.concat(all_vol,   axis=1).sort_index().dropna(how="all")

        # Final clean: drop columns entirely empty
        close = close.dropna(axis=1, how="all")
        vol   = vol.dropna(axis=1,   how="all")

        return MarketData(close=close, volume=vol, failed=failed)
