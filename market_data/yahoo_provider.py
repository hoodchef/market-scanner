from typing import Iterable, Dict, Any, Optional, List
import concurrent.futures as cf
import pandas as pd
import yfinance as yf

from .base import DataProvider

class YahooProvider(DataProvider):
    """
    Fast path: batch quotes via yf.download for price/volume/prev_close.
    Fallback: per-ticker history(2d,1d) in a thread pool.
    """

    def __init__(self, max_workers: int = 16, batch_size: int = 200):
        self.max_workers = max_workers
        self.batch_size = batch_size

    def get_quotes(self, tickers: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        tickers = [t.strip().upper() for t in tickers if t and t.strip()]
        if not tickers:
            return {}

        results: Dict[str, Dict[str, Any]] = {}

        # Batch fetch with yf.download
        for chunk in _chunks(tickers, self.batch_size):
            try:
                df = yf.download(
                    chunk, period="2d", interval="1d", auto_adjust=False,
                    progress=False, group_by="ticker", threads=True
                )
                if isinstance(df.columns, pd.MultiIndex):
                    # multi-ticker frame
                    have = set(df.columns.get_level_values(0))
                    for t in chunk:
                        if t in have:
                            sub = df[t].copy()
                            price, prev_close, vol = _extract_from_df(sub)
                            if price is not None:
                                results[t] = {
                                    "price": price,
                                    "prev_close": prev_close,
                                    "volume": vol,
                                }
                else:
                    # single ticker or flattened frame
                    price, prev_close, vol = _extract_from_df(df)
                    if price is not None and len(chunk) == 1:
                        results[chunk[0]] = {
                            "price": price, "prev_close": prev_close, "volume": vol
                        }
            except Exception:
                # If batch fails, we'll fill via fallback
                pass

        # Fallback for missing tickers
        remaining = [t for t in tickers if t not in results]
        if remaining:
            with cf.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                futs = {ex.submit(self._quote_single, t): t for t in remaining}
                for fut in cf.as_completed(futs):
                    t = futs[fut]
                    try:
                        q = fut.result()
                        if q:
                            results[t] = q
                    except Exception:
                        pass

        return results

    def get_history(self, ticker: str, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
        return yf.Ticker(ticker).history(period=period, interval=interval)

    def get_name(self, ticker: str) -> Optional[str]:
        # Keep lightweight for now; names can be added later if needed
        return None

    def _quote_single(self, ticker: str) -> Optional[Dict[str, Any]]:
        try:
            hist = yf.Ticker(ticker).history(period="2d", interval="1d")
            price, prev_close, vol = _extract_from_df(hist)
            if price is None:
                return None
            return {"price": price, "prev_close": prev_close, "volume": vol}
        except Exception:
            return None

def _chunks(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def _extract_from_df(df: pd.DataFrame):
    if df is None or df.empty:
        return None, None, None
    df = df.dropna()
    if df.empty:
        return None, None, None
    last = df.iloc[-1]
    price = float(last.get("Close")) if "Close" in last else None
    vol = int(last.get("Volume")) if "Volume" in last and pd.notna(last.get("Volume")) else None

    prev_close = None
    if len(df) >= 2:
        prev = df.iloc[-2]
        prev_close = float(prev.get("Close")) if "Close" in prev else None
    return price, prev_close, vol
