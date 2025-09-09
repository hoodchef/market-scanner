from typing import Iterable, Dict, Any, Optional

class DataProvider:
    """Pluggable market-data interface."""

    def get_quotes(self, tickers: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        """
        Return per-ticker dicts with at least:
          - price (float)
          - prev_close (float | None)
          - volume (int | None)
        """
        raise NotImplementedError

    def get_history(
        self, ticker: str, period: str = "1mo", interval: str = "1d"
    ):
        """Historical OHLCV for one symbol (returns a pandas.DataFrame)."""
        raise NotImplementedError

    def get_name(self, ticker: str) -> Optional[str]:
        """Best-effort human name for the symbol."""
        return None
