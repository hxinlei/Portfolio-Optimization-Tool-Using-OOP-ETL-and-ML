from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

@dataclass(frozen=True)
class RunConfig:
    tickers: list[str]
    start: str = "2013-01-01"
    end: str | None = None
    investment_amount: float = 15000.0
    risk_free_rate: float = 0.0          # you used 0.0 in your script
    horizon_days: int = 5
    cache_key: str = "demo"