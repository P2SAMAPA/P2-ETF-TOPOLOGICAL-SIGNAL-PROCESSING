import pandas as pd
from huggingface_hub import HfFileSystem
import config

_fs = HfFileSystem(token=config.HF_TOKEN)
_master_cache = None

def load_master_data():
    """Load master data parquet from HF repo once."""
    global _master_cache
    if _master_cache is not None:
        return _master_cache
    path = f"datasets/{config.DATA_REPO}/master_data.parquet"
    with _fs.open(path, "rb") as f:
        df = pd.read_parquet(f)
    df['date'] = pd.to_datetime(df['date'])
    _master_cache = df
    return df

def get_ticker_returns(tickers, start_date=None, end_date=None):
    """
    Return a DataFrame of daily log returns for given tickers.
    Columns = tickers, index = date.
    """
    df = load_master_data()
    mask = df['ticker'].isin(tickers)
    if start_date:
        mask &= df['date'] >= pd.to_datetime(start_date)
    if end_date:
        mask &= df['date'] <= pd.to_datetime(end_date)
    sub = df[mask].copy()
    # pivot to wide format
    pivot = sub.pivot(index='date', columns='ticker', values='close')
    # log returns
    rets = np.log(pivot / pivot.shift(1)).dropna()
    # align to common dates (all tickers present)
    rets = rets.dropna()
    return rets

def get_universe_returns(universe_name, start_date=None, end_date=None):
    """Convenience wrapper for a full universe."""
    tickers = config.UNIVERSES.get(universe_name, [])
    return get_ticker_returns(tickers, start_date, end_date)
