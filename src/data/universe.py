"""Ticker universes: which stocks/ETFs the agent is allowed to trade.

The "default" universe is a curated list of ~250 liquid, well-known US large
caps plus major ETFs. Names that fail to download (delisted, renamed) are
simply dropped by the downloader, so the list does not need to be perfect.
"""

from __future__ import annotations

# Broad, liquid ETFs (market, sectors, bonds, gold) — cheap diversification.
ETFS = [
    "SPY", "QQQ", "IWM", "DIA", "VTI",
    "XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE",
    "GLD", "SLV", "USO", "TLT", "IEF", "LQD", "HYG", "EEM", "EFA", "VNQ",
]

# Liquid US large caps (roughly S&P 100 plus other heavily traded names).
LARGE_CAPS = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "UNH", "JNJ", "XOM", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV",
    "LLY", "PEP", "KO", "AVGO", "COST", "WMT", "TMO", "MCD", "CSCO", "ACN",
    "ABT", "ADBE", "CRM", "DHR", "LIN", "NKE", "TXN", "NEE", "PM", "ORCL",
    "AMD", "WFC", "DIS", "UPS", "MS", "RTX", "INTC", "VZ", "QCOM", "HON",
    "COP", "T", "LOW", "IBM", "CAT", "AMGN", "SPGI", "GS", "INTU", "BA",
    "UNP", "PLD", "ELV", "SBUX", "DE", "GE", "BKNG", "MDT", "ADI", "LMT",
    "BLK", "AXP", "GILD", "TJX", "MMC", "SYK", "AMT", "CVS", "MDLZ", "ADP",
    "C", "VRTX", "CI", "TMUS", "SCHW", "MO", "ZTS", "CB", "SO", "PGR",
    "FI", "DUK", "BMY", "EOG", "BSX", "ITW", "REGN", "NOC", "SLB", "CSX",
    "MU", "AON", "APD", "CL", "ETN", "WM", "HUM", "ICE", "FDX", "PNC",
    "EMR", "MCK", "ORLY", "SHW", "MMM", "GD", "TGT", "MAR", "ROP", "PSX",
    "MPC", "USB", "AJG", "APH", "NSC", "TDG", "PXD", "F", "GM", "AZO",
    "AFL", "SRE", "AEP", "TRV", "ADSK", "CCI", "HLT", "KMB", "MSI", "TT",
    "PCAR", "CARR", "PAYX", "NXPI", "TEL", "ALL", "OXY", "AIG", "STZ",
    "MET", "D", "EXC", "CTAS", "DOW", "JCI", "WMB", "SPG", "BK", "ROST",
    "AMP", "KMI", "PRU", "MCHP", "FIS", "CME", "HES", "IQV", "A", "YUM",
    "OTIS", "SYY", "CMI", "GIS", "VLO", "LRCX", "KLAC", "SNPS", "CDNS",
    "MNST", "ODFL", "CTSH", "EW", "DD", "BIIB", "HAL", "DVN", "XEL", "ED",
    "PPG", "RSG", "VICI", "ON", "FTNT", "KDP", "DLR", "ANET", "EA", "FAST",
    "KR", "WEC", "GEHC", "EL", "CSGP", "OKE", "KHC", "IDXX", "DXCM", "ES",
    "HSY", "GLW", "IT", "AWK", "WBD", "EIX", "CBRE", "ZBH", "TSCO", "EFX",
    "AVB", "FANG", "WTW", "GPN", "DAL", "ULTA", "EBAY", "PYPL", "SQ",
    "UBER", "ABNB", "SNOW", "PLTR", "COIN", "SHOP", "NET", "DDOG", "CRWD",
    "ZS", "PANW", "NOW", "TEAM", "WDAY", "VEEV", "OKTA", "TWLO", "DOCU",
    "ZM", "ROKU", "PINS", "SNAP", "LYFT", "DASH", "RIVN", "LCID", "NIO",
]

SMOKE = [
    "SPY", "QQQ", "IWM", "GLD", "TLT",
    "AAPL", "MSFT", "AMZN", "JPM", "XOM", "JNJ", "WMT",
]


def get_universe(name: str) -> list[str]:
    """Return the raw candidate ticker list for a named universe."""
    if name == "smoke":
        return list(dict.fromkeys(SMOKE))
    if name == "default":
        return list(dict.fromkeys(ETFS + LARGE_CAPS))
    raise ValueError(f"Unknown universe '{name}' (expected 'default' or 'smoke')")
