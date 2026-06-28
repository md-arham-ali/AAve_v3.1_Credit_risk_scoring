"""Advanced univariate statistics for volatile / heavy-tailed financial data.
These are EDA metrics (tail risk, drawdown, concentration, robust distribution
shape, risk-adjusted return, dependence) that go beyond the validation profile in
``adv_validation.statistical_validation``. Each is a small generic function that
accepts EITHER a 1-D series-like (list / np.ndarray / pd.Series) OR ``(df, column)``,
and returns a float (np.nan when there isn't enough data or a denominator is 0).

Decimals-safety: parsing is delegated to ``adv_validation`` — ``_to_number`` /
``_numeric_pairs`` read uint256 / RAY / WAD integer STRINGS exactly and the
2**256-1 no-debt health-factor sentinel is dropped — so these run unchanged on the
raw query tables and on the scaled transformed frames. The genuinely complex pieces
(robust moments: skewness / excess kurtosis) reuse ``adv._series_stats`` rather than
being re-derived here.

Conventions
-----------
* "returns / P&L" metrics (VaR, CVaR, Sharpe, Sortino, vol, drawdown, ...) treat the
  input as a return / P&L series. Pass ``as_returns=False`` on the level-based ones to
  build a wealth index from a level/price series first.
* annualisation is opt-in: pass ``periods_per_year`` (e.g. ``PERIODS_PER_YEAR_2H``);
  when omitted no scaling is applied.

Libraries imported: numpy, pandas, math, adv_validation (decimals-safe parsing +
robust moments).
"""

import math

import numpy as np
import pandas as pd

import adv_validation as adv

# Period counts for annualisation on this project's grids (365-day year).
PERIODS_PER_YEAR_2H = 365 * 12     # 2-hour buckets
PERIODS_PER_YEAR_6H = 365 * 4      # 6-hour buckets
PERIODS_PER_YEAR_DAILY = 365

__all__ = [
    # tail risk / downside
    "value_at_risk", "conditional_var", "tail_ratio", "hill_tail_index",
    "downside_deviation", "upside_deviation", "semivariance",
    # drawdown
    "max_drawdown", "average_drawdown", "ulcer_index",
    # volatility / dispersion
    "realized_volatility", "robust_cv", "quartile_coeff_dispersion",
    "mean_abs_deviation", "coefficient_of_variation",
    # distribution shape
    "moment_skewness", "excess_kurtosis", "bowley_skewness", "kelly_skewness",
    "nonparametric_skew", "bimodality_coefficient", "jarque_bera",
    # concentration / inequality
    "gini_coefficient", "herfindahl_index", "shannon_entropy", "theil_index",
    "top_k_concentration",
    # risk-adjusted return
    "sharpe_ratio", "sortino_ratio", "calmar_ratio", "omega_ratio",
    # dependence / memory
    "autocorrelation", "hurst_exponent",
    # convenience
    "financial_metrics",
]


# --------------------------------------------------------------------------- #
# Shared primitives (parsing delegated to adv_validation — decimals-safe)
# --------------------------------------------------------------------------- #
def _clean(data, column=None):
    """Return a finite float64 array from a series-like or (df, column).

    Uses ``adv._to_number`` / ``adv._numeric_pairs`` so big-int strings parse exactly;
    None / non-numeric / the 2**256-1 sentinel and non-finite values are dropped.
    """
    if column is not None:
        _, vals = adv._numeric_pairs(data, column)        # (idx, parsed numbers)
    else:
        vals = [adv._to_number(v) for v in data]
        vals = [v for v in vals if v is not None]
    arr = np.asarray([v for v in vals if v != adv.UINT256_MAX], dtype="float64")
    return arr[np.isfinite(arr)]


def _moments(x):
    """(skewness, excess_kurtosis) via adv._series_stats; (nan, nan) if too few rows."""
    if x.size < 4:
        return np.nan, np.nan
    st = adv._series_stats(pd.Series(x, dtype="float64"))
    s, k = st.get("skewness"), st.get("excess_kurtosis")
    return (np.nan if s is None else float(s),
            np.nan if k is None else float(k))


def to_returns(data, column=None, kind="simple"):
    """Period returns from a level series. ``kind`` is 'simple' (Δ/prev) or 'log'."""
    x = _clean(data, column)
    if x.size < 2:
        return np.array([], dtype="float64")
    if kind == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.diff(np.log(x))
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.diff(x) / x[:-1]
    return r[np.isfinite(r)]


def _wealth_index(x, as_returns):
    """A positive level series for drawdown: cumprod(1+returns) or the levels as-is."""
    if as_returns:
        with np.errstate(over="ignore", invalid="ignore"):
            return np.cumprod(1.0 + x)
    return x


def _drawdown_curve(x, as_returns):
    """Drawdown fraction at each point: wealth / running-peak - 1 (<= 0)."""
    w = _wealth_index(x, as_returns)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return w / np.maximum.accumulate(w) - 1.0


def _ann(periods_per_year):
    """sqrt annualisation factor (1.0 when not annualising)."""
    return math.sqrt(periods_per_year) if periods_per_year else 1.0


# --------------------------------------------------------------------------- #
# Tail risk / downside
# --------------------------------------------------------------------------- #
def value_at_risk(data, column=None, level=0.95):
    """Historical VaR: the loss not exceeded with probability ``level`` (positive number)."""
    x = _clean(data, column)
    if x.size == 0:
        return np.nan
    return float(-np.quantile(x, 1.0 - level))


def conditional_var(data, column=None, level=0.95):
    """Conditional VaR / Expected Shortfall: mean loss in the worst ``1-level`` tail."""
    x = _clean(data, column)
    if x.size == 0:
        return np.nan
    cutoff = np.quantile(x, 1.0 - level)
    tail = x[x <= cutoff]
    return float(-tail.mean()) if tail.size else np.nan


def tail_ratio(data, column=None, upper=0.95, lower=0.05):
    """|upper quantile| / |lower quantile| — right-vs-left tail asymmetry."""
    x = _clean(data, column)
    if x.size == 0:
        return np.nan
    lo = abs(np.quantile(x, lower))
    return float(abs(np.quantile(x, upper)) / lo) if lo else np.nan


def hill_tail_index(data, column=None, k=None):
    """Hill estimator of the (right) tail index α; lower α = heavier tail.

    Uses the ``k`` largest positive order statistics (default ~10% of them).
    """
    x = _clean(data, column)
    x = np.sort(x[x > 0])
    n = x.size
    if n < 10:
        return np.nan
    if k is None:
        k = max(10, int(0.1 * n))
    k = min(k, n - 1)
    top = x[-k:]
    hill = np.mean(np.log(top) - np.log(x[-(k + 1)]))
    return float(1.0 / hill) if hill > 0 else np.nan


def downside_deviation(data, column=None, threshold=0.0):
    """Root-mean-square of shortfalls below ``threshold`` (Sortino denominator)."""
    x = _clean(data, column)
    if x.size == 0:
        return np.nan
    short = np.minimum(x - threshold, 0.0)
    return float(np.sqrt(np.mean(short ** 2)))


def upside_deviation(data, column=None, threshold=0.0):
    """Root-mean-square of excesses above ``threshold``."""
    x = _clean(data, column)
    if x.size == 0:
        return np.nan
    up = np.maximum(x - threshold, 0.0)
    return float(np.sqrt(np.mean(up ** 2)))


def semivariance(data, column=None):
    """Variance of values below the mean (downside risk)."""
    x = _clean(data, column)
    if x.size < 2:
        return np.nan
    below = x[x < x.mean()]
    return float(np.mean((below - x.mean()) ** 2)) if below.size else 0.0


# --------------------------------------------------------------------------- #
# Drawdown (level / wealth-index based)
# --------------------------------------------------------------------------- #
def max_drawdown(data, column=None, as_returns=False):
    """Largest peak-to-trough decline (fraction in [0, 1]); input order = time order."""
    x = _clean(data, column)
    if x.size < 2:
        return np.nan
    dd = _drawdown_curve(x, as_returns)
    return float(-np.nanmin(dd))


def average_drawdown(data, column=None, as_returns=False):
    """Pain index — mean depth of the drawdown curve."""
    x = _clean(data, column)
    if x.size < 2:
        return np.nan
    dd = _drawdown_curve(x, as_returns)
    return float(-np.nanmean(dd))


def ulcer_index(data, column=None, as_returns=False):
    """Ulcer index — RMS of the percentage drawdown curve (depth + duration)."""
    x = _clean(data, column)
    if x.size < 2:
        return np.nan
    dd = _drawdown_curve(x, as_returns) * 100.0
    return float(np.sqrt(np.nanmean(dd ** 2)))


# --------------------------------------------------------------------------- #
# Volatility / dispersion
# --------------------------------------------------------------------------- #
def realized_volatility(data, column=None, periods_per_year=None, as_returns=True):
    """Std of returns, optionally annualised. ``as_returns=False`` diffs a level first."""
    x = _clean(data, column) if as_returns else to_returns(data, column)
    if x.size < 2:
        return np.nan
    return float(np.std(x, ddof=1) * _ann(periods_per_year))


def robust_cv(data, column=None, method="mad"):
    """Outlier-resistant CV: 'mad' -> MAD/median, 'iqr' -> IQR/median."""
    x = _clean(data, column)
    if x.size == 0:
        return np.nan
    med = np.median(x)
    if med == 0:
        return np.nan
    if method == "iqr":
        scale = np.quantile(x, 0.75) - np.quantile(x, 0.25)
    else:
        scale = np.median(np.abs(x - med))
    return float(scale / abs(med))


def quartile_coeff_dispersion(data, column=None):
    """(q75 - q25) / (q75 + q25) — scale-free robust dispersion."""
    x = _clean(data, column)
    if x.size == 0:
        return np.nan
    q25, q75 = np.quantile(x, 0.25), np.quantile(x, 0.75)
    denom = q75 + q25
    return float((q75 - q25) / denom) if denom else np.nan


def mean_abs_deviation(data, column=None):
    """Mean absolute deviation around the mean."""
    x = _clean(data, column)
    if x.size == 0:
        return np.nan
    return float(np.mean(np.abs(x - x.mean())))


def coefficient_of_variation(data, column=None):
    """Classic CV = std / mean (sign-sensitive; use robust_cv for heavy tails)."""
    x = _clean(data, column)
    if x.size < 2 or x.mean() == 0:
        return np.nan
    return float(np.std(x, ddof=1) / abs(x.mean()))


# --------------------------------------------------------------------------- #
# Distribution shape (robust moments reused from adv_validation)
# --------------------------------------------------------------------------- #
def moment_skewness(data, column=None):
    """Fisher-Pearson skewness (via adv._series_stats)."""
    return _moments(_clean(data, column))[0]


def excess_kurtosis(data, column=None):
    """Excess kurtosis (via adv._series_stats); 0 = Gaussian, >0 = fat-tailed."""
    return _moments(_clean(data, column))[1]


def bowley_skewness(data, column=None):
    """Quartile (Bowley) skewness in [-1, 1] — robust to outliers."""
    x = _clean(data, column)
    if x.size == 0:
        return np.nan
    q25, q50, q75 = np.quantile(x, [0.25, 0.5, 0.75])
    denom = q75 - q25
    return float((q75 + q25 - 2 * q50) / denom) if denom else np.nan


def kelly_skewness(data, column=None):
    """Kelly (P10/P50/P90) skewness — robust, wider tails than Bowley."""
    x = _clean(data, column)
    if x.size == 0:
        return np.nan
    p10, p50, p90 = np.quantile(x, [0.10, 0.5, 0.90])
    denom = p90 - p10
    return float((p90 + p10 - 2 * p50) / denom) if denom else np.nan


def nonparametric_skew(data, column=None):
    """(mean - median) / std — quick skew sanity check."""
    x = _clean(data, column)
    if x.size < 2:
        return np.nan
    sd = np.std(x, ddof=1)
    return float((x.mean() - np.median(x)) / sd) if sd else np.nan


def bimodality_coefficient(data, column=None):
    """Sarle's bimodality coefficient; > ~0.555 suggests bimodality."""
    x = _clean(data, column)
    n = x.size
    if n < 4:
        return np.nan
    g1, g2 = _moments(x)                      # skew, excess kurtosis
    if np.isnan(g1) or np.isnan(g2):
        return np.nan
    denom = g2 + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return float((g1 ** 2 + 1.0) / denom) if denom else np.nan


def jarque_bera(data, column=None):
    """Jarque-Bera normality statistic: (n/6)(S^2 + K^2/4); larger = less normal."""
    x = _clean(data, column)
    n = x.size
    if n < 4:
        return np.nan
    s, k = _moments(x)
    if np.isnan(s) or np.isnan(k):
        return np.nan
    return float(n / 6.0 * (s ** 2 + (k ** 2) / 4.0))


# --------------------------------------------------------------------------- #
# Concentration / inequality (non-negative magnitudes)
# --------------------------------------------------------------------------- #
def gini_coefficient(data, column=None):
    """Gini in [0, 1] — 0 = perfectly even, 1 = fully concentrated. Negatives dropped."""
    x = np.sort(_clean(data, column))
    x = x[x >= 0]
    n = x.size
    total = x.sum()
    if n == 0 or total == 0:
        return np.nan
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * x)) / (n * total) - (n + 1) / n)


def herfindahl_index(data, column=None):
    """Herfindahl-Hirschman index = Σ shares² in [1/n, 1]; higher = more concentrated."""
    x = _clean(data, column)
    x = x[x >= 0]
    total = x.sum()
    if x.size == 0 or total == 0:
        return np.nan
    shares = x / total
    return float(np.sum(shares ** 2))


def shannon_entropy(data, column=None, bins=20, normalized=True):
    """Histogram Shannon entropy; ``normalized`` scales to [0, 1] (1 = uniform/diffuse)."""
    x = _clean(data, column)
    if x.size == 0:
        return np.nan
    counts, _ = np.histogram(x, bins=bins)
    p = counts[counts > 0] / counts.sum()
    h = float(-np.sum(p * np.log(p)))
    if normalized:
        nb = np.count_nonzero(counts)
        return h / math.log(nb) if nb > 1 else 0.0
    return h


def theil_index(data, column=None):
    """Theil T entropy inequality index (0 = even); positive values only."""
    x = _clean(data, column)
    x = x[x > 0]
    n = x.size
    if n == 0:
        return np.nan
    mu = x.mean()
    if mu == 0:
        return np.nan
    r = x / mu
    return float(np.mean(r * np.log(r)))


def top_k_concentration(data, column=None, k=0.01):
    """Share of total magnitude held by the largest ``k`` fraction of rows (k=0.01 -> top 1%)."""
    x = np.sort(_clean(data, column))[::-1]
    x = x[x >= 0]
    total = x.sum()
    if x.size == 0 or total == 0:
        return np.nan
    m = max(1, int(math.ceil(k * x.size)))
    return float(x[:m].sum() / total)


# --------------------------------------------------------------------------- #
# Risk-adjusted return (input treated as a return / P&L series)
# --------------------------------------------------------------------------- #
def sharpe_ratio(data, column=None, risk_free=0.0, periods_per_year=None):
    """(mean - rf) / std of returns, optionally annualised."""
    x = _clean(data, column)
    if x.size < 2:
        return np.nan
    sd = np.std(x, ddof=1)
    return float((x.mean() - risk_free) / sd * _ann(periods_per_year)) if sd else np.nan


def sortino_ratio(data, column=None, risk_free=0.0, periods_per_year=None):
    """(mean - rf) / downside deviation below ``risk_free``, optionally annualised."""
    x = _clean(data, column)
    if x.size < 2:
        return np.nan
    dd = downside_deviation(x, threshold=risk_free)
    return float((x.mean() - risk_free) / dd * _ann(periods_per_year)) if dd else np.nan


def calmar_ratio(data, column=None, periods_per_year=None, as_returns=True):
    """Annualised mean return / max drawdown."""
    x = _clean(data, column)
    if x.size < 2:
        return np.nan
    mdd = max_drawdown(x, as_returns=as_returns)
    ann_ret = x.mean() * (periods_per_year or 1)
    return float(ann_ret / mdd) if mdd else np.nan


def omega_ratio(data, column=None, threshold=0.0):
    """Omega = Σ gains above threshold / Σ losses below threshold."""
    x = _clean(data, column)
    if x.size == 0:
        return np.nan
    gains = np.sum(np.maximum(x - threshold, 0.0))
    losses = np.sum(np.maximum(threshold - x, 0.0))
    return float(gains / losses) if losses else np.nan


# --------------------------------------------------------------------------- #
# Dependence / memory (input order = time order)
# --------------------------------------------------------------------------- #
def autocorrelation(data, column=None, lag=1):
    """Lag-``lag`` autocorrelation of the series in row order."""
    x = _clean(data, column)
    if x.size <= lag + 1:
        return np.nan
    a, b = x[:-lag], x[lag:]
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def hurst_exponent(data, column=None):
    """Rescaled-range (R/S) Hurst exponent: 0.5 random walk, >0.5 trending, <0.5 mean-reverting."""
    x = _clean(data, column)
    n = x.size
    if n < 20:
        return np.nan
    lags = np.unique(np.floor(np.logspace(np.log10(4), np.log10(n // 2), 20)).astype(int))
    rs, used = [], []
    for lag in lags:
        n_win = n // lag
        if n_win < 1:
            continue
        ratios = []
        for w in range(n_win):
            seg = x[w * lag:(w + 1) * lag]
            dev = np.cumsum(seg - seg.mean())
            spread = dev.max() - dev.min()
            sd = seg.std()
            if sd > 0:
                ratios.append(spread / sd)
        if ratios:
            rs.append(np.mean(ratios))
            used.append(lag)
    if len(used) < 3:
        return np.nan
    return float(np.polyfit(np.log(used), np.log(rs), 1)[0])


# --------------------------------------------------------------------------- #
# Convenience — run every metric over one column
# --------------------------------------------------------------------------- #
def financial_metrics(data, column=None, periods_per_year=None, as_returns=True):
    """Return a {metric_name: value} dict of all metrics for one column/series.

    ``as_returns`` tells the return/drawdown metrics whether the input is already a
    return series (True) or a level/price series (False, in which case it is diffed
    or turned into a wealth index first).
    """
    levels = data if column is None else data
    rets = (_clean(data, column) if as_returns else to_returns(data, column))

    return {
        "value_at_risk_95": value_at_risk(rets, level=0.95),
        "conditional_var_95": conditional_var(rets, level=0.95),
        "tail_ratio_95_05": tail_ratio(data, column),
        "hill_tail_index": hill_tail_index(data, column),
        "downside_deviation": downside_deviation(rets),
        "upside_deviation": upside_deviation(rets),
        "semivariance": semivariance(data, column),
        "max_drawdown": max_drawdown(data, column, as_returns=as_returns),
        "average_drawdown": average_drawdown(data, column, as_returns=as_returns),
        "ulcer_index": ulcer_index(data, column, as_returns=as_returns),
        "realized_volatility": realized_volatility(rets, periods_per_year=periods_per_year),
        "robust_cv_mad": robust_cv(data, column, method="mad"),
        "robust_cv_iqr": robust_cv(data, column, method="iqr"),
        "quartile_coeff_dispersion": quartile_coeff_dispersion(data, column),
        "mean_abs_deviation": mean_abs_deviation(data, column),
        "coefficient_of_variation": coefficient_of_variation(data, column),
        "moment_skewness": moment_skewness(data, column),
        "excess_kurtosis": excess_kurtosis(data, column),
        "bowley_skewness": bowley_skewness(data, column),
        "kelly_skewness": kelly_skewness(data, column),
        "nonparametric_skew": nonparametric_skew(data, column),
        "bimodality_coefficient": bimodality_coefficient(data, column),
        "jarque_bera": jarque_bera(data, column),
        "gini_coefficient": gini_coefficient(data, column),
        "herfindahl_index": herfindahl_index(data, column),
        "shannon_entropy": shannon_entropy(data, column),
        "theil_index": theil_index(data, column),
        "top_1pct_concentration": top_k_concentration(data, column, k=0.01),
        "sharpe_ratio": sharpe_ratio(rets, periods_per_year=periods_per_year),
        "sortino_ratio": sortino_ratio(rets, periods_per_year=periods_per_year),
        "calmar_ratio": calmar_ratio(data, column, periods_per_year=periods_per_year,
                                     as_returns=as_returns),
        "omega_ratio": omega_ratio(rets),
        "autocorrelation_lag1": autocorrelation(data, column, lag=1),
        "hurst_exponent": hurst_exponent(data, column),
    }
