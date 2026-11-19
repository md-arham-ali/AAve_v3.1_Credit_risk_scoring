"""Modeling stage — leak-free scaling of the 2h model frames (DF_model_scaled_*).

Every column goes SHAPE -> RANGE, and NOTHING is fitted on a train window, so the scale is the
same for any split proportion. Two range methods, built side by side to compare:

- FIXED (scale_fixed): formulas + hand-set constants from the notebook config. No statistic of
  the data is used anywhere, so a row's score never depends on any other row. The constants
  are REFERENCE POINTS, not limits. Non-negative mappings: floor 0, no upper cap, 100 = at the
  reference. Signed mappings: bounded 0–100, 50 = center.
    ceiling            100·max(shape(x) / shape(reference), 0)            amounts, counts (0 -> 0)
                       counts + unique counts: ln(1+x), "count"; rolled ones: log10(1+x), "count_rolled"
    symmetric_ceiling  50 + 50·clip(asinh(x/S) / asinh(reference/S), -1, 1)   signed flows
    percent            100·x, raises outside [0, 1]                       shares
    odds               50·log2(1 + r)        (0 -> 0, 1 -> 50, 3 -> 100, unbounded)  ratios
    tanh               50 + 50·tanh((x - center) / c)                     signed rates, momentum
    positive_tanh      100·asinh(x / c) / asinh(1)   (0 -> 0, c -> 100, unbounded)  non-neg rates
    log_range          100·max((log10(x + 10^lo) - lo) / (hi - lo), 0)    tiny positives
  (the keys "odds" / "positive_tanh" are kept so the notebook's SCALE_CONSTANTS stay valid)
- CAUSAL (scale_causal): each row is min-max scaled against ITS OWN PAST only — the min and max
  of the shaped column over (t - window, t]. Leak-free by construction (no row after t is ever
  read); adapts to regime drift; the same raw value scores differently at different times.

Every fixed mapping is monotone, including above the reference, so the order of values is
preserved (scaling_report checks it). Causal stays in [0, 100] by construction.
Raw frames are never changed: both methods return new frames.

TARGETS (DF_target_*) use the same fixed scaling, and every mapping they use has a REVERSE, so
a predicted score turns back into USD / a ratio (unscale_targets). The one difference: a signed
target uses symmetric_open (no ±reference clip), since a clipped value cannot be reversed.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json

import numpy as np
import pandas as pd

from model_dataset import _as_time_index, window_suffixes, TARGET_WINDOWS, FORWARD_SUFFIX
from model_config import TIME_COL


# --- shapes (one per transformation; each refuses input outside its domain) ----------- #

def _check(s, ok, name, rule):
    """Raise if any non-NaN value of s fails `ok` — a shape never clips silently."""
    bad = s.notna() & ~ok
    if bad.any():
        raise ValueError(f"{name}: {int(bad.sum())} value(s) outside the domain of {rule}, "
                         f"e.g. {s[bad].head(3).tolist()}")


def _clip_float_noise(s, tol=1e-6):
    """A tiny negative (e.g. -7e-9) in a column that sums only non-negative inputs — like a
    rolling USD sum crossing a window boundary — is float cancellation noise, never a real
    value; snap it to exactly 0. Anything more negative than `tol` is left untouched, so
    `_check` still raises on an actual negative (a real definition bug)."""
    s = s.astype(float)
    noise = s.notna() & (s < 0) & (s > -tol)
    return s.where(~noise, 0.0)


def identity_shape(s, **_):
    """x unchanged (ratios, shares, rates, spreads, binaries)."""
    return s.astype(float)


def log1p_shape(s, base=np.e, **_):
    """log_base(1 + x) for x >= 0 — counts (base e unrolled, base 10 rolled), heavy ratios."""
    s = _clip_float_noise(s)
    _check(s, s >= 0, s.name, "log1p (x >= 0)")
    return np.log1p(s) / np.log(base)


def sqrt_log_shape(s, base=100, **_):
    """log_base(1 + sqrt(x)) for x >= 0 — USD/base amounts."""
    s = _clip_float_noise(s)
    _check(s, s >= 0, s.name, "sqrt_log (x >= 0)")
    return np.log1p(np.sqrt(s)) / np.log(base)


def asinh_shape(s, scale=1.0, **_):
    """asinh(x / scale) — signed flows. Linear for |x| << scale, log-like beyond it, sign kept.
    scale is a FIXED constant from the notebook config (e.g. $1M), never fitted on data."""
    s = s.astype(float)
    if not scale or scale <= 0:
        raise ValueError(f"{s.name}: asinh scale must be a positive constant, got {scale}")
    return np.arcsinh(s / scale)


def log_floor_shape(s, base=10, floor=None, **_):
    """log_base(x + floor) for x >= 0 — tiny positives (volatility, accrual velocity) that span
    decades. floor is a FIXED constant (10^lo of the log range), so an exact 0 stays finite."""
    s = _clip_float_noise(s)
    _check(s, s >= 0, s.name, "log_floor (x >= 0)")
    if not floor or floor <= 0:
        raise ValueError(f"{s.name}: log_floor needs a positive constant floor, got {floor}")
    return np.log(s + floor) / np.log(base)


SHAPES = {"identity": identity_shape, "log1p": log1p_shape, "sqrt_log": sqrt_log_shape,
          "asinh": asinh_shape, "log_floor": log_floor_shape}


def apply_shape(s, rule):
    """Run the rule's shape on a series — the single entry point every other function uses."""
    return SHAPES[rule["shape"]](s, base=rule.get("base"), floor=rule.get("floor"),
                                 scale=rule.get("scale"))


# --- categories ----------------------------------------------------------------------- #

# category -> (shape, log base, fixed mapping). The shape feeds the causal method and the
# ceiling mappings; the other fixed mappings read the RAW value (see RAW_INPUT_MAPS).
CATEGORY_RULES = {
    "amount":        ("sqrt_log",  100,   "ceiling"),
    "signed_amount": ("asinh",     None,  "symmetric_ceiling"),
    "count":         ("log1p",     np.e,  "ceiling"),          # rolled -> base 10, count_rolled
    "unique_count":  ("log1p",     np.e,  "ceiling"),          # same as count (COUNT_CATEGORIES)
    "share":         ("identity",  None,  "percent"),
    "ratio":         ("identity",  None,  "odds"),             # heavy -> log1p base 10 (causal)
    "signed_rate":   ("identity",  None,  "tanh"),
    "momentum":      ("identity",  None,  "tanh"),
    "positive_rate": ("identity",  None,  "positive_tanh"),
    "tiny_positive": ("log_floor", 10,    "log_range"),
    "binary":        ("identity",  None,  "none"),
    "skip":          ("identity",  None,  "none"),
}
RAW_INPUT_MAPS = {"percent", "odds", "tanh", "positive_tanh", "log_range"}
# Counts and unique counts share ONE scaling: ln(1 + x) with the "count" ceiling; only their
# ROLLED versions differ — log10(1 + x) with the "count_rolled" ceiling.
COUNT_CATEGORIES = ("count", "unique_count")


def _variants(base, columns, *, suffixes, groups=()):
    """The listed column itself, its rolled-time variants (base_rolled_12h, base_rolled_3d, …,
    whichever windows `suffixes` names), and its per-group variants (base_WBTC, from a wide
    pivot), whichever exist in `columns`. The two suffix kinds are independent axes: a rolled
    variant is never also a group variant. `suffixes` has no default — the caller must say
    which window set it's matching (model_dataset.window_suffixes builds it from a window
    set), so a stale/forgotten window set can't silently leave new rolled columns uncategorised.
    Returns [(column, rolled, group), ...] — group is the matched group name, or None."""
    tagged = [(base, False, None)]
    tagged += [(f"{base}_{sfx}", True, None) for sfx in suffixes]
    tagged += [(f"{base}_{g}", False, g) for g in groups]
    seen, out = set(), []
    for col, rolled, group in tagged:
        if col in columns and col not in seen:
            seen.add(col)
            out.append((col, rolled, group))
    return out


def categorize_columns(df, types, *, suffixes, heavy=(), skip=(TIME_COL,), groups=()):
    """Assign every column of df a scaling rule, from {category: base column names}.
    A base name also claims its rolled variants (whichever windows `suffixes` names — pass
    model_dataset.window_suffixes(active_window, active_suffix_template), not a hardcoded
    24h-only tuple) and, when `groups` is given (e.g. a wide per-token pivot's token list), its
    per-group variants (base_WBTC) — a base that was dropped and survives only as one of those
    is fine. One constant per base feature then still covers every group (e.g. one `tanh`
    setting for `rate_change` across all 5 tokens), same as before any pivot. heavy: ratio base
    names whose causal shape is log10(1 + r) (spikes like capital_efficiency would flatten a
    window). `suffixes` is required, not defaulted: passing a stale or too-narrow set is exactly
    how a window-set change used to leave new rolled columns silently uncategorised.
    Returns: {column: rule dict, incl. 'source' = the base name the constants are keyed by,
    'group' = the matched group name or None}.
    Raises on a listed name with no column, a column claimed twice, or a column left
    uncategorised — nothing is ever left unscaled by accident."""
    unknown = set(types) - set(CATEGORY_RULES)
    if unknown:
        raise ValueError(f"unknown categories {sorted(unknown)}; known: {list(CATEGORY_RULES)}")
    cols, colmap, owner = list(df.columns), {}, {}
    for category, bases in types.items():
        for base in bases:
            found = _variants(base, cols, suffixes=suffixes, groups=groups)
            if not found:
                raise KeyError(f"{category}: '{base}' has no column (nor a rolled/group "
                               f"variant) in the frame")
            for col, rolled, group in found:
                if col in owner:
                    raise ValueError(f"'{col}' claimed by both {owner[col]} and {category}")
                owner[col] = category
                shape, log_base, fixed = CATEGORY_RULES[category]
                if category in COUNT_CATEGORIES and rolled:
                    log_base = 10
                if category == "ratio" and base in heavy:
                    shape, log_base = "log1p", 10
                colmap[col] = {"category": category, "source": base, "shape": shape,
                               "base": log_base, "fixed": fixed, "rolled": rolled,
                               "group": group}
    for col in skip:
        if col in cols and col not in colmap:
            colmap[col] = {"category": "skip", "source": col, "shape": "identity", "base": None,
                           "fixed": "none", "rolled": False, "group": None}
    left = [c for c in cols if c not in colmap]
    if left:
        raise ValueError(f"{len(left)} column(s) not categorised: {left}")
    return colmap


def category_table(colmap):
    """The rule per column as a table, sorted by category — what the notebook displays."""
    rows = [{"column": c, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}}
            for c, r in colmap.items()]
    return pd.DataFrame(rows).sort_values(["category", "column"]).reset_index(drop=True)


def check_category_assumptions(df, colmap, float_tol=1e-6):
    """Per column: min, max, share negative / zero / NaN, and whether the shape's domain holds.
    Run before scaling — a negative in a sqrt/log column is a definition bug, not a scaling one.
    `min`/`neg_pct` report the RAW values (so float noise is still visible here), but
    `domain_ok` applies the same `float_tol` tolerance the shapes themselves use
    (_clip_float_noise) — a -7e-9 from a rolling sum's cancellation noise won't crash
    scale_fixed, so it shouldn't fail this check either; a real negative still does."""
    needs_nonneg = {"log1p", "sqrt_log", "log_floor"}
    rows = []
    for col, rule in colmap.items():
        if rule["fixed"] == "none" or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        s = df[col].astype(float)
        real_negative = s.notna() & (s < -float_tol)
        rows.append({"column": col, "category": rule["category"], "shape": rule["shape"],
                     "min": s.min(), "max": s.max(),
                     "neg_pct": round(100 * (s < 0).mean(), 2), "zero_pct": round(100 * (s == 0).mean(), 2),
                     "nan_pct": round(100 * s.isna().mean(), 2),
                     "domain_ok": not (rule["shape"] in needs_nonneg and real_negative.any())})
    return pd.DataFrame(rows)


# --- constants: everything the rules need, from the notebook config, never from data ----- #

def resolve_rule(rule, constants):
    """The rule plus its hand-set constants. constants (notebook SCALE_CONSTANTS):
    ceiling {amount, count, count_rolled: reference raw value} (count / count_rolled serve both
    counts and unique counts), asinh_scale {source: S}, asinh_ceiling (raw $),
    tanh {source: (center, c)}, positive_tanh {source: c}, log_range (lo, hi).
    Raises when a column needs a constant that is not set — no silent default for a scale."""
    r = dict(rule)
    src, cat = r["source"], r["category"]
    if r["shape"] == "asinh":
        r["scale"] = float(constants.get("asinh_scale", {}).get(src, 1.0))
    if r["shape"] == "log_floor":
        r["floor"] = 10.0 ** constants["log_range"][0]
    try:
        if r["fixed"] == "ceiling":
            if cat in COUNT_CATEGORIES:
                key = "count_rolled" if r["rolled"] else "count"
            else:
                key = cat
            r["ceiling"] = float(constants["ceiling"][key])
        elif r["fixed"] in ("symmetric_ceiling", "symmetric_open"):
            r["ceiling"] = float(constants["asinh_ceiling"])
        elif r["fixed"] == "tanh":
            r["center"], r["c"] = map(float, constants["tanh"][src])
        elif r["fixed"] == "positive_tanh":
            r["c"] = float(constants["positive_tanh"][src])
        elif r["fixed"] == "log_range":
            r["lo"], r["hi"] = map(float, constants["log_range"])
    except KeyError as exc:
        raise KeyError(f"'{src}' ({cat}, {r['fixed']}) needs constant {exc} in SCALE_CONSTANTS") from exc
    return r


# --- fixed mappings (one per formula; constants only) --------------------------------- #

# NON-NEGATIVE mappings: floor 0, NO upper cap. The constant is a reference point, not a limit:
# 100 = "at the reference", and a value above it scores above 100, still in order.
# SIGNED mappings (symmetric_ceiling, tanh) stay bounded to 0–100 (Ali, 2026-10-31).

def _floor0(score):
    """Cap only the low end at 0; NaN stays NaN."""
    return score.clip(lower=0)


def ceiling_score(shaped, ceil_shaped):
    """100·max(y / ceil, 0): the shaped value relative to the shaped reference ceiling.
    Above the ceiling the score keeps rising (log-shaped, so slowly: 10x an amount ceiling ≈ 109)."""
    return 100.0 * _floor0(shaped / ceil_shaped)


def symmetric_ceiling_0_100(shaped, ceil_shaped):
    """50 + 50·clip(y / ceil, -1, 1): 0 -> 50, +ceiling -> 100, -ceiling -> 0. Signed: bounded."""
    return 50.0 + 50.0 * (shaped / ceil_shaped).clip(-1, 1)


def symmetric_open_score(shaped, ceil_shaped):
    """50 + 50·y / ceil with NO clip: 0 -> 50, ±ceiling -> 100 / 0, beyond keeps going.
    Targets only — unlike symmetric_ceiling_0_100 every value can be reversed."""
    return 50.0 + 50.0 * shaped / ceil_shaped


def percent_score(x):
    """100·x for a 0–1 share. A value outside [0, 1] is an upstream data bug, so it raises
    instead of being clipped silently."""
    x = x.astype(float)
    _check(x, (x >= -1e-9) & (x <= 1 + 1e-9), x.name, "share (0 <= x <= 1)")
    return 100.0 * x


def log2_ratio_score(r):
    """50·log2(1 + r) for r >= 0: 0 -> 0, 1 -> 50, 3 -> 100, 7 -> 150 — unbounded, log-slow."""
    r = r.astype(float)
    _check(r, r >= 0, r.name, "ratio (r >= 0)")
    return 50.0 * np.log2(1.0 + r)


def tanh_0_100(x, center, c):
    """50 + 50·tanh((x - center) / c): near-linear within +-c of center, bends to 0/100 beyond.
    Signed: bounded."""
    return 50.0 + 50.0 * np.tanh((x.astype(float) - center) / c)


def asinh_positive_score(x, c):
    """100·asinh(x / c) / asinh(1) for x >= 0: 0 -> 0, x = c (the reference) -> 100;
    near-linear below c, log-like above it, no upper cap. Negatives floored to 0."""
    return 100.0 * np.arcsinh(_floor0(x.astype(float)) / c) / np.arcsinh(1.0)


def log_range_score(shaped, lo, hi):
    """100·max((log10(x + 10^lo) - lo) / (hi - lo), 0) on the log_floor-shaped value:
    x = 10^hi -> 100, larger x keeps rising."""
    return 100.0 * _floor0((shaped - lo) / (hi - lo))


def _shaped_ceiling(rule, name=None):
    """The rule's reference value pushed through its own shape."""
    return float(apply_shape(pd.Series([rule["ceiling"]], name=name), rule).iloc[0])


def scale_column_fixed(s, rule):
    """One column through its resolved fixed mapping."""
    m = rule["fixed"]
    if m == "none":
        return s
    if m in ("ceiling", "symmetric_ceiling", "symmetric_open"):
        fn = {"ceiling": ceiling_score, "symmetric_ceiling": symmetric_ceiling_0_100,
              "symmetric_open": symmetric_open_score}[m]
        return fn(apply_shape(s, rule), _shaped_ceiling(rule, s.name))
    if m == "percent":
        return percent_score(s)
    if m == "odds":
        return log2_ratio_score(s)
    if m == "tanh":
        return tanh_0_100(s, rule["center"], rule["c"])
    if m == "positive_tanh":
        return asinh_positive_score(s, rule["c"])
    return log_range_score(apply_shape(s, rule), rule["lo"], rule["hi"])


def scale_fixed(df, colmap, constants):
    """FIXED method: a scaled copy of df, every column through formula + constant. Row-wise, so
    the result for any row is identical whatever other rows (or split) the frame holds."""
    out = df.copy()
    for col, rule in colmap.items():
        out[col] = scale_column_fixed(df[col], resolve_rule(rule, constants))
    print(f" fixed scaling: {sum(r['fixed'] != 'none' for r in colmap.values())} column(s), "
          f"no statistic of the data used")
    return out


# --- reverse (targets: predicted score -> USD / ratio) ---------------------------------- #

def inverse_sqrt_log(y, base=100, **_):
    """Reverse of sqrt_log_shape: x = (base^y - 1)²; y below 0 is read as 0."""
    return (np.power(float(base), y.astype(float).clip(lower=0)) - 1.0) ** 2


def inverse_asinh(y, scale=1.0, **_):
    """Reverse of asinh_shape: x = scale·sinh(y)."""
    return scale * np.sinh(y.astype(float))


def inverse_identity(y, **_):
    return y.astype(float)


INVERSE_SHAPES = {"identity": inverse_identity, "sqrt_log": inverse_sqrt_log,
                  "asinh": inverse_asinh}


def invert_shape(y, rule):
    """Run the reverse of the rule's shape."""
    if rule["shape"] not in INVERSE_SHAPES:
        raise ValueError(f"shape '{rule['shape']}' has no reverse")
    return INVERSE_SHAPES[rule["shape"]](y, base=rule.get("base"), scale=rule.get("scale"))


def inverse_ceiling_score(score, ceil_shaped):
    """Reverse of ceiling_score: shaped = score / 100 · ceil; a score below 0 is read as 0."""
    return _floor0(score.astype(float)) / 100.0 * ceil_shaped


def inverse_symmetric_open_score(score, ceil_shaped):
    """Reverse of symmetric_open_score: shaped = (score - 50) / 50 · ceil."""
    return (score.astype(float) - 50.0) / 50.0 * ceil_shaped


def inverse_percent_score(score):
    """Reverse of percent_score: x = score / 100, the score clipped to [0, 100] first."""
    return score.astype(float).clip(0, 100) / 100.0


TARGET_MAP_SWAP = {"symmetric_ceiling": "symmetric_open"}
REVERSIBLE = {"ceiling", "symmetric_open", "percent"}


def unscale_column_fixed(score, rule):
    """One target column from its score back to the raw unit, through its resolved rule."""
    m = rule["fixed"]
    if m == "none":
        return score
    if m == "percent":
        return inverse_percent_score(score)
    if m not in ("ceiling", "symmetric_open"):
        raise ValueError(f"mapping '{m}' has no reverse — only {sorted(REVERSIBLE)}")
    fn = inverse_ceiling_score if m == "ceiling" else inverse_symmetric_open_score
    return invert_shape(fn(score, _shaped_ceiling(rule, score.name)), rule)


def target_colmap(df, types, window=TARGET_WINDOWS, suffix=FORWARD_SUFFIX):
    """categorize_columns for a target frame: {category: base names}, each base claiming its
    forward columns (base_fwd_24h …). The signed map is swapped for its unclipped version.
    Raises on a mapping with no reverse — a target that cannot be read back is no use."""
    suffixes = window_suffixes(window, suffix)
    colmap = categorize_columns(df, types, suffixes=suffixes)
    for col, rule in colmap.items():
        rule["fixed"] = TARGET_MAP_SWAP.get(rule["fixed"], rule["fixed"])
        if rule["fixed"] not in REVERSIBLE | {"none"}:
            raise ValueError(f"target '{col}' ({rule['category']}) maps by '{rule['fixed']}', "
                             f"which has no reverse — use one of {sorted(REVERSIBLE)}")
    return colmap


def scale_targets(df, colmap, constants):
    """Target frame -> scores, with the same fixed scaling as the features (scale_fixed)."""
    return scale_fixed(df, colmap, constants)


def unscale_targets(scores, colmap, constants):
    """Scores (a scaled target frame, or a frame of predictions) -> raw units. Only the
    columns present in `scores` are reversed; the rest is passed through."""
    out = scores.copy()
    for col, rule in colmap.items():
        if col in out.columns:
            out[col] = unscale_column_fixed(out[col], resolve_rule(rule, constants))
    return out


def target_unscaler(target_frame, column, config):
    """The reverse of ONE target column as a function: score array -> raw-unit array.
    target_frame: the scaled target frame the column lives in (it supplies the categories);
    config: target_scaling_config.json as saved by the targets section of model_dataset.
    Built from the saved config, so a model is read back with the constants its target used."""
    colmap = target_colmap(target_frame, config["types"], window=tuple(config["windows"]))
    if column not in colmap:
        raise KeyError(f"'{column}' is not a target in this scaling config")
    rule = resolve_rule(colmap[column], config["constants"])
    return lambda scores: unscale_column_fixed(
        pd.Series(np.asarray(scores, dtype=float), name=column), rule).to_numpy()


def round_trip_report(raw, colmap, constants):
    """unscale(scale(x)) vs x per target column: max absolute and relative error
    (relative to max(|x|, 1)). Must be ~0 — anything else means a reverse is wrong."""
    back = unscale_targets(scale_targets(raw, colmap, constants), colmap, constants)
    rows = []
    for col, rule in colmap.items():
        if rule["fixed"] == "none":
            continue
        x, y = raw[col].astype(float), back[col].astype(float)
        err = (y - x).abs()
        rows.append({"column": col, "fixed": rule["fixed"], "max_abs_err": err.max(),
                     "max_rel_err": (err / x.abs().clip(lower=1.0)).max(),
                     "nan_same": bool((x.isna() == y.isna()).all())})
    return pd.DataFrame(rows)


# --- causal mapping (each row against its own past only) ------------------------------ #

def causal_minmax_0_100(shaped, window, min_rows):
    """Min-max of a TIME-INDEXED series against its own trailing window (t - window, t].
    Rows whose window holds fewer than min_rows observed values -> NaN (warm-up); a flat
    window (max == min) -> 50. Reads nothing after t."""
    roll = shaped.rolling(window, closed="right", min_periods=min_rows)
    lo, hi = roll.min(), roll.max()
    span = hi - lo
    out = (100.0 * (shaped - lo) / span.where(span > 0)).clip(0, 100)   # clip: float rounding only
    return out.where(span > 0, 50.0).where(shaped.notna() & lo.notna())


def _scale_causal_block(df, colmap, constants, window, min_rows, time_col):
    """scale_causal for a frame with one row per stamp (one asset of the market frame)."""
    indexed, rank = _as_time_index(df, time_col=time_col)
    out = df.copy()
    for col, rule in colmap.items():
        if rule["fixed"] == "none":
            continue
        shaped = apply_shape(indexed[col], resolve_rule(rule, constants))
        out[col] = causal_minmax_0_100(shaped, window, min_rows).to_numpy()[rank]
    return out


def scale_causal(df, colmap, constants, window="30d", min_rows=84, by=None, time_col=TIME_COL):
    """CAUSAL method: a scaled copy of df, each column min-max scaled against its own trailing
    window. by: group column (asset_symbol) — each group is scaled against its own past.
    Returns the frame; warm-up rows are NaN (count printed)."""
    if not by:
        out = _scale_causal_block(df, colmap, constants, window, min_rows, time_col)
    else:
        parts = [_scale_causal_block(part, colmap, constants, window, min_rows, time_col)
                 for _, part in df.groupby(by, sort=False)]
        out = pd.concat(parts).loc[df.index]
    cols = [c for c, r in colmap.items() if r["fixed"] != "none"]
    added = int((out[cols].isna() & df[cols].notna()).any(axis=1).sum())
    print(f" causal scaling over trailing {window} (min {min_rows} rows): {len(cols)} column(s), "
          f"{added} row(s) with warm-up NaN" + (f", per {by}" if by else ""))
    return out


# --- QA ------------------------------------------------------------------------------- #

def _monotone(raw, scaled):
    """True if ordering by raw never makes scaled go down — order of moves preserved."""
    ok = raw.notna() & scaled.notna()
    if ok.sum() < 2:
        return True
    order = np.argsort(raw[ok].to_numpy(), kind="stable")
    return bool((np.diff(scaled[ok].to_numpy()[order]) >= -1e-9).all())


def scaling_report(raw, scaled, colmap, by=None):
    """Per scaled column: scaled p0/p25/p50/p75/p100, IQR, % at 0, % exactly at 100, % above the
    reference (> 100: non-negative fixed mappings only), % NaN added (causal warm-up), monotone
    and Spearman vs raw (within each group when `by`).
    bunched = IQR under 5 points; pct_at_bound = % of rows pinned at 0 or exactly 100 (the
    bounded ends). A high pct_above_ref means the column's reference constant is set too low —
    change it by hand, never fit it. Fixed scaling must be monotone; causal is relative to each
    row's own window, so it is not."""
    keys = raw[by].astype(str) if by else pd.Series("_all", index=raw.index)
    rows = []
    for col, rule in colmap.items():
        if rule["fixed"] == "none" or col == by:
            continue
        r, s = raw[col].astype(float), scaled[col].astype(float)
        q = s.quantile([0, .25, .5, .75, 1]).to_numpy()
        rows.append({"column": col, "category": rule["category"], "fixed": rule["fixed"],
                     "min": q[0], "p25": q[1], "p50": q[2], "p75": q[3], "max": q[4],
                     "iqr": q[3] - q[1],
                     "pct_at_0": 100 * (s <= 1e-9).mean(),
                     "pct_at_100": 100 * ((s - 100).abs() <= 1e-9).mean(),
                     "pct_above_ref": 100 * (s > 100 + 1e-9).mean(),
                     "pct_nan_added": 100 * (s.isna() & r.notna()).mean(),
                     "monotone": all(_monotone(r[keys == k], s[keys == k]) for k in keys.unique()),
                     "spearman": min(r[keys == k].corr(s[keys == k], method="spearman")
                                     for k in keys.unique())})
    out = pd.DataFrame(rows).round(3)
    out["bunched"] = out["iqr"] < 5
    out["pct_at_bound"] = out["pct_at_0"] + out["pct_at_100"]
    return out


def compare_scalings(fixed, causal, colmap, by=None):
    """Fixed vs causal per column: Spearman between the two scores (rows where both exist),
    median absolute gap in points, and each method's IQR — do they tell the same story?"""
    rows = []
    for col, rule in colmap.items():
        if rule["fixed"] == "none" or col == by:
            continue
        f, c = fixed[col].astype(float), causal[col].astype(float)
        both = f.notna() & c.notna()
        rows.append({"column": col, "category": rule["category"],
                     "spearman_fixed_vs_causal": f[both].corr(c[both], method="spearman"),
                     "median_abs_gap": (f[both] - c[both]).abs().median(),
                     "iqr_fixed": f.quantile(.75) - f.quantile(.25),
                     "iqr_causal": c.quantile(.75) - c.quantile(.25)})
    return pd.DataFrame(rows).round(3)


# --- persistence ---------------------------------------------------------------------- #

def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_scaling_config(config, folder, name="scaling_config.json"):
    """Write the hand-set constants, categories and causal window as JSON — nothing in it was
    fitted, so it fully reproduces both scalings on any data. Returns the path."""
    path = os.path.join(str(folder), name)
    with open(path, "w") as fh:
        json.dump(_jsonable(config), fh, indent=2)
    print(f" wrote scaling config -> {path}")
    return path
