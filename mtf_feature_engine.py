"""
MTF Feature Engine — 4H (Context) + 15m (Core) + Interactions
================================================================
Follows STRICT timeframe roles:
  4H  → CONTEXT ONLY (z, slope, vol)
  15m → CORE MODEL (structure, events, momentum, exhaustion)
"""

import pandas as pd
import numpy as np

# ═══════════════════════════════════════════════════════════
# FEATURE COLUMN REGISTRY (single source of truth)
# ═══════════════════════════════════════════════════════════

HTF_FEATURES = ['z_4h', 'slope_4h', 'vol_4h', 'dist_4h_high', 'dist_4h_low']
INT_FEATURES = ['z_1h', 'slope_1h', 'vol_1h']

LTF_FEATURES = [
    'z_20', 'z_50', 'z_200',
    'r1', 'r3', 'r5', 'ema_slope',
    'vol_20', 'range_expansion',
    'breakout_up', 'breakout_down', 'breakout_strength',
    'dist_to_high', 'dist_to_low',
    'wick_ratio', 'momentum_decay', 'fail_to_extend',
    'rsi_15m', 'adx_15m', 'vol_z_15m',
]

INTERACTION_FEATURES = [
    'cross_z', 'cross_slope_momentum', 'cross_vol_breakout',
    'cross_trend_align', 'cross_vol_range', 'cross_z_wick',
]

EVENT_FEATURES = [
    'event_breakout', 'event_failed_breakout', 'event_impulse',
    'event_compression_break', 'event_direction', 'event_quality',
    'bars_since_event', 'failed_attempts',
]

REGIME_FEATURES = [
    'regime_trend_up', 'regime_trend_down', 'regime_range', 'regime_high_vol',
]

TIME_FEATURES = ['hour_sin', 'hour_cos']
FUNDING_FEATURES = ['funding_rate', 'funding_cumulative', 'funding_velocity']
OFLOW_FEATURES = ['oflow_delta_z', 'oflow_cvd_z', 'oflow_ratio_z', 'oflow_intensity_z']

MODEL_FEATURES = (
    HTF_FEATURES + INT_FEATURES + LTF_FEATURES + INTERACTION_FEATURES +
    EVENT_FEATURES + REGIME_FEATURES + TIME_FEATURES +
    FUNDING_FEATURES + OFLOW_FEATURES
)

META_FEATURES = [
    'P_reversal',
    'regime_trend_up', 'regime_trend_down', 'regime_range', 'regime_high_vol',
    'vol_20', 'vol_4h', 'event_quality', 'atr_pct_15m',
    'oflow_delta_z', 'oflow_cvd_z',
    'dist_4h_high', 'dist_4h_low',
]


# ═══════════════════════════════════════════════════════════
# INDICATOR HELPERS
# ═══════════════════════════════════════════════════════════

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_adx(high, low, close, period=14):
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-9))
    minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-9))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    return dx.rolling(period).mean()


# ═══════════════════════════════════════════════════════════
# 4H FEATURES  —  CONTEXT ONLY  (3 features)
# ═══════════════════════════════════════════════════════════

def build_4h_features(df_1m):
    """4H context: z, slope, vol + RANGE BOUNDARIES."""
    p = df_1m.resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna()

    # z_4h
    rm = p['close'].rolling(50).mean()
    rs = p['close'].rolling(50).std()
    p['z_4h'] = (p['close'] - rm) / (rs + 1e-9)

    # slope_4h
    ema20 = p['close'].ewm(span=20).mean()
    p['slope_4h'] = ema20.pct_change(5) * 100

    # vol_4h
    atr_pct = compute_atr(p['high'], p['low'], p['close'], 14) / p['close'] * 100
    p['vol_4h'] = (atr_pct - atr_pct.rolling(50).mean()) / (atr_pct.rolling(50).std() + 1e-9)

    # Kaufman Efficiency Ratio (KER)
    # KER = Net Change / Total Path (Absolute changes)
    # Range 0.0 (Pure Noise) to 1.0 (Strict Trend)
    diff = p['close'].diff().abs()
    total_path = diff.rolling(10).sum()
    net_change = (p['close'] - p['close'].shift(10)).abs()
    p['efficiency_4h'] = (net_change / (total_path + 1e-9)).fillna(0.5)

    # NEW: 4H Boundaries (% distance from high/low of last 20 bars)
    p['dist_4h_high'] = (p['high'].rolling(20).max() - p['close']) / (p['close'] + 1e-9) * 100
    p['dist_4h_low']  = (p['close'] - p['low'].rolling(20).min()) / (p['close'] + 1e-9) * 100

    # Extras for regime
    p['adx_4h'] = compute_adx(p['high'], p['low'], p['close'], 14)

    cols = ['z_4h', 'slope_4h', 'vol_4h', 'dist_4h_high', 'dist_4h_low', 'adx_4h', 'efficiency_4h']
    p[cols] = p[cols].shift(1)  # shift to prevent lookahead
    return p[cols]


def build_1h_features(df_1m):
    """1H context bridge between 4H and 15m."""
    p = df_1m.resample('1h').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna()

    # z_1h
    rm = p['close'].rolling(50).mean()
    rs = p['close'].rolling(50).std()
    p['z_1h'] = (p['close'] - rm) / (rs + 1e-9)

    # slope_1h
    ema20 = p['close'].ewm(span=20).mean()
    p['slope_1h'] = ema20.pct_change(5) * 100

    # vol_1h
    atr_pct = compute_atr(p['high'], p['low'], p['close'], 14) / p['close'] * 100
    p['vol_1h'] = (atr_pct - atr_pct.rolling(50).mean()) / (atr_pct.rolling(50).std() + 1e-9)

    cols = ['z_1h', 'slope_1h', 'vol_1h']
    p[cols] = p[cols].shift(1)
    return p[cols]


# ═══════════════════════════════════════════════════════════
# 15m FEATURES  —  CORE MODEL  (20 features + price cols)
# ═══════════════════════════════════════════════════════════

def build_15m_features(df_1m):
    """Full 15m feature set: PD, momentum, vol, structure, exhaustion, oflow."""
    # Grouping with aggregation to ensure all columns include taker buys
    p = df_1m.resample('15min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
        'taker_buy_base_vol': 'sum'
    }).dropna()

    c, h, l, v = p['close'], p['high'], p['low'], p['volume']

    # ── Order Flow Proxy ──
    p['taker_ratio'] = p['taker_buy_base_vol'] / (v + 1e-9)

    # ── Multi-scale Price Displacement ──
    # [Rest of logic stays same but with new liquidity columns]
    for n in [20, 50, 200]:
        rm = c.rolling(n).mean(); rs = c.rolling(n).std()
        p[f'z_{n}'] = (c - rm) / (rs + 1e-9)

    p['r1'] = c.pct_change(1) * 100
    p['r3'] = c.pct_change(3) * 100
    p['r5'] = c.pct_change(5) * 100
    p['ema_slope'] = c.ewm(span=12).mean().pct_change(3) * 100

    p['vol_20'] = c.pct_change().rolling(20).std() * 100
    avg_rng = (h - l).rolling(20).mean()
    p['range_expansion'] = (h - l) / (avg_rng + 1e-9)

    # ── Structural Liquidity (Sweeps) ──
    # Daily High/Low
    df_day = df_1m.resample('1D').agg({'high': 'max', 'low': 'min'}).shift(1)
    df_day = df_day.reindex(p.index, method='ffill')
    p['dist_to_pdh'] = (df_day['high'] - c) / (c + 1e-9) * 100
    p['dist_to_pdl'] = (c - df_day['low']) / (c + 1e-9) * 100

    # Weekly High/Low
    df_week = df_1m.resample('W').agg({'high': 'max', 'low': 'min'}).shift(1)
    df_week = df_week.reindex(p.index, method='ffill')
    p['dist_to_pwh'] = (df_week['high'] - c) / (c + 1e-9) * 100
    p['dist_to_pwl'] = (c - df_week['low']) / (c + 1e-9) * 100

    lb = 20
    rh = h.rolling(lb).max().shift(1)
    rl = l.rolling(lb).min().shift(1)
    p['breakout_up'] = (c > rh).astype(float)
    p['breakout_down'] = (c < rl).astype(float)
    p['breakout_strength'] = np.where(
        p['breakout_up'] == 1, (c - rh) / (c * 0.01 + 1e-9),
        np.where(p['breakout_down'] == 1, (rl - c) / (c * 0.01 + 1e-9), 0),
    )
    p['dist_to_high'] = (c - h.rolling(lb).max()) / (c + 1e-9) * 100
    p['dist_to_low']  = (c - l.rolling(lb).min()) / (c + 1e-9) * 100

    # ── Exhaustion ──
    body = (c - p['open']).abs()
    rng = h - l
    p['wick_ratio'] = 1 - body / (rng + 1e-9)
    p['momentum_decay'] = np.where(p['r5'].abs() > 0.01, p['r1'] / (p['r5'] + 1e-9), 0).clip(-5, 5)
    p['fail_to_extend'] = np.where(h >= rh, (h - c) / (rng + 1e-9), np.where(l <= rl, (c - l) / (rng + 1e-9), 0.5))

    p['rsi_15m'] = compute_rsi(c, 14)
    p['adx_15m'] = compute_adx(h, l, c, 14)
    p['vol_z_15m'] = (v - v.rolling(100).mean()) / (v.rolling(100).std() + 1e-9)

    p['atr_15m'] = compute_atr(h, l, c, 14)
    p['atr_pct_15m'] = p['atr_15m'] / c * 100

    hr = p.index.hour + p.index.minute / 60.0
    p['hour_sin'] = np.sin(2 * np.pi * hr / 24)
    p['hour_cos'] = np.cos(2 * np.pi * hr / 24)

    # Shift features by 1 bar
    shift_cols = [
        'z_20', 'z_50', 'z_200', 'r1', 'r3', 'r5', 'ema_slope',
        'vol_20', 'range_expansion', 'breakout_up', 'breakout_down',
        'breakout_strength', 'dist_to_high', 'dist_to_low',
        'wick_ratio', 'momentum_decay', 'fail_to_extend',
        'rsi_15m', 'adx_15m', 'vol_z_15m', 'atr_pct_15m',
        'hour_sin', 'hour_cos', 'taker_ratio', 
        'dist_to_pdh', 'dist_to_pdl', 'dist_to_pwh', 'dist_to_pwl'
    ]
    p[shift_cols] = p[shift_cols].shift(1)
    return p


# ═══════════════════════════════════════════════════════════
# INTERACTION FEATURES  (6 features)
# ═══════════════════════════════════════════════════════════

def build_interaction_features(df):
    """Cross-timeframe interactions — where real edge emerges."""
    df['cross_z']               = df['z_4h'] * df['z_50']
    df['cross_slope_momentum']  = df['slope_4h'] * df['r5']
    df['cross_vol_breakout']    = df['vol_4h'] * df['breakout_strength']
    df['cross_trend_align']     = np.sign(df['slope_4h']) * np.sign(df['ema_slope'])
    df['cross_vol_range']       = df['vol_4h'] * df['range_expansion']
    df['cross_z_wick']          = df['z_4h'] * df['wick_ratio']
    return df


# ═══════════════════════════════════════════════════════════
# FUNDING FEATURES  (3 features, optional)
# ═══════════════════════════════════════════════════════════

def build_funding_features(funding_df):
    """Shift-safe funding features."""
    f = funding_df.copy()
    f = f.resample('8h').last().ffill()
    f['funding_rate'] = f['fundingRate'] * 1000
    f['funding_cumulative'] = f['fundingRate'].rolling(21).sum() * 100
    f['funding_velocity'] = f['fundingRate'].diff().fillna(0)
    cols = ['funding_rate', 'funding_cumulative', 'funding_velocity']
    f[cols] = f[cols].shift(1)
    return f[cols]
