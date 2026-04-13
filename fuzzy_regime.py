import numpy as np
import pandas as pd

class FuzzyRegimeFilter:
    """
    Section 4.2: Fuzzy Regime Filter
    Inputs: Trend Strength (ADX), Volatility Level (ATR Z-score), Price Efficiency (Fractal Efficiency)
    Outputs: Soft probabilities for Trend, Mean Reversion, and No Trade regimes.
    """
    def __init__(self):
        pass

    def _membership_low(self, x, a, b):
        """Linear decay membership function."""
        if x <= a: return 1.0
        if x >= b: return 0.0
        return (b - x) / (b - a)

    def _membership_mid(self, x, a, b, c):
        """Triangular membership function."""
        if x <= a or x >= c: return 0.0
        if x == b: return 1.0
        if a < x < b: return (x - a) / (b - a)
        return (c - x) / (c - b)

    def _membership_high(self, x, a, b):
        """Linear growth membership function."""
        if x <= a: return 0.0
        if x >= b: return 1.0
        return (x - a) / (b - a)

    def gaussian(self, x, mean, sigma):
        return np.exp(-((x - mean)**2) / (2 * sigma**2))

    def compute_regime_weight(self, adx, vol_z, efficiency):
        """
        DCA Regime Weighing:
        Categorizes market into 'Accumulation Quality'.
        Returns 0.0 (Worst to buy) to 2.0 (Best to buy).
        """
        # 1. Membership Functions
        # Trends (Avoid buying high efficiency)
        m_trend = self.gaussian(efficiency, 1.0, 0.2) * self.gaussian(adx, 40, 10)
        # Chop (Ideal for DCA)
        m_chop  = self.gaussian(efficiency, 0.2, 0.2) * self.gaussian(adx, 15, 10)
        # Sleepy (Healthy DCA)
        m_sleep = self.gaussian(vol_z, -1.5, 0.5)

        # 2. Rule Weights (Sugeno)
        # Trend -> 0.2x (Taper off)
        # Chop  -> 1.5x (Boost)
        # Sleep -> 1.2x (Healthy)
        # Default Neutral -> 1.0x
        
        sum_w = m_trend + m_chop + m_sleep + 1e-9
        regime_mult = (m_trend * 0.2 + m_chop * 1.5 + m_sleep * 1.2) / sum_w
        
        return regime_mult

    def compute_regime(self, adx, vol_z, efficiency):
        """
        Calculates soft probabilities for market regimes based on fuzzy rules.
        
        Args:
            adx (float): Trend strength (typically 0-100).
            vol_z (float): Volatility Z-score (standardized ATR).
            efficiency (float): Price efficiency (0.0 to 1.0).
            
        Returns:
            dict: Probabilities and the dominant regime label.
        """
        # 1. Fuzzify Inputs (Membership Degrees)
        
        # Trend Strength (ADX)
        trend_weak   = self._membership_low(adx, 15, 25)
        trend_medium = self._membership_mid(adx, 20, 30, 40)
        trend_strong = self._membership_high(adx, 35, 50)
        
        # Volatility (Vol Z-Score)
        vol_low    = self._membership_low(vol_z, -1.0, 0.0)
        vol_normal = self._membership_mid(vol_z, -0.5, 0.5, 1.5)
        vol_high   = self._membership_high(vol_z, 1.0, 2.5)
        
        # Price Efficiency (Directional vs Noisy)
        noisy       = self._membership_low(efficiency, 0.3, 0.6)
        directional = self._membership_high(efficiency, 0.6, 0.9)

        # 2. Apply Fuzzy Rules (Mamdani-style inference)
        
        # Rule 1: IF trend strong AND efficiency directional AND volatility normal -> TREND
        p_trend = trend_strong * directional * vol_normal
        
        # Rule 2: IF trend NOT strong AND efficiency noisy AND volatility low -> MEAN REVERSION
        p_reversion = max(trend_weak, trend_medium) * noisy * vol_low
        
        # Rule 3: IF volatility high AND efficiency noisy -> NO TRADE (Blow-off/Panic)
        p_no_trade_vol = vol_high * noisy
        
        # Rule 4: IF trend weak AND efficiency noisy -> NO TRADE (Chop)
        p_no_trade_chop = trend_weak * noisy
        
        # Combine No Trade rules
        p_no_trade = max(p_no_trade_vol, p_no_trade_chop)
        
        # 3. Defuzzification (Normalization to probabilities)
        scores = np.array([p_trend, p_reversion, p_no_trade])
        total = np.sum(scores)
        
        # Fallback if no rules fire
        if total < 1e-6:
            return {
                "prob_trend": 0.0,
                "prob_reversion": 0.0,
                "prob_no_trade": 1.0,
                "dominant": "no_trade"
            }
        
        probs = scores / total
        return {
            "prob_trend": float(probs[0]),
            "prob_reversion": float(probs[1]),
            "prob_no_trade": float(probs[2]),
            "dominant": ["trend", "mean_reversion", "no_trade"][np.argmax(probs)]
        }

def apply_fuzzy_regime_to_df(df):
    """
    Helper to apply the fuzzy filter to a DataFrame.
    Expects columns: 'adx_4h', 'vol_4h', 'efficiency_4h'
    """
    f = FuzzyRegimeFilter()
    results = []
    
    for _, row in df.iterrows():
        res = f.compute_regime(row['adx_4h'], row['vol_4h'], row['efficiency_4h'])
        results.append(res)
        
    res_df = pd.DataFrame(results, index=df.index)
    return pd.concat([df, res_df], axis=1)
