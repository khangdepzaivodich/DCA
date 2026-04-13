import numpy as np
import pandas as pd

class FuzzySignalModel:
    """
    Section 5 & 6: Fuzzy Signal Model (Type-1 Sugeno-style)
    Inputs: Momentum (RSI), Trend Alignment (Slope), Distance from Mean (Z-score)
    Outputs: Direction (-1 to 1) and Confidence (0 to 1)
    """
    def __init__(self):
        pass

    def _membership_low(self, x, a, b):
        if x <= a: return 1.0
        if x >= b: return 0.0
        return (b - x) / (b - a)

    def _membership_mid(self, x, a, b, c):
        if x <= a or x >= c: return 0.0
        if x == b: return 1.0
        if a < x < b: return (x - a) / (b - a)
        return (c - x) / (c - b)

    def _membership_high(self, x, a, b):
        if x <= a: return 0.0
        if x >= b: return 1.0
        return (x - a) / (b - a)

    def compute_signal(self, rsi, slope, z_score):
        """
        Processes normalized technical inputs into a fuzzy directional signal.
        
        Args:
            rsi (float): Relative Strength Index (0-100).
            slope (float): Normalized EMA slope (-1.0 to 1.0).
            z_score (float): Distance from mean (standardized).
            
        Returns:
            dict: direction (-1 to 1) and confidence (0 to 1).
        """
        # 1. Fuzzify Momentum (RSI)
        rsi_oversold   = self._membership_low(rsi, 20, 40)
        rsi_neutral    = self._membership_mid(rsi, 30, 50, 70)
        rsi_overbought = self._membership_high(rsi, 60, 80)
        
        # 2. Fuzzify Trend (Slope)
        trend_down     = self._membership_low(slope, -0.5, 0.0)
        trend_flat     = self._membership_mid(slope, -0.2, 0.0, 0.2)
        trend_up       = self._membership_high(slope, 0.0, 0.5)
        
        # 3. Fuzzify Value (Z-Score)
        price_cheap    = self._membership_low(z_score, -2.5, -1.0)
        price_fair     = self._membership_mid(z_score, -1.5, 0, 1.5)
        price_dear     = self._membership_high(z_score, 1.0, 2.5)

        # 4. Fuzzy Rules (Section 7)
        # Weights (Rules output constants in Sugeno-style)
        # format: (weight, output_direction)
        rules = []
        
        # Trend Rules
        rules.append(((trend_up * rsi_neutral), 0.8))         # Strong Long
        rules.append(((trend_down * rsi_neutral), -0.8))      # Strong Short
        
        # Contrarian Rules (Overextended)
        rules.append(((trend_up * rsi_overbought), 0.2))      # Exhausted Long (Weak)
        rules.append(((trend_down * rsi_oversold), -0.2))     # Exhausted Short (Weak)
        
        # Value Rules (Mean Reversion)
        rules.append(((trend_flat * price_cheap), 0.6))       # Reversion Long
        rules.append(((trend_flat * price_dear), -0.6))       # Reversion Short
        
        # Chop Protection
        rules.append(((trend_flat * rsi_neutral), 0.0))       # No Signal
        
        # 5. Defuzzification (Weighted Average)
        weighted_sum = 0
        total_weight = 0
        
        for weight, direction in rules:
            weighted_sum += weight * direction
            total_weight += weight
            
        if total_weight < 1e-6:
            return {"direction": 0.0, "confidence": 0.0}
            
        final_direction = weighted_sum / total_weight
        confidence = min(total_weight, 1.0) # Heuristic confidence based on rule firing strength
        
        return {
            "direction": float(final_direction),
            "confidence": float(confidence)
        }

def apply_fuzzy_signal_to_df(df):
    """
    Helper to apply the fuzzy signal to a DataFrame.
    Expects columns: 'rsi_15m', 'ema_slope', 'z_score'
    """
    m = FuzzySignalModel()
    results = []
    
    for _, row in df.iterrows():
        res = m.compute_signal(row['rsi_15m'], row['ema_slope'], row['z_score'])
        results.append(res)
        
    res_df = pd.DataFrame(results, index=df.index)
    return pd.concat([df, res_df], axis=1)
