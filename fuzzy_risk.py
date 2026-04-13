import numpy as np

class FuzzyRiskEngine:
    """
    Section 10: Fuzzy Risk Engine
    Adjusts position size, stops, and targets based on Regime + Signal Confidence + Volatility.
    """
    def __init__(self, base_risk_pct=0.01, max_risk_pct=0.05):
        self.base_risk = base_risk_pct
        self.max_risk = max_risk_pct

    def compute_risk_params(self, regime, confidence, vol_z, atr_pct):
        """
        Calculates position size and stop/target multipliers.
        
        Args:
            regime (dict): Output from FuzzyRegimeFilter (prob_trend, prob_reversion, etc.)
            confidence (float): Output from FuzzySignalModel (0 to 1).
            vol_z (float): Volatility Z-score.
            atr_pct (float): ATR as % of price.
            
        Returns:
            dict: position_size_mult, sl_multiplier, tp_multiplier
        """
        # 1. Position Size adjustment
        # Base multiplier from signal confidence
        size_mult = confidence 
        
        # Regime override: Reduce size in 'No Trade' or 'High Vol' regimes
        if regime['dominant'] == 'no_trade':
            size_mult *= 0.2
        elif regime['dominant'] == 'mean_reversion':
            size_mult *= 0.7  # Typically lower edge than trend
            
        # 2. Stop Loss Multiplier (Volatility Adaptation)
        # If vol is high, we need wider stops
        sl_mult = 1.0 + max(0, vol_z * 0.5) 
        
        # 3. Take Profit Multiplier (Regime Adaptation)
        if regime['dominant'] == 'trend':
            tp_mult = 3.5 # Capturing Macro extensions
        elif regime['dominant'] == 'mean_reversion':
            tp_mult = 2.0 # Standard rotation
        else:
            tp_mult = 1.0
            
        # 4. Mandatory Safety Caps & Gate
        final_risk = min(self.base_risk * size_mult, self.max_risk)
        
        # STRICT GATE: Sitting out is the ultimate Alpha
        is_tradeable = True
        if regime['dominant'] == 'no_trade' or regime['prob_no_trade'] > 0.4:
            is_tradeable = False
        
        return {
            "risk_pct": float(final_risk),
            "sl_mult": float(sl_mult),
            "tp_mult": float(tp_mult),
            "is_tradeable": is_tradeable
        }

    def calculate_trade_levels(self, price, direction, atr, sl_mult, tp_mult):
        """Calculates absolute SL and TP levels."""
        if direction > 0:
            sl = price - (atr * sl_mult)
            tp = price + (atr * tp_mult * 2.0) # Risk/Reward scaling
        else:
            sl = price + (atr * sl_mult)
            tp = price - (atr * tp_mult * 2.0)
            
        return {"entry": price, "sl": sl, "tp": tp}
