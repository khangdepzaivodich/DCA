import pandas as pd
import numpy as np
import os
from mtf_feature_engine import build_4h_features, build_15m_features

class FuzzyDCAEngine:
    """
    Pure Fuzzy Logic (Sugeno-style) DCA Accumulator.
    Uses continuous membership functions for smooth weighting.
    """
    def gaussian(self, x, mean, sigma):
        return np.exp(-((x - mean)**2) / (2 * sigma**2))

    def compute_dca_weight(self, rsi, z_score):
        # ── 1. Fuzzification (Membership Values) ──
        m_dip  = self.gaussian(z_score, -2.5, 0.8)
        m_neut = self.gaussian(z_score, 0.0, 1.2)
        m_peak = self.gaussian(z_score, 2.5, 0.8)
        
        m_low  = self.gaussian(rsi, 25, 10)
        m_mid  = self.gaussian(rsi, 50, 15)
        m_high = self.gaussian(rsi, 75, 10)

        # ── 2. Rule Confluence ──
        w1 = m_dip  * m_low   # Deep Value
        w2 = m_dip  * m_mid   # Good Value
        w3 = m_neut           # Standard 
        w4 = m_peak * m_high  # Euphoria (Avoid Buying)

        # ── 3. Defuzzification ──
        y1, y2, y3, y4 = 5.0, 2.0, 1.0, 0.0
        sum_w = w1 + w2 + w3 + w4 + 1e-9
        return (w1*y1 + w2*y2 + w3*y3 + w4*y4) / sum_w

    def compute_harvest_weight(self, rsi, z_score):
        """
        Determines when to take profits (Sell Side).
        Outputs % of portfolio to sell (0.0 to 0.05).
        """
        # MF for Euphoria
        m_euphoria = self.gaussian(z_score, 3.5, 0.5)
        m_pumping  = self.gaussian(rsi, 85, 10)
        
        # Rule: IF Z is Deep Euphoria AND RSI is Overheated → Harvest
        sell_signal = m_euphoria * m_pumping
        
        # Scaling: Up to 5% sell every 4 hours during extreme peaks
        return float(sell_signal * 0.05)

def run_fuzzy_dca_backtest(csv_path='data/BTCUSDT_futures_1m.csv'):
    print(f"[*] Starting FAIR FIGHT DCA Backtest: {csv_path}")
    df_1m = pd.read_csv(csv_path)
    df_1m['open_time'] = pd.to_datetime(df_1m['open_time'])
    df_1m.set_index('open_time', inplace=True)

    df_4h = build_4h_features(df_1m)
    df_15m = build_15m_features(df_1m)
    df = pd.concat([df_15m, df_4h], axis=1).dropna()
    sim_df = df.resample('4h').first().dropna()

    engine = FuzzyDCAEngine()
    
    # simulation Settings: FIXED $200,000 Wallet
    total_budget = 200000.0
    
    # ── BLIND DCA ──
    blind_wallet = total_budget
    blind_per_period = total_budget / len(sim_df)
    blind_qty = 0.0
    
    # ── FUZZY DCA ──
    fuzzy_wallet = total_budget
    fuzzy_qty = 0.0
    
    from fuzzy_regime import FuzzyRegimeFilter
    regime_engine = FuzzyRegimeFilter()
    
    records = []
    print(f"[*] Simulating over {len(sim_df)} periods with a $200k Fixed Cap...")
    
    for idx, row in sim_df.iterrows():
        price = row['close']
        
        # 1. Blind Buy (Fixed installment)
        buy_amt_blind = min(blind_per_period, blind_wallet)
        blind_qty += (buy_amt_blind / price)
        blind_wallet -= buy_amt_blind
        
        # 2. Fuzzy DCA (V19: Value + Regime Fusion)
        # Value Component (Z + RSI)
        value_mult = engine.compute_dca_weight(row['rsi_15m'], row['z_4h'])
        
        # Context Component (Regime: ADX, Vol, Efficiency)
        regime_mult = regime_engine.compute_regime_weight(
            row['adx_4h'], row['vol_4h'], row['efficiency_4h']
        )
        
        # Final Combined Weight (Accumulator only)
        weight = value_mult * regime_mult
        
        buy_amt_fuzzy = min(blind_per_period * weight, fuzzy_wallet)
        fuzzy_qty += (buy_amt_fuzzy / price)
        fuzzy_wallet -= buy_amt_fuzzy
        
        records.append({
            'time': idx,
            'price': price,
            'blind_val': blind_qty * price + blind_wallet,
            'fuzzy_val': fuzzy_qty * price + fuzzy_wallet
        })
    # 3. Final Portfolio Metrics
    curr_price = sim_df.iloc[-1]['close']
    final_val_blind = blind_qty * curr_price + blind_wallet
    final_val_fuzzy = fuzzy_qty * curr_price + fuzzy_wallet
    
    res_df = pd.DataFrame(records)

    print("\n" + "="*40)
    print("FAIR FIGHT: FIXED $200k BUDGET")
    print("="*40)
    print(f"Final BTC Price: ${curr_price:,.2f}")
    print("-" * 20)
    print(f"BLIND DCA  | Final Value: ${final_val_blind:,.2f} | BTC Held: {blind_qty:.4f}")
    print(f"FUZZY DCA  | Final Value: ${final_val_fuzzy:,.2f} | BTC Held: {fuzzy_qty:.4f}")
    print(f"ALPHA GAIN: ${final_val_fuzzy - final_val_blind:,.2f}")
    print("="*40)

    # 4. PLOTTING
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="darkgrid")
        
        plt.figure(figsize=(12, 6))
        plt.plot(res_df['time'], res_df['blind_val'], label='Blind DCA Portfolio', color='blue')
        plt.plot(res_df['time'], res_df['fuzzy_val'], label='Fuzzy DCA Portfolio', color='gold', linewidth=2)
        
        plt.title('FAIR FIGHT: Fixed $200k Portfolio Performance')
        plt.ylabel('Total Value (USD)')
        plt.legend()
        plt.savefig('fuzzy_dca_results.png')
    except Exception as e:
        print(f"[!] Plotting failed: {e}")

if __name__ == "__main__":
    run_fuzzy_dca_backtest()
