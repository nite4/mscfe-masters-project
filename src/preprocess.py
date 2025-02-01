import warnings
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint

def process_pairs_series(seriesX, seriesY, dfX, dfY, attr='Close'):

    priceX = dfX[dfX['Symbol']==seriesX][attr].rename(seriesX)
    t00 = priceX.index[0]
    
    priceY = dfY[dfY['Symbol']==seriesY][attr].rename(seriesY)
    t10 = priceY.index[0]

    merged = pd.concat([priceX, priceY], axis=1)
    if t00 > t10:
        t0 = t00
    else:
        t0 = t10
    
    return merged.loc[t0:].dropna()


def run_cointegration_test(price_pairs, print_stats=False, plotting=False, std=2):

    priceX = price_pairs.iloc[:, 0]
    priceY = price_pairs.iloc[:, 1]
    
    tickerX = priceX.name
    tickerY = priceY.name

    model = sm.OLS(priceY, sm.add_constant(priceX)).fit()
    beta = model.params.iloc[1] #hedge ratio
    spread = priceY - beta * priceX
    # residuals = model.resid

    # correlation of both series
    correlation = priceX.corr(priceY)

    # ADF test for the Spread
    adf_result = adfuller(spread)
    
    # cointegration of both series
    coint_result = coint(priceX, priceY)
    
    if print_stats:
        print(f"Pairs: {tickerX} & {tickerY}")
        print(f"Correlation: {correlation:.3f}")

        adf_stat, adf_pv, _, num_observations, *_ = adf_result
        print(f"\nSpread ADF Statistic: {adf_stat:.4f}")
        if adf_pv < 0.05:
            print(f"p-value: {adf_pv:.3f} (Spread is stationary)")
        else:
            print(f"p-value: {adf_pv:.3f} (Spread is non-stationary)")

        coint_stat, coint_pv, crit_values = coint_result
        print(f"\nCointegration Test Statistic: {coint_stat:.4f}")
        if coint_pv < 0.05:
            print(f"p-value: {coint_pv:.3f} (Both series are cointegrated)")
        else:
            print(f"p-value: {coint_pv:.3f} (Both series are not cointegrated)")

    if plotting:
        # spreads computes
        spread_mean = spread.mean()
        spread_std = spread.std()
        z_score = (spread - spread_mean) / spread_std

        # Create a 1x3 subplot layout
        fig, axes = plt.subplots(1, 3, figsize=(21, 4))  # 1 row, 3 columns
        
        # Plot 1: Spread and Trading Thresholds
        axes[0].plot(spread, label='Spread')
        axes[0].axhline(spread_mean, color='red', linestyle='--', label='Mean')
        axes[0].axhline(spread_mean + std * spread_std, color='green', linestyle='--', label='Upper Threshold')
        axes[0].axhline(spread_mean - std * spread_std, color='green', linestyle='--', label='Lower Threshold')
        axes[0].set_title('Spread and Trading Thresholds')
        axes[0].legend()
        
        # Plot 2: Z-Score of Spread
        axes[1].plot(z_score, label='Z-Score')
        axes[1].axhline(std, color='green', linestyle='--', label='Upper Threshold')
        axes[1].axhline(-std, color='green', linestyle='--', label='Lower Threshold')
        axes[1].axhline(0, color='red', linestyle='--', label='Mean')
        axes[1].set_title('Z-Score of Spread')
        axes[1].legend()
        
        # Plot 3: Normalized Prices
        price_normalized = price_pairs / price_pairs.iloc[0]
        axes[2].plot(price_normalized[tickerX], label=f'{tickerX} (Normalized)', linestyle='-')
        axes[2].plot(price_normalized[tickerY], label=f'{tickerY} (Normalized)', linestyle='--')
        axes[2].set_title(f'Price Correlation of {tickerX} and {tickerY} (Normalized)')
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Normalized Price')
        axes[2].legend()
        axes[2].grid(True)
        
        # Adjust layout for better display
        plt.tight_layout()
        plt.show()

    return spread, correlation, adf_result, coint_result
