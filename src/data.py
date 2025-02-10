import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import statsmodels.api as sm

from binance import Client
from constants import *
from IPython.display import display
from statsmodels.tsa.stattools import adfuller, coint
from utils import log


def read_excel_sheets(file_path:str='data.xlsx') -> pd.DataFrame:
    '''
    Reads Excel data from the given file path.

    Parameters:
    ___________
    file_path (str):
        string containing the file path to process.
    
    Returns:
    ________
    df (pd.DataFrame):
        DataFrame containing the data from the Excel file.
    '''
    try:
        all_sheets = pd.read_excel(file_path, sheet_name=None)
        df = pd.DataFrame()
        float_cols = ['Open', 'High', 'Low', 'Close']
        final_cols = ['OpenTime', 'Open', 'High', 'Low', 'Close', 'Symbol']
        for sheet_name, df_temp in all_sheets.items():
            print(sheet_name)
            df_temp = df_temp.rename(columns={'Date':'OpenTime'})
            if 'Symbol' not in df_temp.columns:
                df_temp['Symbol'] = sheet_name
            df_temp = df_temp.sort_values(by='OpenTime')
            # display(df_temp.tail())
            df = pd.concat([df, df_temp[final_cols]])
        # Standardize opening time to UTC
        df['OpenTime'] = (pd.to_datetime(df['OpenTime'])
                          .map(lambda x: x.tz_localize('Asia/Dubai'))
                          .map(lambda x: x.tz_convert('UTC'))
                          .apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S')))
        # Ensure proper data types
        df[float_cols] = df[float_cols].astype(float)
        df = df.sort_values(by=['OpenTime', 'Symbol'])
        log.info(f'File from {file_path} read successfully.')
        return df
    except FileNotFoundError as fnf:
        print(f'File not found at {file_path}: {str(fnf)}')
        log.error(f'File not found at {file_path}: {str(fnf)}')
        return pd.DataFrame()
    except Exception as e:
        print(f'Failed to read Excel data: {str(e)}')
        log.error(f'Failed to read Excel data: {str(e)}')
        return pd.DataFrame()


def get_crypto_data(client:Client=None) -> pd.DataFrame:
    '''
    Retrieves OHLC 5-minute data of the 10 most traded coins against USDT:
    BTCUSDT, ETHUSDT, XRPUSDT, BNBUSDT, SOLUSDT,
    DOGEUSDT, USDCUSDT, ADAUSDT, TRXUSDT and AVAXUSDT.

    Parameters:
    ___________
    client (binance.Client):
        Binance API client.
    
    Returns:
    ________
    klines_df (pd.DataFrame):
        A DataFrame containing the data from API.
    '''
    try:
        if client is None:
            client = Client(api_key, api_secret)
        # Highest volume coins against Tether USD
        crypto_symbols = ['BTCUSDT',
                          'ETHUSDT',
                          'XRPUSDT',
                          'BNBUSDT',
                          'SOLUSDT',
                          'DOGEUSDT',
                          'USDCUSDT',
                          'ADAUSDT',
                          'TRXUSDT',
                          'AVAXUSDT',
                         ]
        klines_df = pd.DataFrame()
        final_cols = ['OpenTime', 'Open', 'High', 'Low', 'Close', 'Symbol']
        for cs in crypto_symbols:
            print(cs)
            klines = client.get_historical_klines(cs, Client.KLINE_INTERVAL_5MINUTE,
                                                  '5 Oct, 2024', '8 Jan, 2025')
            temp_df = pd.DataFrame(klines, dtype='float').iloc[:, :5]
            temp_df.columns = ['OpenTime', 'Open', 'High', 'Low', 'Close']
            temp_df['Symbol'] = cs
            temp_df = temp_df[final_cols]
            # display(temp_df.tail())
            klines_df = pd.concat([klines_df, temp_df])
        # Ensure correct data types
        klines_df['OpenTime'] = pd.to_datetime(klines_df['OpenTime'], unit='ms')
        klines_df = klines_df.sort_values(by=['OpenTime', 'Symbol'])
        log.info(f'Data from Binance API retrieved successfully.')
        return klines_df
    except Exception as e:
        print(f'Failed to get cryptocurrency data: {str(e)}')
        log.error(f'Failed to get cryptocurrency data: {str(e)}')
        return pd.DataFrame()


# Helper functions to create a Pandas dataframe with 2 different series
def process_pairs_series(seriesX, seriesY,
                         dfX=None, dfY=None, df=None, data_type='Close') -> pd.DataFrame:
    '''
    Creates Pandas DataFrame with the desired series aligned.

    Parameters:
    ___________
    seriesX, seriesY (str):
        strings with names of symbols to process
    dfX, dfY (pd.DataFrame):
        DataFrames with raw symbol data
    df (pd.DataFrame):
        DataFrame with data of all symbols.
    
    Returns:
    ________
    merged (pd.DataFrame):
        DataFrame with data aligned in time.
    '''
    try:
        # Using separate equity and crypto dfs
        if (dfX is not None) & (dfY is not None):
            priceX = (pd.DataFrame(dfX[dfX['Symbol']==seriesX][[data_type, 'OpenTime']])
                      .set_index('OpenTime')
                    #   .rename(index=[seriesX])
                      )
            t00 = priceX.index[0]
            priceY = (pd.DataFrame(dfY[dfY['Symbol']==seriesY][[data_type, 'OpenTime']])
                      .set_index('OpenTime')
                    #   .rename(index=[seriesY])
                      )
            t10 = priceY.index[0]
        # Using df of all data
        elif df is not None:
            priceX = (pd.DataFrame(df[df['Symbol']==seriesX][[data_type, 'OpenTime']])
                      .set_index('OpenTime')
                    #   .rename(index=seriesX)
                      )
            t00 = priceX.index[0]
            priceY = (pd.DataFrame(df[df['Symbol']==seriesY][[data_type, 'OpenTime']])
                      .set_index('OpenTime')
                    #   .rename(index=seriesY)
                    )
            t10 = priceY.index[0]

        # merged = pd.concat([priceX, priceY], axis=1)
        merged = (pd.merge(left=priceX, right=priceY,
                          left_index=True, right_index=True))
        merged.columns = [seriesX, seriesY]
        if t00 > t10:
            t0 = t00
        else:
            t0 = t10
        log.info(f'{seriesX} and {seriesY} processed pairwise successfully.')
        return merged.loc[t0:].dropna()
    except Exception as e:
        print(f'Failed to process series {seriesX} and {seriesY}: {str(e)}')
        log.error(f'Failed to process series {seriesX} and {seriesY}: {str(e)}')
        return pd.DataFrame()


def run_cointegration_test(price_pairs, print_stats=False, plotting=False, std=2):
    '''
    Conducts stationarity test on a given pair of assets.

    Parameters:
    ___________
    price_pairs (pd.DataFrame):
        DataFrame with the two assets to analyze
    print_stats (bool, default=False):
        flag for printing test results to the console
    plotting (bool, default=False):

    '''
    try:
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
            print(f'Pair: {tickerX} & {tickerY}')
            print(f'Correlation: {correlation:.3f}')

            adf_stat, adf_pv, _, num_observations, *_ = adf_result
            print(f'\nSpread ADF Statistic: {adf_stat:.4f}')
            if adf_pv < 0.05:
                print(f'p-value: {adf_pv:.3f} (Spread is stationary)')
            else:
                print(f'p-value: {adf_pv:.3f} (Spread is non-stationary)')

            coint_stat, coint_pv, crit_values = coint_result
            print(f'\nCointegration Test Statistic: {coint_stat:.4f}')
            if coint_pv < 0.05:
                print(f'p-value: {coint_pv:.3f} (Both series are cointegrated)')
            else:
                print(f'p-value: {coint_pv:.3f} (Both series are not cointegrated)')

        if plotting:
            # spreads computes
            spread_mean = spread.mean()
            spread_std = spread.std()
            z_score = (spread - spread_mean) / spread_std

            # Create a 1x3 subplot layout
            fig, axes = plt.subplots(1, 3, figsize=(21, 6))  # 1 row, 3 columns
            
            # Plot 1: Spread and Trading Thresholds
            axes[0].plot(spread, label='Spread')
            axes[0].axhline(spread_mean, color='red', linestyle='--', label='Mean')
            axes[0].axhline(spread_mean + std * spread_std, color='green',
                            linestyle='--', label='Upper Threshold')
            axes[0].axhline(spread_mean - std * spread_std, color='green',
                            linestyle='--', label='Lower Threshold')
            axes[0].set_title(f'Spread and Trading Thresholds of {tickerX} and {tickerY}')
            axes[0].set_xlabel('Time')
            axes[0].set_ylabel('Spread')
            axes[0].legend()
            # axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            axes[0].xaxis.set_major_locator(plt.IndexLocator(base=200, offset=0))
            plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Plot 2: Z-Score of Spread
            axes[1].plot(z_score, label='Z-Score')
            axes[1].axhline(std, color='green', linestyle='--', label='Upper Threshold')
            axes[1].axhline(-std, color='green', linestyle='--', label='Lower Threshold')
            axes[1].axhline(0, color='red', linestyle='--', label='Mean')
            axes[1].set_title(f'Z-Score of Spread of {tickerX} and {tickerY}')
            axes[1].set_xlabel('Time')
            axes[1].set_ylabel('Spread Z-Score')
            axes[1].legend()
            # axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            axes[1].xaxis.set_major_locator(plt.IndexLocator(base=200, offset=0))
            plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Plot 3: Normalized Prices
            price_normalized = price_pairs / price_pairs.iloc[0]
            axes[2].plot(price_normalized[tickerX], label=f'{tickerX} (Normalized)', linestyle='-')
            axes[2].plot(price_normalized[tickerY], label=f'{tickerY} (Normalized)', linestyle='--')
            axes[2].set_title(f'Price Correlation of {tickerX} and {tickerY} (Normalized)')
            axes[2].set_xlabel('Time')
            axes[2].set_ylabel('Normalized Price')
            axes[2].legend()
            # axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            axes[2].xaxis.set_major_locator(plt.IndexLocator(base=200, offset=0))
            plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(f'output/cointegration_test_{tickerX}_{tickerY}.png')
            plt.show();

        return spread, correlation, adf_result, coint_result
    except Exception as e:
        print(f'Failed to conduct stationarity test on {tickerX} and {tickerY}: {str(e)}')
        log.error(f'Failed to conduct stationarity test on {tickerX} and {tickerY}: {str(e)}')
        return None, None, None, None


def plot_spread(df, X, Y, s:float=2.0, ax=None):
    '''
    Plots the normalized spread between two assets.

    Parameters:
    ___________
    spread_df (pd.DataFrame):
        DataFrame containing the spread data
    tickerX, tickerY (str):
        ticker symbols for pair assets
    ax (matplotlib.axes.Axes, default=None):
        axes object to plot on; if None, a new figure is created.
    '''
    try:
        if ax is None:  # Create a new figure if ax is not provided
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            plt.sca(ax)
        ax.axhline(2, color='orange', linestyle='--', linewidth=0.75)
        ax.axhline(-2, color='orange', linestyle='--', linewidth=0.75)
        ax.plot(df['NormalizedSpread'],
                 label='Normalized Spread',
                 color='dodgerblue',
                 linewidth=0.75,
                 alpha=0.9,
                 zorder=1)
        ax.scatter(df[df['NormalizedSpread']>s]['NormalizedSpread'].index,
                    df[df['NormalizedSpread']>s]['NormalizedSpread'],
                    label=f'Sell {X},\nbuy {Y}',
                    color='crimson',
                    alpha=0.6,
                    marker='.',
                    s=11,
                    zorder=2)
        ax.scatter(df[df['NormalizedSpread']<-1*s]['NormalizedSpread'].index,
                    df[df['NormalizedSpread']<-1*s]['NormalizedSpread'],
                    label=f'Buy {X},\nsell {Y}',
                    color='limegreen',
                    alpha=0.6,
                    marker='.',
                    s=11,
                    zorder=2)
        if (len(X)>6) & (len(Y)>6):
            ax.xaxis.set_major_locator(plt.IndexLocator(base=1000, offset=0))
        else:
            ax.xaxis.set_major_locator(plt.IndexLocator(base=200, offset=0))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', size=6)
        plt.title(f'Normalized Spread of {X} and {Y} - Training Set')
        plt.xlabel('Time')
        plt.ylabel('Normalized Spread')
        plt.legend(fontsize=8,
                   bbox_to_anchor=(1.3, 1),
                   loc='upper right'
                  )
        plt.tight_layout()
        # plt.savefig(f'output/normalized_spread_{X}_{Y}.png')
        return ax
    except Exception as e:
        print(f'Failed to plot normalized spread: {str(e)}')
        log.error(f'Failed to plot normalized spread: {str(e)}')
        return None