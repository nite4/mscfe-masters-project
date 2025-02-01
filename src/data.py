import numpy as np
import os
import pandas as pd

from binance import Client
from constants import *
from IPython.display import display
from utils import log


def read_excel_sheets(file_path:str='data.xlsx') -> pd.DataFrame:
    '''
    Reads Excel data from the given file path.

    Parameters:
    -----------
    file_path (str):
        string containing the file path to process.
    
    Returns:
    --------
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


def get_crypto_data(client:Client=None):
    '''
    Retrieves OHLC 5-minute data of the 10 most traded coins against USDT:
    BTCUSDT, ETHUSDT, XRPUSDT, BNBUSDT, SOLUSDT,
    DOGEUSDT, USDCUSDT, ADAUSDT, TRXUSDT and AVAXUSDT.

    Parameters:
    -----------
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
            klines = client.get_historical_klines(cs, Client.KLINE_INTERVAL_5MINUTE, '5 Oct, 2024', '8 Jan, 2025')
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
def process_pairs_series(seriesX, seriesY, dfX, dfY):
    '''
    Creates Pandas DataFrame with the desired series aligned.

    Parameters:
    ___________
    seriesX, seriesY (str):
        strings with names of symbols to process.
    dfX, dfY (pd.DataFrame):
        DataFrames with raw symbol data.
    
    Returns:
    ________
    merged (pd.DataFrame):
        DataFrame with data aligned in time.
    '''
    try:
        priceX = dfX[dfX['Symbol']==seriesX]['Close'].rename(seriesX)
        t00 = priceX.index[0]
        
        priceY = dfY[dfY['Symbol']==seriesY]['Close'].rename(seriesY)
        t10 = priceY.index[0]

        merged = pd.concat([priceX, priceY], axis=1)
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