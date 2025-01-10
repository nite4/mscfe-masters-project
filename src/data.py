import numpy as np
import os
import pandas as pd

from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
from IPython.display import display


def read_excel_sheets(file_path='data.xlsx'):
    '''
    Reads Excel data from the given file path.
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
        return df
    except Exception as e:
        print(f'Failed to read Excel data: {str(e)}')
        return None


def get_crypto_data(client):
    '''Retrieves OHLC 5-minute data of the 10 most traded coins against USDT.'''
    try:
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
        return klines_df
    except Exception as e:
        print(f'Failed to get cryptocurrency data: {str(e)}')
        return None