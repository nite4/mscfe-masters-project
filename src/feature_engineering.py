import pandas as pd
import numpy as np
import talib

def exponential_moving_average(series, ticker, timeperiod):
    """
    Calculate the Exponential Moving Average (EMA) of a time series data.

    Parameters:
    series (pd.Series): The time series data (prices or other values).
    ticker (str): The ticker symbol for the asset.
    timeperiod (int): The number of periods over which to calculate the EMA.

    Returns:
    pd.Series: The calculated EMA values for the given data, with a name based on the ticker and time period.
    
    The Exponential Moving Average (EMA) is a weighted moving average where more 
    weight is given to recent data points, making it more responsive to recent price changes.
    """

    indicator = talib.EMA(series, timeperiod=timeperiod)
    indicator.name = f"{ticker}_EMA_{timeperiod}"
    
    return indicator

def moving_average_convergence_divergence(series, ticker, fast, slow, signal):
    """
    Calculate the Moving Average Convergence Divergence (MACD) and its signal line.

    Parameters:
    series (pd.Series): The time series data (prices or other values).
    ticker (str): The ticker symbol for the asset.
    fast (int): The number of periods for the fast (short-term) moving average.
    slow (int): The number of periods for the slow (long-term) moving average.
    signal (int): The number of periods for the signal line.

    Returns:
    tuple: A tuple containing:
        - MACD (pd.Series): The difference between the fast and slow EMAs.
        - Signal line (pd.Series): The EMA of the MACD line.
        - Histogram (pd.Series): The difference between the MACD line and the signal line.
    
    The MACD is used to identify changes in the strength, direction, momentum, and duration 
    of a trend in a stock's price. The signal line is used to identify buy and sell signals.
    """

    macd, macdsignal, macdhist = talib.MACD(
        series,
        fastperiod=fast,
        slowperiod=slow,
        signalperiod=signal,
    )
    
    macd.name = f"{ticker}_MACD"
    macdsignal.name = f"{ticker}_MACDsignal"
    macdhist.name = f"{ticker}_MACDhist"
    
    return macd, macdsignal, macdhist

def relative_strength_index(series, ticker, timeperiod):
    """
    Calculate the Relative Strength Index (RSI) of a time series data.

    Parameters:
    series (pd.Series): The time series data (prices or other values).
    ticker (str): The ticker symbol for the asset.
    timeperiod (int): The number of periods over which to calculate the RSI.

    Returns:
    pd.Series: The calculated RSI values for the given data, with a name based on the ticker and time period.
    
    The Relative Strength Index (RSI) is a momentum oscillator that measures the speed 
    and change of price movements. It ranges from 0 to 100, with levels above 70 indicating 
    overbought conditions and levels below 30 indicating oversold conditions.
    """

    indicator = talib.RSI(series, timeperiod=timeperiod)
    indicator.name = f"{ticker}_RSI_{timeperiod}"
    
    return indicator


def bollinger_bands(series, ticker, timeperiod=20, nbdevup=2, nbdevdn=2):
    """
    Computes Bollinger Bands, which consist of an upper, middle, and lower band.
    - The middle band is a simple moving average (SMA) of the closing prices.
    - The upper and lower bands are standard deviations away from the SMA.

    Parameters:
        series (pd.Series): Time series of stock prices.
        ticker (str): Stock ticker symbol.
        timeperiod (int): Lookback period for the SMA.
        nbdevup (int): Number of standard deviations for the upper band.
        nbdevdn (int): Number of standard deviations for the lower band.

    Returns:
        tuple: (upper_band, middle_band, lower_band)
    """
    upper, middle, lower = talib.BBANDS(series, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn)
    upper.name = f"{ticker}_BB_upper"
    middle.name = f"{ticker}_BB_middle"
    lower.name = f"{ticker}_BB_lower"
    
    return upper, middle, lower

def average_true_range(high, low, close, ticker, timeperiod=14):
    """
    Computes the Average True Range (ATR), a volatility indicator.
    ATR measures market volatility by averaging the greatest of three values:
    - The current high minus the current low.
    - The absolute value of the current high minus the previous close.
    - The absolute value of the current low minus the previous close.

    Parameters:
        high (pd.Series): High prices.
        low (pd.Series): Low prices.
        close (pd.Series): Closing prices.
        ticker (str): Stock ticker symbol.
        timeperiod (int): Lookback period for ATR calculation.

    Returns:
        pd.Series: ATR values.
    """
    atr = talib.ATR(high, low, close, timeperiod=timeperiod)
    atr.name = f"{ticker}_ATR_{timeperiod}"
    return atr

def stochastic_oscillator(high, low, close, ticker, fastk_period=14, slowk_period=3, slowd_period=3):
    """
    Computes the Stochastic Oscillator, a momentum indicator comparing a stock's closing price to its price range.
    - %K represents the raw stochastic value.
    - %D is a moving average of %K (signal line).

    Parameters:
        high (pd.Series): High prices.
        low (pd.Series): Low prices.
        close (pd.Series): Closing prices.
        ticker (str): Stock ticker symbol.
        fastk_period (int): Period for %K calculation.
        slowk_period (int): Smoothing period for %K.
        slowd_period (int): Smoothing period for %D.

    Returns:
        tuple: (slowk, slowd)
    """
    slowk, slowd = talib.STOCH(high, low, close, fastk_period=fastk_period, slowk_period=slowk_period, slowd_period=slowd_period)
    slowk.name = f"{ticker}_StochK_{fastk_period}"
    slowd.name = f"{ticker}_StochD_{slowd_period}"
    
    return slowk, slowd

def commodity_channel_index(high, low, close, ticker, timeperiod=20):
    """
    Computes the Commodity Channel Index (CCI), which measures price deviation from the mean.
    - High values indicate overbought conditions.
    - Low values indicate oversold conditions.

    Parameters:
        high (pd.Series): High prices.
        low (pd.Series): Low prices.
        close (pd.Series): Closing prices.
        ticker (str): Stock ticker symbol.
        timeperiod (int): Lookback period for CCI calculation.

    Returns:
        pd.Series: CCI values.
    """
    cci = talib.CCI(high, low, close, timeperiod=timeperiod)
    cci.name = f"{ticker}_CCI_{timeperiod}"
    return cci

def williams_percent_r(high, low, close, ticker, timeperiod=14):
    """
    Computes the Williams %R, a momentum indicator that measures overbought and oversold levels.
    - Values close to -100 indicate oversold conditions.
    - Values close to 0 indicate overbought conditions.

    Parameters:
        high (pd.Series): High prices.
        low (pd.Series): Low prices.
        close (pd.Series): Closing prices.
        ticker (str): Stock ticker symbol.
        timeperiod (int): Lookback period for %R calculation.

    Returns:
        pd.Series: Williams %R values.
    """
    willr = talib.WILLR(high, low, close, timeperiod=timeperiod)
    willr.name = f"{ticker}_WILLR_{timeperiod}"
    return willr

