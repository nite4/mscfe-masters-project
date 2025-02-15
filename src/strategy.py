import numpy as np
import pandas as pd
import portfolio_performance as pp

from utils import log


class Strategy:
    '''
    Trading Strategy Class

    Attributes:
    -----------
    initial_capital : float
        The initial capital allocated for trading.
    m_threshold : int
        The number of consecutive signals required to confirm a trade.
    max_exposure : float
        The maximum percentage of total portfolio value allocated to a single asset.
    max_trade_size : float
        The maximum dollar amount allocated per trade.
    close_threshold : int
        The number of periods after which a position is forcefully closed if no opposite signal appears.
    trade_history : list
        A record of all executed trades.
    '''

    def __init__(self, initial_capital=10000, m_threshold=3, max_exposure=0.25, max_trade_size=1000, close_threshold=144):
        '''
        Initializes the trading strategy with given parameters.

        Parameters:
        ___________
        initial_capital (float, default=10000):
            The starting capital for the strategy.
        m_threshold (int, default=3):
            The required number of consecutive signals to confirm a trade.
        max_exposure (float, default=0.25):
            The maximum exposure allowed per asset as a fraction of total portfolio value.
        max_trade_size (float, default=1000):
            The maximum position size allowed per trade in USD.
        close_threshold (int, default=144):
            The maximum number of periods a trade can stay open before forced closure after 12 pair trading hours.
        '''
        self.capital = initial_capital
        self.positions = {}
        self.m_threshold = m_threshold
        self.max_exposure = max_exposure
        self.max_trade_size = max_trade_size
        self.close_threshold = close_threshold
        self.trade_history = []

    def run_strategy(self, df):
        '''
        Executes the trading strategy on a given dataset.

        Parameters:
        ___________
        df (pandas.DataFrame):
            The input dataframe containing trading signals for various assets.

        Returns:
        ________
        dict:
            A dictionary containing the final portfolio value and trade history.
        '''
        signals = {}
        confirmed_signals = {}
        portfolio_value = self.capital
        open_positions = {}
        position_age = {}

        for index, row in df.iterrows():
            for asset in df.columns:
                if asset == 'OpenTime':
                    continue

                signal = row[asset]
                if asset not in signals:
                    signals[asset] = []

                # Track signal confirmation threshold
                if signal != 0:
                    signals[asset].append(signal)
                    if len(signals[asset]) >= self.m_threshold and all(x == signal for x in signals[asset][-self.m_threshold:]):
                        confirmed_signals[asset] = signal
                else:
                    signals[asset] = []
                    confirmed_signals.pop(asset, None)

                # Trading logic
                if asset in confirmed_signals:
                    trade_signal = confirmed_signals[asset]

                    if trade_signal == 1 and asset not in open_positions:
                        trade_size = min(self.max_trade_size, self.max_exposure * portfolio_value)
                        open_positions[asset] = trade_size
                        position_age[asset] = 0
                        portfolio_value -= trade_size
                        self.trade_history.append((index, asset, 'BUY', trade_size))

                    elif trade_signal == -1 and asset in open_positions:
                        portfolio_value += open_positions[asset]
                        del open_positions[asset]
                        del position_age[asset]
                        self.trade_history.append((index, asset, 'SELL', trade_size))

                # Forced closure after X periods if no opposite signal appears
                if asset in open_positions:
                    position_age[asset] += 1
                    if position_age[asset] >= self.close_threshold:
                        portfolio_value += open_positions[asset]
                        del open_positions[asset]
                        del position_age[asset]
                        self.trade_history.append((index, asset, 'FORCED SELL', trade_size))

        return {
            'Final Portfolio Value': portfolio_value,
            'Trades Executed': self.trade_history
        }