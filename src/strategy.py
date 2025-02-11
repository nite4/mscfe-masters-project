import numpy as np
import pandas as pd
import portfolio_performance as pp

from utils import log


def calculate_position_sizes(initial_investment:float=10000,
                             prices_df:pd.DataFrame=None,
                             max_pct:float=0.25) -> dict:
    '''
    Calculate maximum position sizes for each asset ensuring no more than
    `max_pct` allocation.

    Parameters:
    ___________
    initial_investment (float, default=10,000):
        Initial portfolio value.
    prices_df (pd.DataFrame):
        DataFrame containing asset prices.
    max_pct (float, default=0.25):
        Maximum allowed percentage to allocate into a single asset.

    Returns:
    ________
    position_sizes (dict):
        Maximum position sizes for each asset.
    '''
    try:
        max_position = initial_investment*max_pct  # 25% limit per asset
        position_sizes = {}

        for column in prices_df.columns:
            position_sizes[column] = max_position/prices_df[column].iloc[0]

        return position_sizes
    except Exception as e:
        print(f'Failed to calculate position sizes: {str(e)}')
        log.error(f'Failed to calculate position sizes: {str(e)}')
        return {}


def calculate_portfolio_exposure(current_positions:dict,
                                 prices:pd.Series,
                                 initial_investment:float=10000) -> float:
    '''
    Calculate current portfolio exposure as a percentage of initial investment.

    Parameters:
    ___________
    current_positions (dict):
        Current positions in the portfolio.
    prices (pd.Series):
        Current prices for the assets.
    initial_investment (float, default=10,000):
        Initial portfolio value.

    Returns:
    ________
    float: Current portfolio exposure as percentage.
    '''
    try:
        exposure = sum(abs(pos*prices[asset])
                       for asset, pos in current_positions.items())
        return exposure / initial_investment
    except Exception as e:
        print(f'Failed to calculate portfolio exposure: {str(e)}')
        log.error(f'Failed to calculate portfolio exposure: {str(e)}')
        return 0


def execute_trades(signal:str, prices:pd.Series,
                   position_sizes:dict, current_positions:dict,
                   initial_investment:float=10000) -> tuple:
    '''
    Execute trades based on signals and position constraints.

    Parameters:
    ___________
    signal (str):
        Trading signal indicating buy/sell decisions.
    prices (pd.Series):
        Current prices for the assets.
    position_sizes (dict):
        Maximum allowed position sizes.
    current_positions (dict):
        Current positions in the portfolio.
    initial_investment (float, default=10,000):
        Initial portfolio value.

    Returns:
    ________
    new_positions, cash_flow (tuple):
        Updated positions and cash flows from trades.
    '''
    try:
        if signal == 'No action':
            return current_positions, 0

        action = signal.split(', ')
        tickerX, tickerY = action[0].split(' ')[1], action[1].split(' ')[1]

        current_exposure = calculate_portfolio_exposure(current_positions,
                                                        prices,
                                                        initial_investment)
        existing_position = (current_positions.get(tickerX, 0)!=0
                             or current_positions.get(tickerY, 0)!=0)

        if not existing_position and current_exposure < 0.95:
            new_positions = current_positions.copy()
            trade_size_x, trade_size_y = (position_sizes.get(tickerX, 0),
                                          position_sizes.get(tickerY, 0))
            trade_value_x, trade_value_y = (trade_size_x*prices[tickerX],
                                            trade_size_y*prices[tickerY])

            if ((current_exposure + (trade_value_x + trade_value_y)
                 / initial_investment) <= 1.0):
                if 'Buy' in action[0]:
                    new_positions[tickerX], new_positions[tickerY] = trade_size_x, -trade_size_y
                else:
                    new_positions[tickerX], new_positions[tickerY] = -trade_size_x, trade_size_y

                cash_flow = -(new_positions[tickerX]*prices[tickerX]
                              + new_positions[tickerY]*prices[tickerY])
                return new_positions, cash_flow
        return current_positions, 0
    except Exception as e:
        print(f'Failed to execute trades: {str(e)}')
        log.error(f'Failed to execute trades: {str(e)}')
        return current_positions, 0


def run_strategy(df:pd.DataFrame, prices_df:pd.DataFrame,
                 initial_investment:float=10000, risk_free_rate:float=0.02) -> dict:
    '''
    Simulates trading decisions based on the forecasted normalized spread.

    Parameters:
    ___________
    df (pd.DataFrame):
        DataFrame with model predictions.
    initial_investment (float):
        Initial portfolio value.
    risk_free_rate (float):
        Annual risk-free rate for performance calculations.

    Returns:
    ________
    performance_metrics (dict):
        Portfolio performance metrics.
    '''
    try:
        exclude_columns = ['NormalizedSpread',
                           'PredictedSpread',
                           'PredictedSignal',
                           'Pair',
                           'Model']
        portfolio_value = [initial_investment]
        current_positions = {col: 0 for col in df.columns
                             if col not in exclude_columns}
        cash = initial_investment
        daily_returns = []

        position_sizes = calculate_position_sizes(initial_investment, prices_df)
        for i in range(1, len(df)):
            signal = df['PredictedSignal'].iloc[i-1]
            new_positions, cash_flow = execute_trades(signal,
                                                      df.iloc[i].drop(labels=exclude_columns,
                                                                      errors='ignore'),
                                                      position_sizes,
                                                      current_positions)
            cash += cash_flow
            current_positions = new_positions
            current_value = cash + sum(pos*df[asset].iloc[i]
                                       for asset, pos
                                       in current_positions.items())

            portfolio_value.append(current_value)
            daily_returns.append((current_value - portfolio_value[-2])
                                 / portfolio_value[-2])

        daily_returns = np.array(daily_returns)
        performance_metrics = {
            'Final Portfolio Value': portfolio_value[-1],
            'Total Return': ((portfolio_value[-1] - initial_investment)
                             / initial_investment),
            'Annualized Return': pp.annualized_return(initial_investment,
                                                      portfolio_value[-1],
                                                      len(df)/252),
            'Sharpe Ratio': pp.sharpe_ratio(daily_returns,
                                            risk_free_rate/252),
            'Sortino Ratio': pp.sortino_ratio(daily_returns,
                                              risk_free_rate/252),
            'Hit Ratio': pp.hit_ratio(daily_returns),
            'Max Drawdown': pp.max_drawdown(portfolio_value),
            'Volatility': pp.portfolio_volatility(daily_returns),
            'Portfolio Values': portfolio_value,
            'Daily Returns': daily_returns.tolist()
        }
        log.info('Portfolio performance metrics calculated correctly.')
        return performance_metrics
    except Exception as e:
        print(f'Failed to simulate trading strategy: {str(e)}')
        log.error(f'Failed to simulate trading strategy: {str(e)}')
        return {}