import numpy as np
import pandas as pd

from portfolio_performance import *
from utils import log


def calculate_position_sizes(initial_investment:float=10000,
                             prices_df:pd.DataFrame=None, max_pct:float=0.25) -> dict:
    '''
    Calculate maximum position sizes for each asset ensuring no more than `max_pct` allocation.
    
    Parameters:
    ___________
    initial_investment (float, default=10000):
        Initial portfolio value
    prices_df (pd.DataFrame):
        DataFrame containing asset prices
    max_pct (float, default=0.25)
        maximum allowed percentage to allocate into a single asset.
        
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


def calculate_portfolio_exposure(current_positions:dict, prices:pd.Series, 
                               initial_investment:float) -> float:
    '''
    Calculate current portfolio exposure as percentage of initial investment.
    
    Parameters:
    ___________
    current_positions: dict
        Current positions in the portfolio
    prices: pd.Series
        Current prices for the assets
    initial_investment: float
        Initial portfolio value
        
    Returns:
    ________
    float: Current portfolio exposure as percentage
    '''
    try:
        exposure = 0
        for asset, pos in current_positions.items():
            exposure += abs(pos*prices[asset])/initial_investment
        return exposure
    except Exception as e:
        print(f'Failed to calculate portfolio exposure: {str(e)}')
        log.error(f'Failed to calculate portfolio exposure: {str(e)}')


def execute_trades(signal:str, prices:pd.Series, position_sizes:dict, 
                  current_positions:dict, initial_investment:float=10000) -> tuple:
    '''
    Execute trades based on signals and position constraints.
    
    Parameters:
    ___________
    signal: str
        Trading signal indicating buy/sell decisions
    prices: pd.Series
        Current prices for the assets
    position_sizes: dict
        Maximum allowed position sizes
    current_positions: dict
        Current positions in the portfolio.
        
    Returns:
    ________
    new_positions, cash_flow (tuple):
        Updated positions and cash flows from trades.
    '''
    try:
        cash_flow = 0
        new_positions = current_positions.copy()
        
        if signal != 'No action':
            # Parse signal
            action = signal.split(',')
            tickerX = action[0].split(' ')[1]
            tickerY = action[1].split(' ')[1]
            
            # Calculate current portfolio exposure
            current_exposure = calculate_portfolio_exposure(current_positions, prices, 
                                                        initial_investment)
            
            # Check if we have existing positions in these assets
            existing_position = (current_positions[tickerX] != 0 or 
                            current_positions[tickerY] != 0)
            
            # Only execute new trades if we don't have existing positions and have room in the portfolio
            if not existing_position and current_exposure < 0.95:  # 95% to leave room for small price movements
                # Calculate new position sizes
                tickerX_exposure = (position_sizes[tickerX] 
                                        * prices[tickerX]
                                        / initial_investment)
                tickerY_exposure = (position_sizes[tickerY]
                                         * prices[tickerY]
                                         / initial_investment)
            
            if current_exposure + tickerX_exposure + tickerY_exposure <= 1.0:
                # Execute new positions
                if 'Buy' in action[0]:
                    new_positions[tickerX] = position_sizes[tickerX]
                    new_positions[tickerY] = -position_sizes[tickerY]
                else:
                    new_positions[tickerX] = -position_sizes[tickerX]
                    new_positions[tickerY] = position_sizes[tickerY]
                    
                cash_flow -= (new_positions[tickerX]*prices[tickerX] + 
                            new_positions[tickerY]*prices[tickerY])
        
        return new_positions, cash_flow
    except Exception as e:
        print(f'Failed to execute trades: {str(e)}')
        log.error(f'Failed to execute trades: {str(e)}')
        return pd.DataFrame(), 0


def run_strategy(df:pd.DataFrame, initial_investment:float=10000, 
                risk_free_rate:float=0.02) -> dict:
    '''
    Simulates trading decisions based on the forecasted normalized spread.
    
    Parameters:
    ___________
    df: pd.DataFrame
        DataFrame with model predictions
    initial_investment: float
        Initial portfolio value
    risk_free_rate: float
        Annual risk-free rate for performance calculations.
        
    Returns:
    ________
    performance_metrics (dict):
        Portfolio performance metrics.
    '''
    try:
        exclude_columns = ['NormalizedSpread', 'PredictedSpread', 'PredictedSignal', 
                            'Pair', 'Model']
        # Initialize portfolio tracking
        portfolio_value = [initial_investment]
        current_positions = {col: 0 for col in df.columns if col not in exclude_columns}
        cash = initial_investment
        daily_returns = []
        
        position_sizes = calculate_position_sizes(initial_investment, 
                                               df[[col for col in df.columns 
                                                  if col not in exclude_columns]])
        
        # Simulate trading
        for i in range(1, len(df)):
            # Get signal from previous day
            signal = df['PredictedSignal'].iloc[i-1]
            
            # Execute trades at opening prices
            new_positions, cash_flow = execute_trades(
                signal,
                df[[col for col in df.columns if col not in exclude_columns]].iloc[i],
                position_sizes,
                current_positions
            )
            
            cash += cash_flow
            current_positions = new_positions
            
            current_value = cash
            for asset, pos in current_positions.items():
                current_value += pos*df[asset].iloc[i]
            
            portfolio_value.append(current_value)
            daily_returns.append((current_value - portfolio_value[-2])
                                 / portfolio_value[-2])
        
        # Calculate performance metrics
        daily_returns = np.array(daily_returns)
        performance_metrics = {
            'Final Portfolio Value': portfolio_value[-1],
            'Total Return': (portfolio_value[-1]-initial_investment)
                           / initial_investment,
            'Annualized Return': annualized_return(initial_investment, 
                                                 portfolio_value[-1], 
                                                 len(df)/252),
            'Sharpe Ratio': sharpe_ratio(daily_returns, 
                                       risk_free_rate/252),
            'Sortino Ratio': sortino_ratio(daily_returns, 
                                         risk_free_rate/252),
            'Hit Ratio': hit_ratio(daily_returns),
            'Max Drawdown': max_drawdown(portfolio_value),
            'Volatility': portfolio_volatility(daily_returns),
            'Portfolio Values': portfolio_value,
            'Daily Returns': daily_returns.tolist()
        }
        return performance_metrics
    except Exception as e:
        print(f'Failed to simulate trading strategy: {str(e)}')
        log.error(f'Failed to simulate trading strategy: {str(e)}')
        return None