import numpy as np
import pandas as pd

from utils import log


def annualized_return(start_value, end_value, t):
    """Calculates annualized portfolio return."""
    try:
        return (end_value / start_value) ** (1 / t) - 1
    except Exception as e:
        print(f"Failed to calculate annualized portfolio return: {str(e)}")
        log.error(f"Failed to calculate annualized portfolio return: {str(e)}")
        return None


def sharpe_ratio(portfolio_returns, risk_free_rate):
    """Calculated Sharpe Ratio of the portfolio."""
    try:
        excess_returns = portfolio_returns - risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns)
    except Exception as e:
        print(f"Failed to calculate Sharpe Ratio: {str(e)}")
        log.error(f"Failed to calculate Sharpe Ratio: {str(e)}")
        return None


def sortino_ratio(portfolio_returns, risk_free_rate):
    """Calculated Sortino Ratio of the portfolio."""
    try:
        excess_returns = portfolio_returns - risk_free_rate
        downside_risk = np.std([r for r in excess_returns if r < 0])
        return np.mean(excess_returns) / downside_risk
    except Exception as e:
        print(f"Failed to calculate Sortino Ratio: {str(e)}")
        log.error(f"Failed to calculate Sortino Ratio: {str(e)}")
        return None


def hit_ratio(returns):
    """Calculates Hit Ratio of the portfolio."""
    try:
        return np.sum(returns > 0) / len(returns)
    except Exception as e:
        print(f"Failed to calculate Hit Ratio: {str(e)}")
        log.error(f"Failed to calculate Hit Ratio: {str(e)}")
        return None


def max_drawdown(values):
    """Calculates Maximum Drawdown of the portfolio."""
    try:
        drawdowns = np.maximum.accumulate(values) - values
        return np.max(drawdowns) / np.max(values)
    except Exception as e:
        print(f"Failed to calculate Maximum Drawdown: {str(e)}")
        log.error(f"Failed to calculate Maximum Drawdown: {str(e)}")
        return None


def portfolio_volatility(returns):
    """Calculates portfolio volatility given as standard deviation of returns."""
    try:
        return np.std(returns)
    except Exception as e:
        print(f"Failed to calculate portfolio volatility: {str(e)}")
        log.error(f"Failed to calculate portfolio volatility: {str(e)}")
        return None
