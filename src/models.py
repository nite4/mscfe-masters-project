import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import log
from xgboost import XGBRegressor


def ridge_regression(df:pd.DataFrame, alpha:float=1.0, s:float=2.0, solver:str='auto'):
    '''
    Runs a Ridge Regression model on the input data.

    Parameters:
    ___________
    df (pd.DataFrame):
        DataFrame with features and target variable
    alpha (float, default=1.0):
        Ridge Regression regularization parameter, must be ≥0
    s (float, default=2.0):
        threshold for trading signal generation
    solver (str, default='auto'):
        Ridge Regression solver; available options as in
        scikit-learn Ridge (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
    
    Returns:
    ________
    ridge (sklearn.linear_model.Ridge):
        fitted Ridge Regression model
    mse (float):
        Mean Squared Error of model predictions on test set
    df_test (pd.DataFrame): 
        test set DataFrame containing original features,
        predicted spread values, and generated trading signals
    '''
    try:
        if alpha < 0:
            raise ValueError('Alpha must be greater than or equal to 0.')
        
        # Create a copy of the dataframe to avoid SettingWithCopyWarning
        df = df.copy()

        X = df.drop(['NormalizedSpread'], axis=1)
        y = df['NormalizedSpread']

        # Train-Validation split 80:20
        test_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_size,
                                                            shuffle=False)
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        ridge = Ridge(alpha=alpha, solver=solver)
        ridge.fit(X_train_scaled, y_train)
        y_pred = ridge.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        print(f'Ridge Regression MSE: {mse}')
        log.info(f'Ridge Regression with alpha={alpha} MSE: {mse}')

        # Interpret forecast as trading signals
        signals = np.where(y_pred>s, 'Sell A, Buy B',
                          np.where(y_pred<-1*s, 'Buy A, Sell B',
                                  'No Trade'))

        df_test = X_test.copy()
        df_test['NormalizedSpread'] = y_test
        df_test['PredictedSpread'] = y_pred
        df_test['PredictedSignal'] = signals

        return ridge, mse, df_test
    except Exception as e:
        print(f'Failed to run Ridge Regression (alpha={alpha}, s={s}): {str(e)}')
        log.error(f'Failed to run Ridge Regression (alpha={alpha}, s={s}): {str(e)}')
        return None, np.inf, pd.DataFrame()


def xgboost_regression(df:pd.DataFrame, learning_rate:float=0.1,
                       n_estimators:int=100, max_depth:int=3, s:float=2.0):
    '''
    Runs an XGBoost model on the input data.

    Parameters:
    ___________
    df (pd.DataFrame):
        DataFrame with features and target variable
    learning_rate (float, default=0.1):
        Step size shrinkage used in updating weights
    n_estimators (int, default=100):
        Number of boosting rounds
    max_depth (int, default=3):
        Maximum depth of a tree
    s (float, default=2.0):
        Threshold for trading signal generation
    
    Returns:
    ________
    xgb_model (XGBRegressor):
        Fitted XGBoost model
    mse (float):
        Mean Squared Error of model predictions on test set
    df_test (pd.DataFrame): 
        Test set DataFrame containing original features,
        predicted spread values, and generated trading signals
    '''
    try:
        if learning_rate <= 0 or n_estimators <= 0 or max_depth <= 0:
            raise ValueError('Learning rate, n_estimators, and max_depth must be greater than 0.')
        
        # Create a copy of the dataframe to avoid SettingWithCopyWarning
        df = df.copy()
        
        X = df.drop(['NormalizedSpread'], axis=1)
        y = df['NormalizedSpread']

        # Train-Validation split 80:20
        test_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_size,
                                                            shuffle=False)
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        xgb_model = XGBRegressor(learning_rate=learning_rate,
                                 n_estimators=n_estimators,
                                 max_depth=max_depth)
        xgb_model.fit(X_train_scaled, y_train)
        y_pred = xgb_model.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        print(f'XGBoost MSE: {mse}')
        log.info(f'XGBoost with learning_rate={learning_rate}, n_estimators={n_estimators}, max_depth={max_depth} MSE: {mse}')

        # Interpret forecast as trading signals
        signals = np.where(y_pred>s, 'Sell A, Buy B',
                          np.where(y_pred<-1*s, 'Buy A, Sell B',
                                  'No Trade'))
        
        df_test = X_test.copy()
        df_test['NormalizedSpread'] = y_test
        df_test['PredictedSpread'] = y_pred
        df_test['PredictedSignal'] = signals
        
        return xgb_model, mse, df_test
    except Exception as e:
        print(f'Failed to run XGBoost (learning_rate={learning_rate}, n_estimators={n_estimators}, max_depth={max_depth}, s={s}): {str(e)}')
        log.error(f'Failed to run XGBoost (learning_rate={learning_rate}, n_estimators={n_estimators}, max_depth={max_depth}, s={s}): {str(e)}')
        return None, np.inf, pd.DataFrame()


def plot_model_forecasts(ridge_test_df:pd.DataFrame, xgb_test_df:pd.DataFrame,
                         lstm_test_df:pd.DataFrame, transformer_test_df:pd.DataFrame,
                         tickerX:str, tickerY:str, s:float=2.0):
    '''
    Plots actual vs predicted spread values for both Ridge and XGBoost models.

    Parameters:
    ___________
    ridge_test_df, xgb_test_df, lstm_test_df, transformer_test_df (pd.DataFrame):
        test set DataFrames from models containing actual and predicted spread values
    tickerX, tickerY (str):
        ticker symbols for pair assets
    s (float, default=2.0):
        threshold for trading strategy signals detection.
    '''
    try:
        plt.figure(figsize=(8,5))
        # Add trading signal threshold lines
        plt.axhline(y=s, color='orange', linestyle='--', alpha=0.5)
        plt.axhline(y=-1*s, color='orange', linestyle='--', alpha=0.5)
        # Plot actual values
        plt.plot(ridge_test_df['NormalizedSpread'], 
                label='Actual Spread',
                color='black',
                alpha=0.7)
        
        # Plot Ridge predictions if available
        if ridge_test_df is not None and not ridge_test_df.empty:
            plt.plot(ridge_test_df['PredictedSpread'],
                    label='Ridge Prediction',
                    linestyle='--',
                    alpha=0.8)
            
        # Plot XGBoost predictions if available
        if xgb_test_df is not None and not xgb_test_df.empty:
            plt.plot(xgb_test_df['PredictedSpread'],
                    label='XGBoost Prediction',
                    linestyle=':',
                    alpha=0.8)
            
        # Plot LSTM predictions if available
        if lstm_test_df is not None and not lstm_test_df.empty:
            plt.plot(lstm_test_df['PredictedSpread'],
                    label='LSTM Prediction',
                    linestyle='-.',
                    alpha=0.8)
            
        # Plot Transformer predictions if available 
        if transformer_test_df is not None and not transformer_test_df.empty:
            plt.plot(transformer_test_df['PredictedSpread'],
                    label='Transformer Prediction',
                    linestyle='-',
                    alpha=0.8)
            
        plt.title(f'Model Predictions for {tickerX}-{tickerY} Spread')
        plt.xlabel('Time')
        plt.ylabel('Normalized Spread')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'forecast_{tickerX}_{tickerY}.png')
        plt.show();
    except Exception as e:
        print(f'Failed to plot model forecasts: {str(e)}')
        log.error(f'Failed to plot model forecasts: {str(e)}')