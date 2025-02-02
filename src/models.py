import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import log


def ridge_regression(df:pd.DataFrame, alpha:float=1.0, s:float=2.0):
    '''
    Runs a Ridge Regression model on the input data.

    Parameters:
    ___________
    df (pd.DataFrame):
        DataFrame with features and target variable
    alpha (float, default=1.0):
        Ridge Regression regularization parameter, must be ≥0
    s (float, default=2.0):
        threshold for trading signal generation.
    
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
        
        # Create 5 lags
        for lag in range(1, 6):
            df[f'NormalizedSpread_Lag{lag}'] = df.loc[:, 'NormalizedSpread'].shift(lag)
        
        # Drop NA values after creating lags
        df = df.dropna()

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

        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_scaled, y_train)
        y_pred = ridge.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        print(f'Ridge Regression MSE: {mse}')
        log.info(f'Ridge Regression with alpha={alpha} MSE: {mse}')

        # Interpret forecast as trading signals
        signals = np.where(y_pred > s, 'Sell A, Buy B',
                          np.where(y_pred < -1*s, 'Buy A, Sell B',
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