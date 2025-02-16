import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import psutil
import pypickle
import tensorflow as tf
import time

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import (Dense, Dropout, Flatten,
                                     Input, LayerNormalization, LSTM,
                                     MultiHeadAttention, SimpleRNN)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from utils import log
from xgboost import XGBRegressor


def get_memory_usage():
    '''
    Helper function measuring memory usage.
    '''
    process = psutil.Process()
    return process.memory_info().rss / 1024**2


def ridge_regression(df:pd.DataFrame, p:str, alpha:float=1.0,
                     s:float=2.0, solver:str='auto',
                     pickle_file:str=None):
    '''
    Runs a Ridge Regression model on the input data.

    Parameters:
    ___________
    df (pd.DataFrame):
        DataFrame with features and target variable
    p (str):
        pair to trade, separated with space
    alpha (float, default=1.0):
        Ridge Regression regularization parameter, must be ≥0
    s (float, default=2.0):
        threshold for trading signal generation
    solver (str, default='auto'):
        Ridge Regression solver; available options as in
        scikit-learn Ridge (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
    pickle_file (str, default=None):
        string with path to the pickle file if no need to train the model from scratch.

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

        tickerX, tickerY = p.split(' ')

        # Train-Validation split 80:20
        test_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_size,
                                                            shuffle=False)
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if (pickle_file is not None) & (os.path.exists(pickle_file)):
            # Load the model instance using pypickle
            ridge = pypickle.load(pickle_file)
            print(f'Loaded Ridge Regression model from {pickle_file}.')
            log.info(f'Loaded Ridge Regression model from {pickle_file}.')
            time_usage = 0
            memory_usage = 0
        else:
            # Measure time and memory usage during training
            start_time = time.time()
            start_memory = get_memory_usage()

            ridge = Ridge(alpha=alpha, solver=solver)
            ridge.fit(X_train_scaled, y_train)

            end_time = time.time()
            end_memory = get_memory_usage()
            time_usage = end_time - start_time
            memory_usage = end_memory - start_memory

        y_pred = ridge.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        print(f'Ridge Regression MSE: {mse}')
        log.info(f'Ridge Regression with alpha={alpha} MSE: {mse}')

        # Interpret forecast as trading signals
        signals = np.where(y_pred>s, f'Sell {tickerX}, buy {tickerY}',
                          np.where(y_pred<-1*s, f'Buy {tickerX}, sell {tickerY}',
                                  'No action'))

        df_test = X_test.copy()
        df_test['NormalizedSpread'] = y_test
        df_test['PredictedSpread'] = y_pred
        df_test['PredictedSignal'] = signals
        df_test['Pair'] = p
        df_test['Model'] = 'Ridge Regression'

        return ridge, mse, df_test, time_usage, memory_usage
    except Exception as e:
        print(f'Failed to run Ridge Regression (alpha={alpha}, s={s}): {str(e)}')
        log.error(f'Failed to run Ridge Regression (alpha={alpha}, s={s}): {str(e)}')
        return None, np.inf, pd.DataFrame(), 0, 0


def xgboost_regression(df:pd.DataFrame, p:str, learning_rate:float=0.1,
                       n_estimators:int=100, max_depth:int=3, s:float=2.0,
                       pickle_file:str=None):
    '''
    Runs an XGBoost model on the input data.

    Parameters:
    ___________
    df (pd.DataFrame):
        DataFrame with features and target variable
    p (str):
        pair to trade, separated with space
    learning_rate (float, default=0.1):
        Step size shrinkage used in updating weights
    n_estimators (int, default=100):
        Number of boosting rounds
    max_depth (int, default=3):
        Maximum depth of a tree
    s (float, default=2.0):
        Threshold for trading signal generation
    pickle_file (str, default=None):
        string with path to the pickle file if no need to train the model from scratch.

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
        if (learning_rate <= 0) or (n_estimators <= 0) or (max_depth <= 0):
            raise ValueError('Learning rate, n_estimators, and max_depth must be greater than 0.')

        # Create a copy of the dataframe to avoid SettingWithCopyWarning
        df = df.copy()

        X = df.drop(['NormalizedSpread'], axis=1)
        y = df['NormalizedSpread']

        tickerX, tickerY = p.split(' ')

        # Train-Validation split 80:20
        test_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_size,
                                                            shuffle=False)
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if (pickle_file is not None) and (os.path.exists(pickle_file)):
            xgb_model = pypickle.load(pickle_file)
            print(f'Loaded XGBoost model from {pickle_file}.')
            log.info(f'Loaded XGBoost model from {pickle_file}.')
            time_usage = 0
            memory_usage = 0
        else:
            start_time = time.time()
            start_memory = get_memory_usage()

            xgb_model = XGBRegressor(learning_rate=learning_rate,
                                     n_estimators=n_estimators,
                                     max_depth=max_depth)
            xgb_model.fit(X_train_scaled, y_train)

            end_time = time.time()
            end_memory = get_memory_usage()
            time_usage = end_time - start_time
            memory_usage = end_memory - start_memory
        
        y_pred = xgb_model.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        print(f'XGBoost MSE: {mse}')
        log.info(f'XGBoost with learning_rate={learning_rate}, n_estimators={n_estimators}, max_depth={max_depth} MSE: {mse}')

        # Interpret forecast as trading signals
        signals = np.where(y_pred>s, f'Sell {tickerX}, buy {tickerY}',
                          np.where(y_pred<-1*s, f'Buy {tickerX}, sell {tickerY}',
                                  'No action'))

        df_test = X_test.copy()
        df_test['NormalizedSpread'] = y_test
        df_test['PredictedSpread'] = y_pred
        df_test['PredictedSignal'] = signals
        df_test['Pair'] = p
        df_test['Model'] = 'XGBoost'

        return xgb_model, mse, df_test, time_usage, memory_usage
    except Exception as e:
        print(f'Failed to run XGBoost (learning_rate={learning_rate}, n_estimators={n_estimators}, max_depth={max_depth}, s={s}): {str(e)}')
        log.error(f'Failed to run XGBoost (learning_rate={learning_rate}, n_estimators={n_estimators}, max_depth={max_depth}, s={s}): {str(e)}')
        return None, np.inf, pd.DataFrame(), 0, 0


# @tf.function(reduce_retracing=True)
def lstm_regression(df: pd.DataFrame, p:str, lookback:int=10, s:float=2.0,
                     units:int=50, dropout_rate:float=0.2,
                     learning_rate:float=0.001, epochs:int=50,
                     batch_size:int=32, pickle_file:str=None):
    '''
    Runs an LSTM model on the input data.

    Parameters:
    ___________
    df (pd.DataFrame):
        DataFrame with features and target variable
    p (str):
        pair to trade, separated with space
    lookback (int, default=10):
        number of previous time steps used as input for LSTM
    s (float, default=2.0):
        threshold for trading signal generation
    units (int, default=50):
        number of LSTM units in the layer
    dropout_rate (float, default=0.2):
        dropout rate for regularization
    learning_rate (float, default=0.001):
        learning rate for optimizer
    epochs (int, default=50):
        number of training epochs
    batch_size (int, default=32):
        batch size for training
    pickle_file (str, default=None):
        string with path to the pickle file if no need to train the model from scratch.
    
    Returns:
    ________
    model (tf.keras.Model):
        trained LSTM model
    mse (float):
        Mean Squared Error of model predictions on test set
    df_test (pd.DataFrame): 
        test set DataFrame containing original features,
        predicted spread values, and generated trading signals.
    '''
    try:
        df = df.copy()
        X = df.drop(['NormalizedSpread'], axis=1).values
        y = df['NormalizedSpread'].values

        tickerX, tickerY = p.split(' ')

        # Convert data to sequences
        X_seq, y_seq = [], []
        for i in range(lookback, len(X)):
            X_seq.append(X[i - lookback:i])
            y_seq.append(y[i])

        X_seq, y_seq = np.array(X_seq), np.array(y_seq)

        # Train-Test split
        test_size = int(0.2 * len(X_seq))
        X_train, X_test = X_seq[:-test_size], X_seq[-test_size:]
        y_train, y_test = y_seq[:-test_size], y_seq[-test_size:]

        if (pickle_file is not None) and (os.path.exists(pickle_file)):
            model = pypickle.load(pickle_file)
            print(f'Loaded LSTM model from {pickle_file}.')
            log.info(f'Loaded LSTM model from {pickle_file}.')
            time_usage = 0
            memory_usage = 0
        else:
            start_time = time.time()
            start_memory = get_memory_usage()

            model = Sequential()
            model.add(Input(shape=(lookback, X.shape[1])))
            model.add(LSTM(units, return_sequences=False))
            model.add(Dropout(dropout_rate))
            model.add(Dense(1))
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=False)

            end_time = time.time()
            end_memory = get_memory_usage()
            time_usage = end_time - start_time
            memory_usage = end_memory - start_memory
        
        y_pred = model.predict(X_test).flatten()

        mse = mean_squared_error(y_test, y_pred)
        print(f'LSTM MSE: {mse}')
        log.info(f'LSTM MSE: {mse}')

        # Generate trading signals
        signals = np.where(y_pred>s, f'Sell {tickerX}, buy {tickerY}',
                          np.where(y_pred<-1*s, f'Buy {tickerX}, sell {tickerY}',
                                  'No action'))

        df_test = df.iloc[-test_size:].copy()
        df_test['PredictedSpread'] = y_pred
        df_test['PredictedSignal'] = signals
        df_test['Pair'] = p
        df_test['Model'] = 'LSTM'

        return model, mse, df_test, time_usage, memory_usage
    except Exception as e:
        print(f'Failed to run LSTM model: {str(e)}')
        log.error(f'Failed to run LSTM model: {str(e)}')
        return None, np.inf, pd.DataFrame(), 0, 0


# @tf.function(reduce_retracing=True)
def rnn_regression(df: pd.DataFrame, p:str, lookback:int=10, s:float=2.0,
                    units:int=50, dropout_rate:float=0.2,
                    learning_rate:float=0.001, epochs:int=50,
                    batch_size:int=32, pickle_file:str=None):
    '''
    Runs an RNN model on the input data.

    Parameters:
    ___________
    df (pd.DataFrame):
        DataFrame with features and target variable
    p (str):
        pair to trade, separated with space
    lookback (int, default=10):
        number of previous time steps used as input for RNN
    s (float, default=2.0):
        threshold for trading signal generation
    units (int, default=50):
        number of RNN units in the layer
    dropout_rate (float, default=0.2):
        dropout rate for regularization
    learning_rate (float, default=0.001):
        learning rate for optimizer
    epochs (int, default=50):
        number of training epochs
    batch_size (int, default=32):
        batch size for training
    pickle_file (str, default=None):
        string with path to the pickle file if no need to train the model from scratch.
    
    Returns:
    ________
    model (tf.keras.Model):
        trained RNN model
    mse (float):
        Mean Squared Error of model predictions on test set
    df_test (pd.DataFrame): 
        test set DataFrame containing original features,
        predicted spread values, and generated trading signals.
    '''
    try:
        df = df.copy()
        X = df.drop(['NormalizedSpread'], axis=1).values
        y = df['NormalizedSpread'].values

        tickerX, tickerY = p.split(' ')

        # Convert data to sequences
        X_seq, y_seq = [], []
        for i in range(lookback, len(X)):
            X_seq.append(X[i - lookback:i])
            y_seq.append(y[i])

        X_seq, y_seq = np.array(X_seq), np.array(y_seq)

        # Train-Test split
        test_size = int(0.2 * len(X_seq))
        X_train, X_test = X_seq[:-test_size], X_seq[-test_size:]
        y_train, y_test = y_seq[:-test_size], y_seq[-test_size:]

        if (pickle_file is not None) and (os.path.exists(pickle_file)):
            model = pypickle.load(pickle_file)
            print(f'Loaded RNN model from {pickle_file}.')
            log.info(f'Loaded RNN model from {pickle_file}.')
            time_usage = 0
            memory_usage = 0
        else:
            start_time = time.time()
            start_memory = get_memory_usage()

            model = Sequential()
            model.add(Input(shape=(lookback, X.shape[1])))
            model.add(SimpleRNN(units, return_sequences=False))
            model.add(Dropout(dropout_rate))
            model.add(Dense(1))
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=False)

            end_time = time.time()
            end_memory = get_memory_usage()
            time_usage = end_time - start_time
            memory_usage = end_memory - start_memory
        
        y_pred = model.predict(X_test).flatten()

        mse = mean_squared_error(y_test, y_pred)
        print(f'RNN MSE: {mse}')
        log.info(f'RNN MSE: {mse}')

        # Generate trading signals
        signals = np.where(y_pred>s,
                           f'Sell {tickerX}, buy {tickerY}',
                           np.where(y_pred<-1*s,
                                    f'Buy {tickerX}, sell {tickerY}',
                                    'No action'))

        df_test = df.iloc[-test_size:].copy()
        df_test['PredictedSpread'] = y_pred
        df_test['PredictedSignal'] = signals
        df_test['Pair'] = p
        df_test['Model'] = 'RNN'

        return model, mse, df_test, time_usage, memory_usage
    except Exception as e:
        print(f'Failed to run RNN model: {str(e)}')
        log.error(f'Failed to run RNN model: {str(e)}')
        return None, np.inf, pd.DataFrame(), 0, 0


# @tf.function(reduce_retracing=True)
def transformer_regression(df: pd.DataFrame, p:str, lookback:int=10,
                           s:float=2.0, num_heads:int=2, ff_dim:int=64,
                           dropout_rate:float=0.1, learning_rate:float=0.001,
                           epochs:int=50, batch_size:int=32, pickle_file:str=None):
    '''
    Runs a Transformer model on the input data.

    Parameters:
    ___________
    df (pd.DataFrame):
        DataFrame with features and target variable
    p (str):
        pair to trade, separated with space
    lookback (int, default=10):
        Number of previous time steps used as input for the Transformer model
    s (float, default=2.0):
        Threshold for trading signal generation
    num_heads (int, default=2):
        Number of attention heads in the Transformer encoder
    ff_dim (int, default=64):
        Dimensionality of the feed-forward network in the Transformer
    dropout_rate (float, default=0.1):
        Dropout rate for regularization
    learning_rate (float, default=0.001):
        Learning rate for optimizer
    epochs (int, default=50):
        Number of training epochs
    batch_size (int, default=32):
        Batch size for training
    pickle_file (str, default=None):
        string with path to the pickle file if no need to train the model from scratch.

    Returns:
    ________
    model (tf.keras.Model):
        Trained Transformer model
    mse (float):
        Mean Squared Error of model predictions on test set
    df_test (pd.DataFrame): 
        Test set DataFrame containing original features,
        predicted spread values, and generated trading signals.
    '''
    try:
        df = df.copy()
        X = df.drop(['NormalizedSpread'], axis=1).values
        y = df['NormalizedSpread'].values

        tickerX, tickerY = p.split(' ')

        # Convert data to sequences
        X_seq, y_seq = [], []
        for i in range(lookback, len(X)):
            X_seq.append(X[i - lookback:i])
            y_seq.append(y[i])

        X_seq, y_seq = np.array(X_seq), np.array(y_seq)

        # Train-Test split
        test_size = int(0.2 * len(X_seq))
        X_train, X_test = X_seq[:-test_size], X_seq[-test_size:]
        y_train, y_test = y_seq[:-test_size], y_seq[-test_size:]

        if (pickle_file is not None) and (os.path.exists(pickle_file)):
            model = pypickle.load(pickle_file)
            print(f'Loaded Transformer model from {pickle_file}')
            log.info(f'Loaded Transformer model from {pickle_file}')
            time_usage = 0
            memory_usage = 0
        else:
            start_time = time.time()
            start_memory = get_memory_usage()

            # Define the Transformer encoder block
            def transformer_encoder(inputs, num_heads, ff_dim, dropout_rate):
                attention = MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(inputs, inputs)
                attention = Dropout(dropout_rate)(attention)
                attention = LayerNormalization(epsilon=1e-6)(inputs + attention)
                ff_output = Dense(ff_dim, activation='relu')(attention)
                ff_output = Dense(inputs.shape[-1])(ff_output)
                ff_output = Dropout(dropout_rate)(ff_output)
                return LayerNormalization(epsilon=1e-6)(attention + ff_output)

            inputs = Input(shape=(lookback, X.shape[1]))
            x = transformer_encoder(inputs, num_heads, ff_dim, dropout_rate)
            x = Flatten()(x)
            x = Dense(32, activation='relu')(x)
            x = Dropout(dropout_rate)(x)
            outputs = Dense(1)(x)
            model = Model(inputs, outputs)
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=False)

            end_time = time.time()
            end_memory = get_memory_usage()
            time_usage = end_time - start_time
            memory_usage = end_memory - start_memory
        
        y_pred = model.predict(X_test).flatten()

        mse = mean_squared_error(y_test, y_pred)
        print(f'Transformer MSE: {mse}')
        log.info(f'Transformer MSE: {mse}')

        # Generate trading signals
        signals = np.where(y_pred>s,
                           f'Sell {tickerX}, buy {tickerY}',
                           np.where(y_pred<-1*s,
                                    f'Buy {tickerX}, sell {tickerY}',
                                    'No action'))

        df_test = df.iloc[-test_size:].copy()
        df_test['PredictedSpread'] = y_pred
        df_test['PredictedSignal'] = signals
        df_test['Pair'] = p
        df_test['Model'] = 'Transformer'

        return model, mse, df_test, time_usage, memory_usage
    except Exception as e:
        print(f'Failed to run Transformer model: {str(e)}')
        log.error(f'Failed to run Transformer model: {str(e)}')
        return None, np.inf, pd.DataFrame(), 0, 0


def plot_model_forecasts(ridge_test_df:pd.DataFrame=None,
                         xgb_test_df:pd.DataFrame=None,
                         lstm_test_df:pd.DataFrame=None,
                         transformer_test_df:pd.DataFrame=None,
                         rnn_test_df:pd.DataFrame=None,
                         tickerX:str=None, tickerY:str=None,
                         s:float=2.0, ax:matplotlib.axes=None):
    '''
    Plots actual vs predicted spread values for both Ridge and XGBoost models.

    Parameters:
    ___________
    ridge_test_df, xgb_test_df, lstm_test_df, transformer_test_df (pd.DataFrame):
        test set DataFrames from models containing actual and predicted spread values
    tickerX, tickerY (str):
        ticker symbols for pair assets
    s (float, default=2.0):
        threshold for trading strategy signals detection
     ax (matplotlib.axes.Axes, default=None):
        axes object to plot on; if None, a new figure is created.
    '''
    try:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            plt.sca(ax)
        # Add trading signal threshold lines
        ax.axhline(y=s, color='orange', linestyle='--', alpha=0.5)
        ax.axhline(y=-1*s, color='orange', linestyle='--', alpha=0.5)
        # Plot actual values
        ax.plot(ridge_test_df['NormalizedSpread'],
                label='Actual Spread',
                color='black',
                linewidth=0.75,
                )

        # Plot Ridge predictions if available
        if ridge_test_df is not None and not ridge_test_df.empty:
            ax.plot(ridge_test_df['PredictedSpread'],
                    label='Ridge Prediction',
                    color='teal',
                    linewidth=0.75,
                    linestyle='--',
                    )

        # Plot XGBoost predictions if available
        if xgb_test_df is not None and not xgb_test_df.empty:
            ax.plot(xgb_test_df['PredictedSpread'],
                    label='XGBoost Prediction',
                    color='darkorchid',
                    linewidth=0.75,
                    linestyle=':',
                    )

        # Plot LSTM predictions if available
        if lstm_test_df is not None and not lstm_test_df.empty:
            ax.plot(lstm_test_df['PredictedSpread'],
                    label='LSTM Prediction',
                    color='forestgreen',
                    linewidth=0.75,
                    linestyle='-.',
                    )

        # Plot Transformer predictions if available 
        if transformer_test_df is not None and not transformer_test_df.empty:
            ax.plot(transformer_test_df['PredictedSpread'],
                    label='Transformer Prediction',
                    color='crimson',
                    linewidth=0.75,
                    linestyle='-',
                    )

        # Plot RNN predictions if available 
        if rnn_test_df is not None and not rnn_test_df.empty:
            ax.plot(rnn_test_df['PredictedSpread'],
                    label='RNN Prediction',
                    color='darkorange',
                    linewidth=0.75,
                    linestyle='-',
                    )

        # Clearer timestamp displaying
        if (len(tickerX)>6) & (len(tickerY)>6):
            ax.xaxis.set_major_locator(plt.IndexLocator(base=200, offset=0))
        else:
            ax.xaxis.set_major_locator(plt.IndexLocator(base=40, offset=0))

        plt.title(f'Model Predictions for {tickerX}-{tickerY} Spread - Validation Set')
        plt.xlabel('Time')
        plt.ylabel('Normalized Spread')
        plt.legend(fontsize=8,
                   bbox_to_anchor=(1.35, 1),
                   loc='upper right'
                   )
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', size=6)
        plt.tight_layout()
        plt.savefig(f'output/forecast_{tickerX}_{tickerY}.png')
        return ax
    except Exception as e:
        print(f'Failed to plot model forecasts: {str(e)}')
        log.error(f'Failed to plot model forecasts: {str(e)}')
        return None