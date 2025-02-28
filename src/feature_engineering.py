import data
import pandas as pd
import talib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import log


class FeaturesEngineering:
    def __init__(self, default_config=None):
        if default_config is None:
            self.default_config = {
                "ema": [8, 21, 55],
                "macd": {
                    "fast": 12,
                    "slow": 26,
                    "signal": 9,
                },
                "rsi": [14],
                "bb": {
                    "timeperiod": 20,
                    "nbdevup": 2,
                    "nbdevdn": 2,
                },
                "atr": {
                    "timeperiod": 14,
                },
                "stoch": {
                    "fastk_period": 14,
                    "slowk_period": 3,
                    "slowd_period": 3,
                },
                "cci": {
                    "timeperiod": 20,
                },
                "willr": {
                    "timeperiod": 14,
                },
            }
        else:
            self.default_config = default_config

    def exponential_moving_average(
        self, series: pd.Series, ticker: str, timeperiod: int
    ):
        """
        Calculates Exponential Moving Average (EMA) of a time series data.
        The Exponential Moving Average (EMA) is a weighted moving average
        where more weight is given to recent data points, making it more
        responsive to recent price changes.

        Parameters:
        ___________
        series (pd.Series):
            The time series data (prices or other values).
        ticker (str):
            The ticker symbol for the asset.
        timeperiod (int, default=12,26):
            The number of periods over which to calculate the EMA.

        Returns:
        ________
        indicator (pd.Series):
            The calculated EMA values for the given data, with a name based
            on the ticker and time period.
        """
        try:
            indicator = talib.EMA(series, timeperiod=timeperiod)
            indicator.name = f"{ticker}_EMA{timeperiod}"
            return indicator
        except Exception as e:
            print(f"Failed to calculate EMA: {str(e)}")
            log.error(f"Failed to calculate EMA: {str(e)}")
            return pd.Series()

    def moving_average_convergence_divergence(
        self,
        series: pd.Series,
        ticker: str,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ):
        """
        Calculates Moving Average Convergence Divergence (MACD)
        and its signal line. The MACD is used to identify changes in the strength,
        direction, momentum, and duration of a trend in a stock's price.
        The signal line is used to identify buy and sell signals.

        Parameters:
        ___________
        series (pd.Series):
            The time series data (prices or other values).
        ticker (str):
            The ticker symbol for the asset.
        fast (int, default=12):
            The number of periods for the fast (short-term) moving average.
        slow (int, default=26):
            The number of periods for the slow (long-term) moving average.
        signal (int, default=9):
            The number of periods for the signal line.

        Returns:
        ________
        tuple: A tuple containing:
            - MACD (pd.Series): The difference between the fast and slow EMAs.
            - Signal line (pd.Series): The EMA of the MACD line.
            - Histogram (pd.Series): The difference between the MACD line
            and the signal line.
        """
        try:
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
        except Exception as e:
            print(f"Failed to calculate MACD: {str(e)}")
            log.error(f"Failed to calculate MACD: {str(e)}")
            return pd.Series(), pd.Series(), pd.Series()

    def relative_strength_index(
        self, series: pd.Series, ticker: str, timeperiod: int = 14
    ):
        """
        Calculates Relative Strength Index (RSI) of a time series data.
        The Relative Strength Index (RSI) is a momentum oscillator that measures
        the speed and change of price movements. It ranges from 0 to 100,
        with levels above 70 indicating overbought conditions and levels below 30
        indicating oversold conditions.

        Parameters:
        ___________
        series (pd.Series):
            The time series data (prices or other values).
        ticker (str):
            The ticker symbol for the asset.
        timeperiod (int, default=14):
            The number of periods over which to calculate the RSI.

        Returns:
        ________
        indicator (pd.Series):
            The calculated RSI values for the given data with a name
            based on the ticker and time period.
        """
        try:
            indicator = talib.RSI(series, timeperiod=timeperiod)
            indicator.name = f"{ticker}_RSI{timeperiod}"
            return indicator
        except Exception as e:
            print(f"Failed to calculate RSI: {str(e)}")
            log.error(f"Failed to calculate RSI: {str(e)}")
            return pd.Series()

    def bollinger_bands(
        self,
        series: pd.Series,
        ticker: str,
        timeperiod: int = 20,
        nbdevup: int = 2,
        nbdevdn: int = 2,
    ):
        """
        Calculates Bollinger Bands, which consist of an upper, middle
        and lower band:
        - Middle band is a simple moving average (SMA) of the closing prices.
        - Upper and lower bands are standard deviations away from the SMA.

        Parameters:
        ___________
        series (pd.Series):
            Time series of stock prices.
        ticker (str):
            Stock ticker symbol.
        timeperiod (int, default=20):
            Lookback period for the SMA.
        nbdevup (int. default=2):
            Number of standard deviations for the upper band.
        nbdevdn (int, default=2):
            Number of standard deviations for the lower band.

        Returns:
        ________
        tuple: (upper_band, middle_band, lower_band)
        """
        try:
            upper, middle, lower = talib.BBANDS(
                series, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn
            )
            upper.name = f"{ticker}_BBupper"
            middle.name = f"{ticker}_BBmiddle"
            lower.name = f"{ticker}_BBlower"
            return upper, middle, lower
        except Exception as e:
            print(f"Failed to calculate Bollinger Bands: {str(e)}")
            log.error(f"Failed to calculate Bollinger Bands: {str(e)}")
            return pd.Series(), pd.Series(), pd.Series()

    def average_true_range(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        ticker: str,
        timeperiod: int = 14,
    ):
        """
        Calculates Average True Range (ATR), a volatility indicator.
        ATR measures market volatility by averaging the greatest of three values:
        - The current high minus the current low.
        - The absolute value of the current high minus the previous close.
        - The absolute value of the current low minus the previous close.

        Parameters:
        ___________
        high (pd.Series):
            High prices.
        low (pd.Series):
            Low prices.
        close (pd.Series):
            Closing prices.
        ticker (str):
            Stock ticker symbol.
        timeperiod (int, default=14):
            Lookback period for ATR calculation.

        Returns:
        ________
        atr (pd.Series):
            ATR values.
        """
        try:
            atr = talib.ATR(high, low, close, timeperiod=timeperiod)
            atr.name = f"{ticker}_ATR{timeperiod}"
            return atr
        except Exception as e:
            print(f"Failed to calculate ATR: {str(e)}")
            log.error(f"Failed to calculate ATR: {str(e)}")
            return pd.Series()

    def stochastic_oscillator(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        ticker: str,
        fastk_period: int = 14,
        slowk_period: int = 3,
        slowd_period: int = 3,
    ):
        """
        Calculates Stochastic Oscillator, a momentum indicator comparing
        a stock's closing price to its price range:
        - %K represents the raw stochastic value.
        - %D is a moving average of %K (signal line).

        Parameters:
        ___________
        high (pd.Series):
            High prices.
        low (pd.Series):
            Low prices.
        close (pd.Series):
            Closing prices.
        ticker (str):
            Stock ticker symbol.
        fastk_period (int, default=14):
            Period for %K calculation.
        slowk_period (int, default=3):
            Smoothing period for %K.
        slowd_period (int, default=3):
            Smoothing period for %D.

        Returns:
        ________
        tuple: (slowk, slowd).
        """
        try:
            slowk, slowd = talib.STOCH(
                high,
                low,
                close,
                fastk_period=fastk_period,
                slowk_period=slowk_period,
                slowd_period=slowd_period,
            )
            slowk.name = f"{ticker}_StochK{fastk_period}"
            slowd.name = f"{ticker}_StochD{slowd_period}"
            return slowk, slowd
        except Exception as e:
            print(f"Failed to calculate Stochastic Oscillator: {str(e)}")
            log.error(f"Failed to calculate Stochastic Oscillator: {str(e)}")
            return pd.Series(), pd.Series()

    def commodity_channel_index(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        ticker: str,
        timeperiod: int = 20,
    ):
        """
        Calculates Commodity Channel Index (CCI), which measures price
        deviation from the mean:
        - High values indicate overbought conditions.
        - Low values indicate oversold conditions.

        Parameters:
        ___________
        high (pd.Series):
            High prices.
        low (pd.Series):
            Low prices.
        close (pd.Series):
            Closing prices.
        ticker (str):
            Stock ticker symbol.
        timeperiod (int, default=20):
            Lookback period for CCI calculation.

        Returns:
        ________
        cci (pd.Series):
            CCI values.
        """
        try:
            cci = talib.CCI(high, low, close, timeperiod=timeperiod)
            cci.name = f"{ticker}_CCI{timeperiod}"
            return cci
        except Exception as e:
            print(f"Failed to calculate CCI: {str(e)}")
            log.error(f"Failed to calculate CCI: {str(e)}")
            return pd.Series()

    def williams_percent_r(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        ticker: str,
        timeperiod: int = 14,
    ):
        """
        Calculates Williams %R, a momentum indicator that measures
        overbought and oversold levels:
        - Values close to -100 indicate oversold conditions.
        - Values close to 0 indicate overbought conditions.

        Parameters:
        ___________
        high (pd.Series):
            High prices.
        low (pd.Series):
            Low prices.
        close (pd.Series):
            Closing prices.
        ticker (str):
            Stock ticker symbol.
        timeperiod (int, default=14):
            Lookback period for %R calculation.

        Returns:
        ________
        willr (pd.Series):
            Williams %R values.
        """
        try:
            willr = talib.WILLR(high, low, close, timeperiod=timeperiod)
            willr.name = f"{ticker}_WILLR{timeperiod}"
            return willr
        except Exception as e:
            print(f"Failed to calculate Williams %R: {str(e)}")
            log.error(f"Failed to calculate Williams %R: {str(e)}")
            return pd.Series()

    def calculate_spread(self, close: pd.Series, return_dataframe: bool = False):
        """
        Calculates the price spread and its normalized version between two assets.

        The spread is computed as the difference between the closing prices
        of two assets. The normalized spread is then obtained by subtracting
        the mean and dividing by the standard deviation of the spread,
        making it comparable across different time periods.

        Parameters:
        ___________
        close (pd.DataFrame):
            A DataFrame containing closing price data for two assets,
            where each column represents one asset.
        return_dataframe (bool):
            If True, returns the original DataFrame with 'Spread'
            and 'NormalizedSpread' columns added. If False,
            only returns the 'NormalizedSpread' series.

        Returns:
        ________
        pd.Series or pd.DataFrame:
            - If return_dataframe=False: Returns a pd.Series of the normalized spread.
            - If return_dataframe=True: Returns a pd.DataFrame with added 'Spread'
              and 'NormalizedSpread' columns.

        Example:
        ________
        >>> data = pd.DataFrame({'AssetX': [100, 102, 104], 'AssetY': [98, 101, 103]})
        >>> calculate_spread(data, return_dataframe=True)
           AssetX  AssetY  Spread  NormalizedSpread
        0    100     98       2            -1.0
        1    102    101       1             0.0
        2    104    103       1             1.0
        """
        try:
            df = close.copy()

            spread = df.iloc[:, 0] - df.iloc[:, 1]
            norm_spread = (spread - spread.mean()) / spread.std()

            if return_dataframe:
                df["Spread"] = spread
                df["NormalizedSpread"] = norm_spread
                return df
            else:
                norm_spread.name = "NormalizedSpread"
                return norm_spread
        except Exception as e:
            print(f"Failed to calculate normalized spread: {str(e)}")
            log.error(f"Failed to calculate normalized spread: {str(e)}")
            return pd.Series()

    def create_features(
        self,
        ticker_eqt: str,
        ticker_cpy: str,
        df_equity: pd.DataFrame,
        df_crypto: pd.DataFrame,
        config: dict,
        dropna: bool = False,
    ):
        """
        Generate technical indicators and spread features for a given
        trading pair.

        Parameters:
        ___________
        ticker_eqt (str):
            The ticker symbol for the equity asset.
        ticker_cpy (str):
            The ticker symbol for the crypto asset.
        df_equity (pd.DataFrame):
            DataFrame containing equity price data with columns such as 'High', 'Low', and 'Close'.
        df_crypto (pd.DataFrame):
            DataFrame containing cryptocurrency price data with similar columns.
        config (dict):
            Configuration dictionary specifying which indicators to compute and their parameters.
            Supported indicators:
            - 'ema' (list of time periods): Exponential Moving Average (EMA)
            - 'macd' (dict): Moving Average Convergence Divergence (MACD)
              - 'fast': Fast period
              - 'slow': Slow period
              - 'signal': Signal period
            - 'rsi' (list of time periods): Relative Strength Index (RSI)
            - 'bb' (dict): Bollinger Bands
              - 'timeperiod': Period for calculation
              - 'nbdevup': Number of standard deviations for upper band
              - 'nbdevdn': Number of standard deviations for lower band
            - 'atr' (dict): Average True Range (ATR)
              - 'timeperiod': Period for ATR calculation
            - 'stoch' (dict): Stochastic Oscillator
              - 'fastk_period': %K period
              - 'slowk_period': %D period (smoothing)
              - 'slowd_period': %D period (further smoothing)
            - 'cci' (dict): Commodity Channel Index (CCI)
              - 'timeperiod': Period for calculation
            - 'willr' (dict): Williams %R
              - 'timeperiod': Period for calculation
        dropna (bool, default=False):
            Whether to drop NaN values from the resulting feature set.

        Returns:
        ________
        feat (pd.DataFrame):
            A DataFrame containing the calculated technical indicators and the
            spread between the
            given equity and cryptocurrency assets.

        Notes:
        ______
        - The function first extracts high, low, and close prices for the asset pair.
        - It then computes various technical indicators based on the provided `config`.
        - The spread between the closing prices of the two assets is also computed.
        - If `dropna` is True, rows containing NaN values are removed.
        """
        try:
            if config is None:
                config = self.default_config

            price_pairs = [ticker_eqt, ticker_cpy]
            list_indicators = []

            high = data.process_pairs_series(
                ticker_eqt,
                ticker_cpy,
                df_equity.reset_index(),
                df_crypto.reset_index(),
                "High",
            )
            low = data.process_pairs_series(
                ticker_eqt,
                ticker_cpy,
                df_equity.reset_index(),
                df_crypto.reset_index(),
                "Low",
            )
            close = data.process_pairs_series(
                ticker_eqt,
                ticker_cpy,
                df_equity.reset_index(),
                df_crypto.reset_index(),
                "Close",
            )

            for ticker in price_pairs:
                if "ema" in config and isinstance(config["ema"], list):
                    for timeperiod in config["ema"]:
                        list_indicators.append(
                            self.exponential_moving_average(
                                close[ticker], ticker, timeperiod
                            )
                        )

                if "macd" in config:
                    macd, macdsignal, macdhist = (
                        self.moving_average_convergence_divergence(
                            close[ticker],
                            ticker,
                            config["macd"]["fast"],
                            config["macd"]["slow"],
                            config["macd"]["signal"],
                        )
                    )
                    list_indicators.append(macd)

                if "rsi" in config and isinstance(config["rsi"], list):
                    for timeperiod in config["rsi"]:
                        list_indicators.append(
                            self.relative_strength_index(
                                close[ticker], ticker, timeperiod
                            )
                        )

                if "bb" in config:
                    upper, middle, lower = self.bollinger_bands(
                        close[ticker],
                        ticker,
                        config["bb"]["timeperiod"],
                        config["bb"]["nbdevup"],
                        config["bb"]["nbdevdn"],
                    )
                    list_indicators.extend([upper, middle, lower])

                if "atr" in config:
                    atr = self.average_true_range(
                        high[ticker],
                        low[ticker],
                        close[ticker],
                        ticker,
                        config["atr"]["timeperiod"],
                    )
                    list_indicators.append(atr)

                if "stoch" in config:
                    slowk, slowd = self.stochastic_oscillator(
                        high[ticker],
                        low[ticker],
                        close[ticker],
                        ticker,
                        config["stoch"]["fastk_period"],
                        config["stoch"]["slowk_period"],
                        config["stoch"]["slowd_period"],
                    )
                    list_indicators.extend([slowk, slowd])

                if "cci" in config:
                    cci = self.commodity_channel_index(
                        high[ticker],
                        low[ticker],
                        close[ticker],
                        ticker,
                        config["cci"]["timeperiod"],
                    )
                    list_indicators.append(cci)

                if "willr" in config:
                    willr = self.williams_percent_r(
                        high[ticker],
                        low[ticker],
                        close[ticker],
                        ticker,
                        config["willr"]["timeperiod"],
                    )
                    list_indicators.append(willr)

            # Compute pair spread on Close
            feat = pd.DataFrame(list_indicators).T
            feat.loc[:, "Spread"] = close.iloc[:, 0] - close.iloc[:, 1]

            if dropna:
                feat = feat.dropna()
            log.info("Features created successfully.")
            return feat
        except Exception as e:
            print(f"Failed to create features:{str(e)}")
            log.error(f"Failed to create features:{str(e)}")
            return pd.DataFrame()

    def normalize_features(self, data: pd.DataFrame, scaler: str = "StandardScaler"):
        """
        Normalize the given feature DataFrame using the specified scaler.

        Parameters:
        ___________
        feat (pd.DataFrame):
            The DataFrame containing features to be normalized.
        scaler (str):
            The type of scaler to use. Options are:
            - 'StandardScaler' (default): Standardizes features (0 mean, 1 variance).
            - 'MinMax': Scales features to range [0,1].

        Returns:
        ________
        tuple: (normalized DataFrame, scaler object)
        """
        try:
            data = data.copy()

            # Drop 'NormalizedSpread' if it exists
            if "NormalizedSpread" in data.columns:
                data = data.drop(columns=["NormalizedSpread"])

            # Select scaler
            if scaler == "StandardScaler":
                scaler_obj = StandardScaler()
            elif scaler == "MinMax":
                scaler_obj = MinMaxScaler()
            else:
                raise ValueError(
                    "Invalid scaler type. Choose 'StandardScaler' or 'MinMax'."
                )

            # Fit and transform the data
            data_scaled = scaler_obj.fit_transform(data)

            # Convert back to DataFrame with same column names
            data_scaled_df = pd.DataFrame(
                data_scaled, index=data.index, columns=data.columns
            )
            log.info("Features normalized successfully.")
            return data_scaled_df, scaler_obj
        except Exception as e:
            print(f"Failed to normalize features:{str(e)}")
            log.error(f"Failed to normalize features:{str(e)}")
            return pd.DataFrame(), None
