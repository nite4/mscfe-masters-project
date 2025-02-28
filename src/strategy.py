import portfolio_performance as pp

from utils import log


class Strategy:
    """
    Trading Strategy Class for long/short pair trading using forecast spreads.

    Assumptions:
      1) A trading signal is generated based on PredictedSignal:
         e.g. 'No action', 'Long SPY_ETF / Short AVAXUSDT', or 'Short SPY_ETF / Long AVAXUSDT'.
      2) A signal is confirmed only if it occurs at least m times consecutively (default m=1).
      3) At a confirmed signal, the strategy invests a maximum of 1000 USD per asset.
      4) Total exposure per asset cannot exceed 25% of the initial capital.
      5) If no reversal signal occurs, a position is force‐closed after 144 periods.

    Note: This version always uses NextOpen prices for both trade execution and closure,
          and the allowed instruments check has been removed.
    """

    def __init__(
        self,
        initial_capital=10000,
        m_threshold=1,
        max_exposure=0.25,
        max_trade_size=1000,
        close_threshold=144,
    ):
        self.balance = initial_capital
        self.asset_max_exposure = max_exposure * initial_capital
        self.m_threshold = m_threshold
        self.max_trade_size = max_trade_size
        self.close_threshold = close_threshold
        self.trade_history = []  # Record of all executed trades
        self.positions = {}  # Open positions, keyed by pair

    def execute_trade(
        self, pair, signal, open_time, asset_A, asset_B, next_open_A, next_open_B
    ):
        """
        Execute a trade for the given pair using NextOpen prices.
        Trades 1000 USD worth of each asset.
        """
        # Determine the trade direction from the signal.
        # Assumes signal is either 'Long A / Short B' or 'Short A / Long B'
        if "Long" in signal and "Short" in signal:
            if signal.strip().split()[0] == "Long":
                direction = "Long A / Short B"
            else:
                direction = "Short A / Long B"
        else:
            return  # Invalid signal

        # Use the NextOpen prices provided.
        price_A = next_open_A
        price_B = next_open_B

        # Calculate the quantity for each asset.
        qty_A = self.max_trade_size / price_A
        qty_B = self.max_trade_size / price_B

        # Check current exposure (based on open trades) for each asset.
        exposure_A = sum(
            trade["trade_size"]
            for trade in self.trade_history
            if (trade.get("asset_A") == asset_A) & (not trade.get("closed", False))
        )
        exposure_B = sum(
            trade["trade_size"]
            for trade in self.trade_history
            if (trade.get("asset_B") == asset_B) & (not trade.get("closed", False))
        )

        if (exposure_A + self.max_trade_size > self.asset_max_exposure) | (
            exposure_B + self.max_trade_size > self.asset_max_exposure
        ):
            return  # Skip trade if exposure limits would be exceeded

        # Record the trade
        trade_record = {
            "pair": pair,
            "signal": signal,
            "open_time": open_time,
            "asset_A": asset_A,
            "asset_B": asset_B,
            "price_A": price_A,
            "price_B": price_B,
            "trade_size": self.max_trade_size,
            "direction": direction,
            "closed": False,
        }
        self.trade_history.append(trade_record)

        # Open a new position
        self.positions[pair] = {
            "entry_time": open_time,
            "asset_A": asset_A,
            "asset_B": asset_B,
            "direction": direction,
            "entry_price_A": price_A,
            "entry_price_B": price_B,
            "quantity_A": qty_A,
            "quantity_B": qty_B,
            "age": 0,
        }

    def close_trade(self, pair, close_time, asset_A, asset_B, next_open_A, next_open_B):
        """
        Close an open position for the given pair using NextOpen prices.
        """
        if pair not in self.positions:
            return

        pos = self.positions.pop(pair)
        price_A = next_open_A
        price_B = next_open_B

        # Calculate P&L based on the trade direction
        if pos["direction"] == "Long A / Short B":
            pnl_A = (price_A - pos["entry_price_A"]) * pos["quantity_A"]
            pnl_B = (pos["entry_price_B"] - price_B) * pos["quantity_B"]
        else:  # 'Short A / Long B'
            pnl_A = (pos["entry_price_A"] - price_A) * pos["quantity_A"]
            pnl_B = (price_B - pos["entry_price_B"]) * pos["quantity_B"]
        total_pnl = pnl_A + pnl_B
        self.balance += total_pnl

        # Update trade history to mark trades as closed
        for trade in self.trade_history:
            if (trade["pair"] == pair) & (not trade.get("closed", False)):
                trade.update(
                    {
                        "close_time": close_time,
                        "close_price_A": price_A,
                        "close_price_B": price_B,
                        "pnl": total_pnl,
                        "closed": True,
                    }
                )

    def run_strategy(self, df):
        """
        Executes the trading strategy over a dataframe of signals.

        The dataframe is expected to have the following columns:
          - OpenTime (timestamp)
          - Pair (e.g. 'AMZN BNBUSDT')
          - NormalizedSpread
          - PredictedSpread
          - PredictedSignal (e.g. 'No action', 'Long SPY_ETF / Short AVAXUSDT', etc.)
          - AssetA
          - AssetB
          - NextOpenA
          - NextOpenB

        Prices from NextOpenA and NextOpenB are always used.
        """
        df["OpenTime"] = pd.to_datetime(df["OpenTime"])
        df = df.sort_values(by=["Pair", "OpenTime"])

        # Dictionary to count consecutive signals per pair
        signal_counters = {pair: 0 for pair in df["Pair"].unique()}

        for _, row in df.iterrows():
            pair = row["Pair"]
            signal = row["PredictedSignal"]
            open_time = row["OpenTime"]
            asset_A = row["AssetA"]
            asset_B = row["AssetB"]
            next_open_A = row["NextOpenA"]
            next_open_B = row["NextOpenB"]

            position_open = pair in self.positions

            # Increment counter if signal is not "No action", otherwise reset
            if signal != "No action":
                signal_counters[pair] += 1
            else:
                signal_counters[pair] = 0

            # Once a signal is confirmed over m consecutive periods, execute logic
            if signal_counters[pair] >= self.m_threshold:
                if not position_open:
                    self.execute_trade(
                        pair,
                        signal,
                        open_time,
                        asset_A,
                        asset_B,
                        next_open_A,
                        next_open_B,
                    )
                else:
                    current_pos = self.positions[pair]
                    reversal = False
                    # Determine reversal based on the new signal and current position.
                    if (current_pos["direction"] == "Long A / Short B") & (
                        "Short" in signal
                    ):
                        reversal = True
                    elif (current_pos["direction"] == "Short A / Long B") & (
                        "Long" in signal
                    ):
                        reversal = True
                    if reversal:
                        self.close_trade(
                            pair, open_time, asset_A, asset_B, next_open_A, next_open_B
                        )
                        self.execute_trade(
                            pair,
                            signal,
                            open_time,
                            asset_A,
                            asset_B,
                            next_open_A,
                            next_open_B,
                        )
                # Reset counter after executing a trade
                signal_counters[pair] = 0

            # For any open position, increment its age
            # and force-close if it reaches the threshold
            if pair in self.positions:
                self.positions[pair]["age"] += 1
                if self.positions[pair]["age"] >= self.close_threshold:
                    self.close_trade(
                        pair, open_time, asset_A, asset_B, next_open_A, next_open_B
                    )

        return {
            "Final Portfolio Value": self.balance,
            "Trades Executed": self.trade_history,
        }
