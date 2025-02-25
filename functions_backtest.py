import datetime as dt
import logging
import itertools
from typing import Tuple

from binance.client import Client as BinanceClient
import pandas as pd
import numpy as np
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from scipy.stats import shapiro

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BINANCE_INTERVAL_TO_PANDAS_FREQ = {
    "1m": "T",    # 1 Minute
    "5m": "5T",   # 5 Minutes
    "15m": "15T", # 15 Minutes
    "30m": "30T", # 30 Minutes
    "1h": "H",    # Hourly
    "4h": "4H",   # 4 Hours
    "1d": "D"     # Daily
}

class BinanceDataHandler:
    """
    A class to handle fetching and processing historical candlestick data from the Binance API.
    """

    def __init__(self, api_key: str = None, api_secret: str = None):
        """
        Initialize the BinanceDataHandler with optional API credentials.
        
        Parameters
        ----------
        api_key : str, optional
            The Binance API key.
        api_secret : str, optional
            The Binance API secret.
        """
        self.client = BinanceClient(api_key, api_secret)

    def fetch_data(
        self,
        symbol: str,
        interval: str,
        start_time: dt.datetime,
        end_time: dt.datetime
    ) -> pd.DataFrame:
        """
        Fetch historical candlestick data from the Binance API and return it as a pandas DataFrame.
        
        Parameters
        ----------
        symbol : str
            The trading pair symbol (e.g., 'BTCUSDT').
        interval : str
            The time interval between data points (e.g., '1m', '5m', '1h').
        start_time : datetime
            The start date for the data retrieval.
        end_time : datetime
            The end date for the data retrieval.
        
        Returns
        -------
        pd.DataFrame or None
            A DataFrame containing the historical data if successful; otherwise, None.
        """
        try:
            start_ms = int(start_time.timestamp() * 1000)  # Convert to milliseconds
            end_ms = int(end_time.timestamp() * 1000)
            
            data = self.client.get_historical_klines(symbol, interval, start_ms, end_ms)
            if not data:
                logger.warning(f"No data retrieved for {symbol}. The request might have failed or returned empty data.")
                return None

            logger.info(f"Retrieved {len(data)} rows for {symbol}")

            columns = [
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "num_trades",
                "taker_buy_base", "taker_buy_quote", "ignore"
            ]
            df = pd.DataFrame(data, columns=columns)

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
            df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')
            df.set_index("timestamp", inplace=True)

            float_columns = ["open", "high", "low", "close", "volume", "quote_asset_volume", "taker_buy_base", "taker_buy_quote"]
            df[float_columns] = df[float_columns].astype(float)

            int_columns = ["num_trades", "ignore"]
            df[int_columns] = df[int_columns].astype(int)

            if interval in BINANCE_INTERVAL_TO_PANDAS_FREQ:
                pandas_freq = BINANCE_INTERVAL_TO_PANDAS_FREQ[interval]
                df = df.asfreq(pandas_freq)  # Automatically set the frequency to ease time series analysis
                df = df.fillna(method="ffill")

            return df

        except Exception as e:
            logger.error(f"Failed to retrieve data for {symbol}: {str(e)}", exc_info=True)
            return None

    @staticmethod
    def select_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """
        Select and return the OHLCV (Open, High, Low, Close, Volume) columns from the provided DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame that should contain at least the columns 'open', 'high', 'low', 'close', and 'volume'.
        
        Returns
        -------
        pd.DataFrame or None
            A new DataFrame containing only the OHLCV columns if successful; otherwise, None.
        """
        required_columns = ["open", "high", "low", "close", "volume"]
        try:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Input DataFrame is missing required columns: {missing_columns}")

            df_ohlcv = df[required_columns].copy()
            return df_ohlcv

        except Exception as e:
            logger.error(f"Failed to select OHLCV columns: {str(e)}", exc_info=True)
            return None

    @staticmethod
    def aggregate_data(df: pd.DataFrame, frequency: str = '1H') -> pd.DataFrame:
        """
        Aggregate OHLCV data to a specified frequency using the datetime index.
        
        Parameters
        ----------
        df : pd.DataFrame
            A DataFrame containing OHLCV data with a datetime index.
        frequency : str, optional
            A string representing the resampling frequency (e.g., '1H' for hourly, '1D' for daily).
        
        Returns
        -------
        pd.DataFrame or None
            A new DataFrame with aggregated OHLCV data if successful; otherwise, None.
        """
        try:
            if df is None or df.empty:
                logger.warning("The input DataFrame is None or empty. Nothing to aggregate.")
                return df

            df_aggregated = df.resample(frequency).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            logger.info(f"Data aggregated successfully with frequency '{frequency}'.")
            return df_aggregated

        except Exception as e:
            logger.error(f"Failed to aggregate data: {str(e)}", exc_info=True)
            return None


class DataHandler:
    """
    A class for performing various data manipulation tasks.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataHandler with a pandas DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be manipulated.
        """
        self.df = df

    def remove_duplicates(self) -> pd.DataFrame:
        """
        Remove duplicate rows from the DataFrame and log the number of duplicates removed.
        
        Returns
        -------
        pd.DataFrame or None
            A new DataFrame with duplicate rows removed if successful; otherwise, None.
        """
        try:
            if self.df is None or self.df.empty:
                logger.warning("The input DataFrame is None or empty. Nothing to remove.")
                return self.df

            duplicate_count = self.df.duplicated().sum()
            df_cleaned = self.df.drop_duplicates()
            logger.info(f"Removed {duplicate_count} duplicate rows from the DataFrame.")
            return df_cleaned

        except Exception as e:
            logger.error(f"Failed to remove duplicates: {str(e)}", exc_info=True)
            return None

    def convert_timezone(self, target_timezone: str = "CET") -> pd.DataFrame:
        """
        Convert the timestamps in the DataFrame's index to the specified timezone.
        If the index is naive, it is assumed to be in UTC before conversion.
        
        Parameters
        ----------
        target_timezone : str, optional
            The target timezone to convert the index timestamps to. Default is "CET".
        
        Returns
        -------
        pd.DataFrame or None
            A new DataFrame with the index timestamps converted to the specified timezone if successful; otherwise, None.
        """
        try:
            df_copy = self.df.copy()
            if df_copy.index.tz is None:
                df_copy.index = df_copy.index.tz_localize('UTC').tz_convert(target_timezone)
            else:
                df_copy.index = df_copy.index.tz_convert(target_timezone)

            logger.info(f"Converted DataFrame index to timezone: {target_timezone}")
            return df_copy

        except Exception as e:
            logger.error(f"Failed to convert index timezone: {str(e)}", exc_info=True)
            return None

    def calculate_typical_price(
        self,
        col_high: str = "high",
        col_low: str = "low",
        col_close: str = "close",
        new_col: str = "typical_price"
    ) -> pd.DataFrame:
        """
        Compute the typical price for each row using the formula:
        (High + Low + Close) / 3 and add it as a new column.
        
        Parameters
        ----------
        col_high : str, optional
            Column name for the high price. Default is "high".
        col_low : str, optional
            Column name for the low price. Default is "low".
        col_close : str, optional
            Column name for the close price. Default is "close".
        new_col : str, optional
            Column name for the typical price. Default is "typical_price".
        
        Returns
        -------
        pd.DataFrame or None
            The DataFrame with the new typical price column if successful; otherwise, None.
        """
        try:
            required_columns = {col_high, col_low, col_close}
            if not required_columns.issubset(self.df.columns):
                missing_cols = required_columns - set(self.df.columns)
                logger.error(f"Missing required column(s) for typical price calculation: {missing_cols}")
                return None

            self.df[new_col] = (self.df[col_high] + self.df[col_low] + self.df[col_close]) / 3
            return self.df

        except Exception as e:
            logger.error(f"Failed to calculate typical price: {str(e)}", exc_info=True)
            return None

    def calculate_log(self, quantity_col: str) -> pd.DataFrame:
        """
        Calculate the log returns for the specified quantity column.

        Parameters
        ----------
        quantity_col : str
            The column name for which log returns should be calculated.

        Returns
        -------
        pd.DataFrame or None
            A new DataFrame with an added column of log returns if successful; otherwise, None.
        """
        try:
            df = self.df.copy()

            if quantity_col not in df.columns:
                raise ValueError(f"Column '{quantity_col}' not found in the DataFrame.")

            df[f"log_{quantity_col}"] = np.log(df[quantity_col]).diff()

            logger.info(f"Log returns calculated for column '{quantity_col}'.")

            return df

        except Exception as e:
            logger.error(f"Failed to calculate log returns for '{quantity_col}': {str(e)}", exc_info=True)
            return None


class DataVisualisation:
    """
    A class for performing various data visualisation tasks.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        time_col: str = None,
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume"
    ):
        """
        Initialize the DataVisualisation instance.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the OHLCV data.
        time_col : str, optional
            Name of the column representing time.
        open_col : str, optional
            Column name for the open price.
        high_col : str, optional
            Column name for the high price.
        low_col : str, optional
            Column name for the low price.
        close_col : str, optional
            Column name for the close price.
        volume_col : str, optional
            Column name for the volume data.
        """
        self.df = df
        self.time_col = time_col
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col

        # If a time column is specified and exists in df, set it as the index.
        if self.df is not None and self.time_col is not None and self.time_col in self.df.columns:
            self.df.set_index(self.time_col, inplace=True)

    def plot_candlestick_volume(
        self,
        base_ccy: str = None,
        quote_ccy: str = None,
        title: str = "Candlestick and Volume Chart",
        height: int = 800,
        show_rangeslider: bool = False,
        open_col: str = None,
        high_col: str = None,
        low_col: str = None,
        close_col: str = None,
        volume_col: str = None
    ):
        """
        Plot a candlestick chart with volume bars using OHLCV data.
        
        Parameters
        ----------
        base_ccy : str
            The base currency of the trading pair (e.g., 'BTC' in 'BTCUSDT').
        quote_ccy : str
            The quote currency of the trading pair (e.g., 'USDT' in 'BTCUSDT').
        title : str, optional
            The title of the chart. Default is "Candlestick and Volume Chart".
        height : int, optional
            The height of the chart in pixels. Default is 800.
        show_rangeslider : bool, optional
            Whether to display the range slider below the candlestick chart. Default is False.
        open_col : str, optional
            The column name for the open price. Defaults to the instance's `open_col`.
        high_col : str, optional
            The column name for the high price. Defaults to the instance's `high_col`.
        low_col : str, optional
            The column name for the low price. Defaults to the instance's `low_col`.
        close_col : str, optional
            The column name for the close price. Defaults to the instance's `close_col`.
        volume_col : str, optional
            The column name for the volume data. Defaults to the instance's `volume_col`.
        
        Returns
        -------
        None
            Displays the plot if successful; otherwise, returns None.
        """
        try:
            if self.df is None:
                logger.error("No data provided for plotting candlestick and volume chart.")
                return None

            df = self.df
            open_col = open_col or self.open_col
            high_col = high_col or self.high_col
            low_col = low_col or self.low_col
            close_col = close_col or self.close_col
            volume_col = volume_col or self.volume_col

            candlesticks = go.Candlestick(
                x=df.index,
                open=df[open_col],
                high=df[high_col],
                low=df[low_col],
                close=df[close_col],
                name="OHLC"
            )
            volume_bars = go.Bar(
                x=df.index,
                y=df[volume_col],
                marker={"color": "rgb(59, 59, 59)"},
                name="Volume"
            )

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(candlesticks, secondary_y=True)
            fig.add_trace(volume_bars, secondary_y=False)

            fig.update_layout(
                title=title,
                height=height,
                xaxis=dict(rangeslider=dict(visible=show_rangeslider))
            )
            fig.update_yaxes(title_text=f"Price {quote_ccy}", secondary_y=True, showgrid=True)
            fig.update_yaxes(title_text=f"Volume {base_ccy}", secondary_y=False, showgrid=False)

            fig.show()

        except Exception as e:
            logger.error(f"Failed to plot candlestick volume chart: {str(e)}", exc_info=True)
            return None

    def plot_quantity(
        self,
        quantity_col: str,
        title: str = None,
        width: int = 800,
        height: int = 400
    ) -> alt.Chart:
        """
        Plot a quantity column over time using Altair.
        
        Parameters
        ----------
        quantity_col : str
            Column name to plot.
        title : str, optional
            Title of the chart.
        width : int, optional
            Width of the chart.
        height : int, optional
            Height of the chart.
        
        Returns
        -------
        alt.Chart or None
            An Altair chart object if successful; otherwise, None.
        """
        try:
            if self.df is None:
                logger.error("No data provided for plotting quantity.")
                return None

            if quantity_col not in self.df.columns:
                logger.error(f"Invalid quantity column: {quantity_col}. Ensure it exists in the DataFrame.")
                return None

            time_col = self.time_col or self.df.index.name
            title = title or f"{quantity_col.capitalize()} Plot"

            base = alt.Chart(self.df.reset_index()).encode(
                x=alt.X(f"{time_col}:T", title="Time")
            )
            quantity_line = base.mark_line(color="steelblue").encode(
                y=alt.Y(f"{quantity_col}:Q", title=quantity_col.capitalize()),
                tooltip=[f"{time_col}:T", f"{quantity_col}:Q"]
            )
            chart = alt.layer(quantity_line).interactive().properties(
                title=title,
                width=width,
                height=height
            )
            return chart

        except Exception as e:
            logger.error(f"Failed to generate plot for {quantity_col}: {str(e)}", exc_info=True)
            return None

    def plot_moving_average(
        self,
        quantity_col: str,
        window_short: int = 7,
        window_long: int = 30,
        title: str = None,
        width: int = 800,
        height: int = 400
    ) -> alt.Chart:
        """
        Plot spot prices along with short- and long-window moving averages using Altair.
        
        Parameters
        ----------
        quantity_col : str
            Name of the column containing prices.
        window_short : int, optional
            Number of periods for the short moving average window.
        window_long : int, optional
            Number of periods for the long moving average window.
        title : str, optional
            Title of the chart.
        width : int, optional
            Width of the chart.
        height : int, optional
            Height of the chart.
        
        Returns
        -------
        alt.Chart or None
            An Altair chart object if successful; otherwise, None.
        """
        try:
            if self.df is None:
                logger.error("No data provided for plotting moving averages.")
                return None

            if quantity_col not in self.df.columns:
                logger.error(f"Invalid price column: {quantity_col}. Ensure it exists in the DataFrame.")
                return None

            df = self.df.copy()
            df["MA_short"] = df[quantity_col].rolling(window=window_short).mean()
            df["MA_long"] = df[quantity_col].rolling(window=window_long).mean()

            time_col = self.time_col or df.index.name
            title = title or f"Moving Averages ({window_short}-period & {window_long}-period)"

            base = alt.Chart(df.reset_index()).encode(
                x=alt.X(f"{time_col}:T", title="Time")
            )
            spot_line = base.mark_line(color="steelblue").encode(
                y=alt.Y(f"{quantity_col}:Q", title=quantity_col.capitalize()),
                tooltip=[f"{time_col}:T", f"{quantity_col}:Q"]
            )
            short_line = base.mark_line(color="orangered").encode(
                y=alt.Y("MA_short:Q", title="Short MA"),
                tooltip=[f"{time_col}:T", "MA_short:Q"]
            )
            long_line = base.mark_line(color="green").encode(
                y=alt.Y("MA_long:Q", title="Long MA"),
                tooltip=[f"{time_col}:T", "MA_long:Q"]
            )
            chart = alt.layer(spot_line, short_line, long_line).interactive().properties(
                title=title,
                width=width,
                height=height
            )
            return chart

        except Exception as e:
            logger.error(f"Failed to plot moving averages: {str(e)}", exc_info=True)
            return None

    def plot_heatmap(
        self,
        quantity_col: str,
        plot_type: str,
        aggfunc: str = "mean",
        day_of_week: int = None,
        figsize: tuple = (10, 6),
        cmap: str = "plasma"
    ) -> None:
        """
        Plot a heatmap for a given quantity based on time components of the DataFrame's index.

        Parameters
        ----------
        quantity_col : str
            The name of the column in the DataFrame to be plotted.
        plot_type : str
            The type of heatmap to generate. Accepted values are:
                - "day_hour": Generates a heatmap of Hour-of-Day vs Day-of-Week.
                - "hour_minute": Generates a heatmap of Minute-of-Hour vs Hour-of-Day.
        aggfunc : str, optional
            The aggregation function to use when pivoting the data (default is "mean").
        day_of_week : int, optional
            When using plot_type "hour_minute", optionally filter data to a specific day of the week
            (0=Monday, 6=Sunday). If None, no day-based filtering is applied.
        figsize : tuple, optional
            A tuple specifying the figure size (width, height) for the plot (default is (10, 6)).
        cmap : str, optional
            The colormap to use for the heatmap (default is "YlGnBu").

        Returns
        -------
        None
            The function displays the heatmap. If an error occurs, it logs the error and returns None.
        """
        try:
            if self.df is None:
                logger.error("No data provided for plotting heatmap.")
                return None

            df = self.df.copy()
            if quantity_col not in df.columns:
                logger.error(f"Column '{quantity_col}' not found in the DataFrame.")
                return None

            if plot_type == "day_hour":
                df["day_of_week"] = df.index.dayofweek
                df["hour_of_day"] = df.index.hour

                pivot_data = df.pivot_table(
                    index="hour_of_day",
                    columns="day_of_week",
                    values=quantity_col,
                    aggfunc=aggfunc
                )
                title = f"{quantity_col.capitalize()} by Day-of-Week and Hour-of-Day ({aggfunc.capitalize()})"
                xlabel = "Day of Week (Mon=0 ... Sun=6)"
                ylabel = "Hour of Day (0 ... 23)"

            elif plot_type == "hour_minute":
                df["hour_of_day"] = df.index.hour
                df["minute_of_hour"] = df.index.minute

                # Optionally filter for a specific day of the week.
                day_label = ""
                if day_of_week is not None:
                    df = df[df.index.dayofweek == day_of_week]
                    day_label = f" on day {day_of_week}"

                pivot_data = df.pivot_table(
                    index="minute_of_hour",
                    columns="hour_of_day",
                    values=quantity_col,
                    aggfunc=aggfunc
                )
                title = f"{quantity_col.capitalize()} by Hour-of-Day and Minute-of-Hour{day_label} ({aggfunc.capitalize()})"
                xlabel = "Hour of Day (0 ... 23)"
                ylabel = "Minute of Hour (0 ... 59)"

            else:
                logger.error("Invalid plot_type provided. Supported types are 'day_hour' and 'hour_minute'.")
                return None

            plt.figure(figsize=figsize)
            sns.heatmap(pivot_data, cmap=cmap, annot=False, fmt=".1f")
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            logger.error(f"Failed to plot heatmap: {str(e)}", exc_info=True)
            return None

    def plot_scatter(
        self,
        quantity_col_1: str,
        quantity_col_2: str,
        figsize: tuple = (10, 6),
        alpha: float = 0.5,
        color: str = "blue",
        grid: bool = True
    ) -> None:
        """
        Plot a scatter plot to show the relationship between two quantities in the DataFrame.

        Parameters
        ----------
        quantity_col_1 : str
            The name of the column in the DataFrame to be used for the x-axis.
        quantity_col_2 : str
            The name of the column in the DataFrame to be used for the y-axis.
        figsize : tuple, optional
            A tuple specifying the figure size (width, height). Default is (10, 6).
        alpha : float, optional
            The alpha blending value for the scatter points, between 0 (transparent) and 1 (opaque). Default is 0.5.
        color : str, optional
            The color of the scatter points. Default is "blue".
        grid : bool, optional
            Whether to display a grid on the plot. Default is True.

        Returns
        -------
        None
            Displays the scatter plot. If an error occurs, the error is logged and None is returned.
        """
        try:
            if self.df is None:
                logger.error("No data provided for plotting scatter plot.")
                return None

            df = self.df.copy()

            if quantity_col_1 not in df.columns or quantity_col_2 not in df.columns:
                logger.error(f"Required columns '{quantity_col_1}' or '{quantity_col_2}' not found in the DataFrame.")
                return None

            plt.figure(figsize=figsize)
            plt.scatter(df[quantity_col_1], df[quantity_col_2], alpha=alpha, color=color)
            plt.title(f"Relationship between {quantity_col_1.capitalize()} and {quantity_col_2.capitalize()}")
            plt.xlabel(f"{quantity_col_1.capitalize()}")
            plt.ylabel(f"{quantity_col_2.capitalize()}")
            plt.grid(grid)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            logger.error(f"Failed to plot scatter plot: {str(e)}", exc_info=True)
            return None


class TimeSeriesAnalysis:
    """
    A class to perform ARMA/ARIMA time series analysis on log returns or any other stationary series.
    """

    def __init__(self, df, quantity_col: str):
        """
        Initialize the TimeSeriesAnalysis with a DataFrame and the name of the column to analyze.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing your time series data (e.g., log returns).
        quantity_col : str
            The column name in df to be used for time series modeling (e.g. 'log_close').
        """
        self.df = df
        self.quantity_col = quantity_col

    def check_stationarity(
        self,
        significance_level: float = 0.05,
        maxlag: int = 20,
        regression: str = 'c',
        autolag: str = 'AIC'
    ) -> dict:
        """
        Perform the Augmented Dickey-Fuller (ADF) test to check the stationarity of the series
        in self.df[self.quantity_col].

        A low p-value (below the chosen significance level) indicates that the null hypothesis 
        (the series has a unit root, i.e., is non-stationary) can be rejected, suggesting 
        the series is stationary.

        Parameters
        ----------
        significance_level : float, optional
            The significance level for the stationarity test. Default is 0.05.
        maxlag : int, optional
            Maximum lag used for the ADF test. If None, the function automatically selects the appropriate lag.
        regression : str, optional
            The regression type for the ADF test. Options are 'c', 'ct', 'ctt', 'nc'.
            Default is 'c' (constant in the test regression).
        autolag : str, optional
            The method to use when automatically determining the lag. Options are 'AIC', 'BIC', 't-stat', None.
            Default is 'AIC'.

        Returns
        -------
        dict or None
            A dictionary with the ADF test results if successful; otherwise, None.
            The dictionary keys include:
                - 'test_statistic': The ADF test statistic.
                - 'p_value': The p-value for the test.
                - 'n_lags': The number of lags used.
                - 'n_obs': The number of observations used in the regression.
                - 'critical_values': The critical values for the test statistic at different confidence levels.
                - 'stationary': A boolean indicating whether the series is stationary (True)
                  or not (False), based on the provided significance_level.
        """
        try:
            if self.df is None or self.df.empty:
                logger.warning("The DataFrame is None or empty. Unable to perform stationarity check.")
                return None

            if self.quantity_col not in self.df.columns:
                logger.error(f"Column '{self.quantity_col}' not found in the DataFrame.")
                return None

            series = self.df[self.quantity_col].dropna()

            # Perform the Augmented Dickey-Fuller test
            adf_result = adfuller(series, maxlag=maxlag, regression=regression, autolag=autolag)
            test_statistic, p_value, n_lags, n_obs, critical_values, icbest = adf_result

            is_stationary = p_value < significance_level

            results = {
                'test_statistic': test_statistic,
                'p_value': p_value,
                'n_lags': n_lags,
                'n_obs': n_obs,
                'critical_values': critical_values,
                'stationary': is_stationary
            }

            logger.info(
                f"ADF test on column '{self.quantity_col}': p-value={p_value}, "
                f"stationary={is_stationary}"
            )
            return results

        except Exception as e:
            logger.error(
                f"Failed to perform stationarity check on column '{self.quantity_col}': {str(e)}",
                exc_info=True
            )
            return None

    def evaluate_arma_models(
        self,
        p_range: Tuple[int, int] = (0, 5),
        d: int = 0,
        q_range: Tuple[int, int] = (0, 5)
    ) -> pd.DataFrame:
        """
        Evaluate ARIMA(p, d, q) models using AIC and BIC for model selection.

        This method iterates through different combinations of p (AR order), d (differencing order),
        and q (MA order) within the specified ranges, fits ARIMA models to the time series, 
        and returns a DataFrame of results sorted by AIC. Additionally, it provides a summary statement
        indicating the best ARIMA model according to AIC and BIC.

        Parameters
        ----------
        p_range : tuple of (int, int), optional
            The range of values for the autoregressive (AR) component (default: (0, 5)).
        d : int, optional
            The order of differencing (default: 0).
        q_range : tuple of (int, int), optional
            The range of values for the moving average (MA) component (default: (0, 5)).

        Returns
        -------
        results_df : pd.DataFrame
            A DataFrame containing the evaluated ARIMA models with their corresponding p, d, q,
            AIC, and BIC values, sorted by AIC.
        """

        time_series = self.df[self.quantity_col]

        if not isinstance(time_series, pd.Series):
            raise TypeError("time_series must be a Pandas Series.")
        
        if not (isinstance(p_range, tuple) and len(p_range) == 2 and all(isinstance(x, int) and x >= 0 for x in p_range)):
            raise TypeError("p_range must be a tuple of two non-negative integers.")
        
        if not isinstance(d, int) or d < 0:
            raise TypeError("d must be a non-negative integer.")
        
        if not (isinstance(q_range, tuple) and len(q_range) == 2 and all(isinstance(x, int) and x >= 0 for x in q_range)):
            raise TypeError("q_range must be a tuple of two non-negative integers.")

        best_aic = float("inf")
        best_bic = float("inf")
        best_order_aic = None
        best_order_bic = None
        results_list = []

        for p, q in itertools.product(range(*p_range), range(*q_range)):
            try:
                model = ARIMA(time_series, order=(p, d, q))
                results = model.fit()
                
                aic_value = results.aic
                bic_value = results.bic
                
                results_list.append({'p': p, 'q': q, 'AIC': aic_value, 'BIC': bic_value})
                
                if aic_value < best_aic:
                    best_aic = aic_value
                    best_order_aic = (p, q)
                
                if bic_value < best_bic:
                    best_bic = bic_value
                    best_order_bic = (p, q)
                    
            except Exception:
                continue  

        results_df = pd.DataFrame(results_list)

        if results_df.empty:
            raise ValueError("No ARIMA models could be estimated. Check the time series data.")

        results_df = results_df.sort_values(by="AIC").reset_index(drop=True)

        best_p_aic, best_q_aic = best_order_aic
        best_p_bic, best_q_bic = best_order_bic

        logger.info(f"According to AIC, the best ARIMA order is ARIMA({best_p_aic}, {d}, {best_q_aic}) with an AIC of {best_aic:.2f}.")
        logger.info(f"According to BIC, the best ARIMA order is ARIMA({best_p_bic}, {d}, {best_q_bic}) with a BIC of {best_bic:.2f}.")

        return results_df

    def fit_arima_model(
        self,
        order: tuple = (1, 0, 1),
        seasonal_order: tuple = (0, 0, 0, 0),
        trend: str = None
    ):
        """
        Fit an ARIMA (or ARMA if d=0) model to the specified column in the DataFrame.

        Parameters
        ----------
        order : tuple, optional
            The (p, d, q) order of the ARIMA model. Default is (1, 0, 1).
        seasonal_order : tuple, optional
            The seasonal (P, D, Q, s) order of the seasonal ARIMA model (SARIMAX).
            Default is (0, 0, 0, 0), i.e., no seasonality.
        trend : str, optional
            The trend parameter (e.g., 'c' for constant, 't' for linear trend).
            If None, no explicit trend is included.

        Returns
        -------
        results : statsmodels.tsa.arima.model.ARIMAResults or None
            The fitted ARIMA model results object if successful; otherwise, None.
        """
        try:
            if self.quantity_col not in self.df.columns:
                raise ValueError(f"Column '{self.quantity_col}' not found in the DataFrame.")

            series = self.df[self.quantity_col].dropna()
            if series.empty:
                raise ValueError(f"No valid data in column '{self.quantity_col}' after dropping NaNs.")

            # If seasonal_order == (0, 0, 0, 0), it's a non-seasonal ARIMA/ARMA.
            model = sm.tsa.statespace.SARIMAX(
                series,
                order=order,
                seasonal_order=seasonal_order,
                trend=trend,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            results = model.fit()

            logger.info(f"ARIMA model fitted successfully with order={order}, seasonal_order={seasonal_order}, trend={trend}")
            logger.info(f"Model Summary:\n{results.summary()}")

            return results

        except Exception as e:
            logger.error(f"Failed to fit ARIMA model for '{self.quantity_col}': {str(e)}", exc_info=True)
            return None

    def check_residuals(
        self,
        fitted_model: ARIMAResults,
        lags: int = 20,
        alpha: float = 0.05
    ) -> dict:
        """
        Perform diagnostic checks on residuals of a fitted ARMA/ARIMA model.

        The function tests:
        1. Serial correlation using the Ljung-Box test over multiple lags (1 through `lags`).
        2. Non-normality using the Shapiro-Wilk test.
        3. Heteroskedasticity using the Breusch-Pagan test.
        4. Additional Durbin-Watson statistic for autocorrelation.

        Parameters
        ----------
        fitted_model : statsmodels.tsa.arima.model.ARIMAResults
            The fitted ARMA/ARIMA model from which residuals are extracted.
        lags : int, optional
            The maximum lag to consider for the Ljung-Box test (testing lags=1..lags). Default is 20.
        alpha : float, optional
            The significance level for the tests. Default is 0.05.

        Returns
        -------
        dict
            A dictionary with results of the residual diagnostics. Keys include:

            - 'ljung_box_pass': bool
                True if Ljung-Box test p-values exceed alpha for *all* tested lags.
            - 'ljung_box_pvalues': pd.Series or list
                The series of p-values from the Ljung-Box test for each lag.
            - 'normality_pass': bool
                True if the Shapiro-Wilk test p-value is above alpha.
            - 'shapiro_pvalue': float
                The p-value for the Shapiro-Wilk test for normality.
            - 'homoskedasticity_pass': bool
                True if the Breusch-Pagan test p-value is above alpha (no heteroskedasticity).
            - 'breusch_pagan_pvalue': float
                The p-value for the Breusch-Pagan test.
            - 'durbin_watson': float
                The Durbin-Watson statistic (values near 2.0 suggest no autocorrelation).
        """
        try:
            residuals = fitted_model.resid

            # ---------------------------------------------------
            # (1) Ljung-Box test for serial correlation
            #     Test multiple lags from 1 to `lags`.
            # ---------------------------------------------------
            ljung_box_results = acorr_ljungbox(
                residuals,
                lags=list(range(1, lags + 1)),
                return_df=True
            )
            # We say there's "no autocorrelation" only if
            # all p-values are above alpha
            ljung_box_pvalues = ljung_box_results["lb_pvalue"]
            ljung_box_pass = (ljung_box_pvalues > alpha).all()

            # ---------------------------------------------------
            # (2) Shapiro-Wilk test for normality
            # ---------------------------------------------------
            stat_shapiro, pvalue_shapiro = shapiro(residuals)
            normality_pass = pvalue_shapiro > alpha

            # ---------------------------------------------------
            # (3) Breusch-Pagan test for heteroskedasticity
            #     If model.exog is None, create a column of ones.
            # ---------------------------------------------------
            exog = fitted_model.model.exog
            if exog is None:
                exog = np.ones((len(residuals), 1))

            # Make sure exog has a constant term
            exog = sm.add_constant(exog, has_constant="add")

            bp_test_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, exog)
            homoskedasticity_pass = bp_pvalue > alpha

            # ---------------------------------------------------
            # (4) Durbin-Watson statistic
            #     Another measure of autocorrelation
            # ---------------------------------------------------
            dw_stat = sm.stats.durbin_watson(residuals)

            results = {
                "ljung_box_pass": ljung_box_pass,
                "ljung_box_pvalues": ljung_box_pvalues.tolist(),
                "normality_pass": normality_pass,
                "shapiro_pvalue": float(pvalue_shapiro),
                "homoskedasticity_pass": homoskedasticity_pass,
                "breusch_pagan_pvalue": float(bp_pvalue),
                "durbin_watson": float(dw_stat)
            }

            logger.info(
                "Residual Diagnostic Results:\n"
                f" - Ljung-Box pass (no autocorr at lags=1..{lags}): {ljung_box_pass} "
                f"(p-values = {ljung_box_pvalues.tolist()})\n"
                f" - Shapiro-Wilk pass (normality): {normality_pass} (p={pvalue_shapiro:.5f})\n"
                f" - Breusch-Pagan pass (homoskedastic): {homoskedasticity_pass} (p={bp_pvalue:.5f})\n"
                f" - Durbin-Watson stat: {dw_stat:.3f}"
            )

            return results

        except Exception as e:
            logger.error(f"Failed to check residual diagnostics: {str(e)}", exc_info=True)
            return {}


