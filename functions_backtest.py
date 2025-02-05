import pandas as pd
import datetime as dt
import logging
from binance.client import Client as BinanceClient
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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
    
    def fetch_data(self, symbol: str, interval: str, start_time: dt.datetime, end_time: dt.datetime) -> pd.DataFrame:
        """
        Fetch historical candlestick data from the Binance API and return it as a pandas DataFrame.
        Always instantiate the BinanceDataHandler class to initialize the Binance API client.

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
        pandas.DataFrame
            A DataFrame containing the historical data with columns for timestamp, open, high, low,
            close, volume, and other relevant data. Returns None if the request fails or if no data is retrieved.
        
        Raises
        ------
        Exception
            If the API request fails or returns an error, the exception will be caught and logged.
        """
        date_format = "%Y-%m-%d"
        
        try:
            data = self.client.get_historical_klines(
                symbol, interval, start_time.strftime(date_format), end_time.strftime(date_format)
            )
            
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

            return df

        except Exception as e:
            logger.error(f"Failed to retrieve data for {symbol}: {str(e)}")
            logger.debug("Full exception details", exc_info=True)
            return None
    
    @staticmethod
    def select_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """
        Select and return the OHLCV (Open, High, Low, Close, Volume) columns from the provided DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame that should contain at least the columns 'open', 'high', 'low', 'close', and 'volume'.

        Returns
        -------
        pandas.DataFrame
            A new DataFrame containing only the OHLCV columns.

        Raises
        ------
        ValueError
            If the input DataFrame is missing one or more of the required OHLCV columns.
        """
        required_columns = ["open", "high", "low", "close", "volume"]

        try:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Input DataFrame is missing required columns: {missing_columns}")

            df_ohlcv = df[required_columns].copy()
            return df_ohlcv

        except Exception as e:
            logger.error(f"Failed to select OHLCV columns: {str(e)}")
            logger.debug("Full exception details", exc_info=True)
            return None

    @staticmethod
    def aggregate_data(df: pd.DataFrame, frequency: str = '1H') -> pd.DataFrame:
        """
        Aggregate OHLCV data to a specified frequency using the datetime index.

        Parameters
        ----------
        df : pandas.DataFrame
            A DataFrame containing OHLCV data with a datetime index.
        frequency : str
            A string representing the resampling frequency (e.g., '1H' for hourly, '1D' for daily).

        Returns
        -------
        pandas.DataFrame
            A new DataFrame with aggregated OHLCV data resampled at the specified frequency.
            If the input DataFrame is None or empty, returns the original DataFrame.

        Raises
        ------
        Exception
            If an error occurs during the aggregation, the exception is logged and None is returned.
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
        Remove duplicate rows from the instance's DataFrame and log the number of duplicates removed.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with duplicate rows removed. If the input is None or empty, returns the original DataFrame.

        Raises
        ------
        Exception
            If an error occurs during the removal of duplicates, the exception is logged and None is returned.
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
            The target timezone to convert the index timestamps to. The default is "CET".

        Returns
        -------
        pd.DataFrame
            A new DataFrame with the index timestamps converted to the specified timezone.

        Raises
        ------
        Exception
            If an error occurs during the conversion, the exception is logged and None is returned.
        """
        try:
            df_copy = self.df.copy()

            # If the index is naive (no timezone), assume UTC and convert to target timezone.
            if df_copy.index.tz is None:
                df_copy.index = df_copy.index.tz_localize('UTC').tz_convert(target_timezone)
            else:
                df_copy.index = df_copy.index.tz_convert(target_timezone)

            logger.info(f"Converted DataFrame index to timezone: {target_timezone}")
            return df_copy

        except Exception as e:
            logger.error(f"Failed to convert index timezone: {str(e)}", exc_info=True)
            return None


class DataVisualisation:
    """
    A class for performing various data visualisation tasks.
    Designed to be consistent with the structure of DataHandler and BinanceDataHandler.
    """

    def __init__(self, df=None, time_col=None,
                 open_col='open', high_col='high',
                 low_col='low', close_col='close', volume_col='volume'):
        """
        Initialize the DataVisualisation instance.

        Parameters:
            df (pandas.DataFrame, optional): DataFrame containing the OHLCV data.
                If provided, it can be used as the default data source for plotting methods.
            time_col (str, optional): Name of the column representing time.
                If provided, and if df is not None, the DataFrame index will be set to this column.
            open_col (str): Column name for the open price.
            high_col (str): Column name for the high price.
            low_col (str): Column name for the low price.
            close_col (str): Column name for the close price.
            volume_col (str): Column name for the volume data.
        """
        self.df = df
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col

        # If a time column is specified, set it as the DataFrame index.
        if self.df is not None and time_col is not None and time_col in self.df.columns:
            self.df.set_index(time_col, inplace=True)

    def plot_candlestick_volume(self, df=None, title="Candlestick and Volume Chart",
                                  height=800, show_rangeslider=False,
                                  open_col=None, high_col=None, low_col=None,
                                  close_col=None, volume_col=None, ccy='$'):
        """
        Plots a candlestick chart with volume bars using OHLCV data.

        Parameters:
            df (pandas.DataFrame, optional): DataFrame containing the OHLCV data.
                If not provided, the method will use the DataFrame stored in the instance.
            title (str): Title of the chart.
            height (int): Height of the chart (in pixels).
            show_rangeslider (bool): Whether to display the Plotly rangeslider.
            open_col (str, optional): Column name for the open price.
                If not provided, defaults to the instance's open_col.
            high_col (str, optional): Column name for the high price.
                If not provided, defaults to the instance's high_col.
            low_col (str, optional): Column name for the low price.
                If not provided, defaults to the instance's low_col.
            close_col (str, optional): Column name for the close price.
                If not provided, defaults to the instance's close_col.
            volume_col (str, optional): Column name for the volume data.
                If not provided, defaults to the instance's volume_col.

        Raises:
            ValueError: If no DataFrame is available for plotting.
        """
        if df is None:
            if self.df is None:
                raise ValueError("No data provided. Please pass a DataFrame or initialize the instance with data.")
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
            marker={"color": "rgba(128,128,128,0.5)"},
            name="Volume"
        )

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add the candlestick trace on the secondary y-axis (price)
        fig.add_trace(candlesticks, secondary_y=True)
        # Add the volume bars on the primary y-axis (volume)
        fig.add_trace(volume_bars, secondary_y=False)

        fig.update_layout(
            title=title,
            height=height,
            xaxis=dict(rangeslider=dict(visible=show_rangeslider))
        )
        fig.update_yaxes(title_text=f"Price {ccy}", secondary_y=True, showgrid=True)
        fig.update_yaxes(title_text=f"Volume {ccy}", secondary_y=False, showgrid=False)

        fig.show()
