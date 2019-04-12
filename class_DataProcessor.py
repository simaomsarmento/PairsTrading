import numpy as np
import pandas as pd

# Import Datetime and the Pandas DataReader
from datetime import datetime
from pandas_datareader import data, wb

# just set the seed for the random number generator
np.random.seed(107)

class DataProcessor:
    """
    This class contains a set of pairs trading strategies along
    with some auxiliary functions

    """
    def __init__(self, path):
        """
        :initial elements
        """
        self.path = path

    def read_ticker_excel(self, ticker_attribute):
        """
        Assumes the relevant tickers are saved in an excel file.

        :param ticker_attribute: str corresponding to ticker column
        :return: df with tickers data, list with tickers
        """

        df = pd.read_excel(self.path)

        # remove duplicated
        unique_df = df[~df.duplicated(subset=['Ticker'], keep='first')].sort_values(['Ticker'])
        tickers = unique_df.Ticker.unique()

        return df, unique_df, tickers

    def read_tickers_prices(self, tickers, initial_date, final_date, data_source):
        """
        This function reads the price series for the requested tickers

        :param tickers: list with tickers from which to retrieve prices
        :param initial_date: start date to retrieve price series
        :param final_date: end point
        :param data_source: data source from where to retrieve data

        :return: dictionary with price series for each ticker
        """
        error_counter = 0
        dataset = {key: None for key in tickers}
        for ticker in tickers:
            try:
                df = data.DataReader(ticker, data_source, initial_date, final_date)
                series = df['Adj Close']
                series.name = ticker  # filter close price only
                dataset[ticker] = series.copy()
            except:
                error_counter = error_counter + 1
                print('Not Possible to retrieve information for ' + ticker)

        print('\nUnable to download ' + str(error_counter / len(tickers) * 100) + '% of the ETFs')

        return dataset

    def dict_to_df(self, dataset, threshold):
        """
        Transforms a dictionary into a Dataframe

        :param dataset: dictionary containing tickers as keys and corresponding price series
        :param threshold: threshold for number of Nan Values
        :return: df with tickers as columns
        :return: df_clean with tickers as columns, and columns with null values dropped
        """

        first_count = True
        for k in dataset.keys():
            if dataset[k] is not None:
                if first_count:
                    df = dataset[k]
                    first_count = False
                else:
                    df = pd.concat([df, dataset[k]], axis=1)

        df_clean = self.remove_tickers_with_nan(df, threshold)

        return df, df_clean

    def remove_tickers_with_nan(self, df, threshold):
        """
        Removes columns with more than threshold null values
        """
        null_values = df.isnull().sum()
        null_values = null_values[null_values > 0]

        to_remove = list(null_values[null_values > threshold].index)
        df = df.drop(columns=to_remove)

        print('From now on, we are only considering ' + str(df.shape[1]) + ' ETFs')

        return df

    def get_return_series(self, df_prices):
        """
        This function calculates the return series of a given price series

        :param prices: time series with prices
        :return: return series
        """
        df_returns = df_prices.pct_change()
        df_returns = df_returns.iloc[1:]

        return df_returns

