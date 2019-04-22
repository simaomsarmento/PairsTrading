import numpy as np
import pandas as pd
from openpyxl import load_workbook

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

    def append_df_to_excel(self, filename, df, sheet_name='Sheet1', startrow=None,
                           truncate_sheet=False,
                           **to_excel_kwargs):
        """
        Source: https://stackoverflow.com/questions/20219254/how-to-write-to-an-existing-excel-file-without-overwriting
        -data-using-pandas/47740262#47740262

        Append a DataFrame [df] to existing Excel file [filename]
        into [sheet_name] Sheet.
        If [filename] doesn't exist, then this function will create it.

        Parameters:
          filename : File path or existing ExcelWriter
                     (Example: '/path/to/file.xlsx')
          df : dataframe to save to workbook
          sheet_name : Name of sheet which will contain DataFrame.
                       (default: 'Sheet1')
          startrow : upper left cell row to dump data frame.
                     Per default (startrow=None) calculate the last row
                     in the existing DF and write to the next row...
          truncate_sheet : truncate (remove and recreate) [sheet_name]
                           before writing DataFrame to Excel file
          to_excel_kwargs : arguments which will be passed to `DataFrame.to_excel()`
                            [can be dictionary]

        Returns: None
        """

        # ignore [engine] parameter if it was passed
        if 'engine' in to_excel_kwargs:
            to_excel_kwargs.pop('engine')

        writer = pd.ExcelWriter(filename, engine='openpyxl')

        # Python 2.x: define [FileNotFoundError] exception if it doesn't exist
        #try:
        #    FileNotFoundError
        #except NameError:
        #    FileNotFoundError = IOError

        try:
            # try to open an existing workbook
            writer.book = load_workbook(filename)

            # get the last row in the existing Excel sheet
            # if it was not specified explicitly
            if startrow is None and sheet_name in writer.book.sheetnames:
                startrow = writer.book[sheet_name].max_row

            # truncate sheet
            if truncate_sheet and sheet_name in writer.book.sheetnames:
                # index of [sheet_name] sheet
                idx = writer.book.sheetnames.index(sheet_name)
                # remove [sheet_name]
                writer.book.remove(writer.book.worksheets[idx])
                # create an empty sheet [sheet_name] using old index
                writer.book.create_sheet(sheet_name, idx)

            # copy existing sheets
            writer.sheets = {ws.title: ws for ws in writer.book.worksheets}
        except FileNotFoundError:
            # file does not exist yet, we will create it
            pass

        if startrow is None:
            startrow = 0

        # write out the new sheet
        df.to_excel(writer, sheet_name, startrow=startrow, **to_excel_kwargs)

        # save the workbook
        writer.save()

    def dump_results(self, dataset, pca, clustering, pair_restrictions, trading, trading_filter, results, filename):
        """
        This functions appends the results obtained into the file given as input
        :param dataset: dictionary containg information regarding the dataset
        :param pca: dictionary containg information regarding the PCA parameters
        :param clustering: dictionary containg information regarding the clustering configurations
        :param pair_restrictions: dictionary containg information regarding the restritions applied for pairs
        :param trading: dictionary containg information regarding the trading requisites
        :param trading_filter: dictionary containg information regarding the trading filter used
        :param results: dictionary containg information regarding the results obtained
        :param filename: filename of where to write the summary info
        """

        # intialise data of lists.
        summary = {
                   # Dataset
                   "path": [dataset['path']],
                   "training_initial_date": [dataset['initial_date']],
                   "training_final_date": [dataset['final_date']],
                   "testing_initial_date": [dataset['testing_initial_date']],
                   "testing_final_date": [dataset['testing_final_date']],
                   # PCA
                   "n_components_PCA": [pca["N_COMPONENTS"]],
                   # clustering
                   "clustering_algo": [clustering["algo"]],
                   "epsilon": [clustering["epsilon"]],
                   "min_samples": [clustering["min_samples"]],
                   # Pair restrictions
                   "min_half_life": [pair_restrictions["min_half_life"]],
                   "min_zero_crossings": [pair_restrictions["min_zero_crossings"]],
                   "p_value_threshold": [pair_restrictions["p_value_threshold"]],
                   "hurst_threshold": [pair_restrictions["hurst_threshold"]],
                   # Trading
                   "strategy": [trading["strategy"]],
                   "lookback_multiplier": [trading["lookback_multiplier"]],
                   "entry_multiplier": [trading["entry_multiplier"]],
                   "exit_multiplier": [trading["exit_multiplier"]],
                   # Trading filter
                   "active": [trading_filter["active"]],
                   "filter_name": [trading_filter["name"]],
                   "filter_lookback_multiplier": [trading_filter["filter_lookback_multiplier"]],
                   "filter_lag": [trading_filter["lag"]],
                   "filter_diff_threshold": [trading_filter["diff_threshold"]],
                   # Result statistics
                   "n_pairs": [results["n_pairs"]],
                   'avg_sharpe_ratio': [results["avg_sharpe_ratio"]],
                   'avg_ROI': [results["avg_ROI"]],
                   'positive_negative_ratio': [results["positive_negative_ratio"]],
                   'negative_pairs': [results["negative_pairs_percentage"]],
                   'avg_half_life': [results["avg_half_life"]],
                   'avg_hurst_exponent': [results["avg_hurst_exponent"]]
                   }

        # Create DataFrame
        df = pd.DataFrame(summary, index=[datetime.now()])
        df.index.name = 'Date'
        # set order of columns
        cols = ["path", "training_initial_date", "training_final_date", "testing_initial_date", "testing_final_date",
                "n_components_PCA", "clustering_algo", "epsilon", "min_samples",
                "min_half_life", "min_zero_crossings", "p_value_threshold", "hurst_threshold", "strategy",
                "lookback_multiplier", "entry_multiplier", "exit_multiplier", "active", "filter_name",
                "filter_lookback_multiplier", "filter_lag", "filter_diff_threshold", "n_pairs", 'avg_sharpe_ratio',
                'avg_ROI', 'positive_negative_ratio', 'negative_pairs', 'avg_half_life', 'avg_hurst_exponent']
        df = df[cols]

        self.append_df_to_excel(filename=filename, df=df)