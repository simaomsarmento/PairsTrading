"""
This script does the following:
    1- Uploads relevant price series into a Dataframe (import class_DataReader)
    2- Performs dimensionality reduction and unsupervised learning on the time series (import class_SeriesAnalyser)
    3- Implements a pairs trading strategy (import class_Trader)
"""


import numpy as np
import json
import class_SeriesAnalyser, class_Trader, class_DataProcessor

# just set the seed for the random number generator
np.random.seed(107)

if __name__ == "__main__":

    # create json file
    dataset = {'path': 'data/etfs/commodity_ETFs.xlsx',
               'ticker_attribute': 'Ticker',
               'initial_date': '01-06-2017',
               'final_date': '01-01-2018',
               'data_source': 'yahoo',
               'nan_threshold': 0
               }
    PCA = {'N_COMPONENTS': 14}
    config = {'dataset': dataset, 'PCA': PCA}
    with open('config.json', 'w') as fp:
        json.dump(config, fp, indent=4)

    # 1. UPLOAD DATASET
    # initialize data processor
    data_processor = class_DataProcessor.DataProcessor(path=config['dataset']['path'])

    # read tickers
    df_tickers, tickers = data_processor.read_ticker_excel(ticker_attribute=config['dataset']['ticker_attribute'])
    ######WARNING
    tickers = tickers[:25]

    # get price series for tickers
    ticker_prices = data_processor.read_tickers_prices(tickers,
                                                       initial_date=config['dataset']['initial_date'],
                                                       final_date=config['dataset']['final_date'],
                                                       data_source=config['dataset']['data_source']
                                                       )
    df_prices = data_processor.dict_to_df(ticker_prices, config['dataset']['nan_threshold'])

    # get return series
    df_returns = df_prices.pct_change()
    df_returns = df_returns.iloc[1:]

    # 2. Apply PCA and clustering
    series_analyser = class_SeriesAnalyser.SeriesAnalyser()
    X, explained_variance = series_analyser.apply_PCA(config['PCA']['N_COMPONENTS'], df_returns)





