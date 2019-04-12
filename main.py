"""
This script does the following:
    1- Uploads relevant price series into a Dataframe (import class_DataReader)
    2- Performs dimensionality reduction and unsupervised learning on the time series (import class_SeriesAnalyser)
    3- Find good candidates
    4- Implements a pairs trading strategy (import class_Trader)
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
    clustering = {
                  'algo': 'DBSCAN',
                  'epsilon': 0.4,
                  'min_samples': 2
                  }
    pair_restrictions = {
                         'min_half_life': 5,
                         'min_zero_crossings': 12,
                         'p_value_threshold': 0.05,
                         'hurst_threshold': 0.5
                         }

    config = {'dataset': dataset,
              'PCA': PCA,
              'clustering': clustering,
              'pair_restrictions': pair_restrictions
              }
    with open('config.json', 'w') as fp:
        json.dump(config, fp, indent=4)

    # Read configuration file
    #with open('config.json', 'r') as f:
    #    config = json.load(f)

    # 1. UPLOAD DATASET
    # initialize data processor
    data_processor = class_DataProcessor.DataProcessor(path=config['dataset']['path'])

    # read tickers
    _, df_tickers, tickers = data_processor.read_ticker_excel(ticker_attribute=config['dataset']['ticker_attribute'])

    # get price series for tickers
    ticker_prices = data_processor.read_tickers_prices(tickers=tickers,
                                                       initial_date=config['dataset']['initial_date'],
                                                       final_date=config['dataset']['final_date'],
                                                       data_source=config['dataset']['data_source']
                                                       )
    _, df_prices = data_processor.dict_to_df(ticker_prices, config['dataset']['nan_threshold'])

    # get return series
    df_returns = data_processor.get_return_series(df_prices)

    # 2. Apply PCA and clustering
    series_analyser = class_SeriesAnalyser.SeriesAnalyser()
    # PCA
    X, explained_variance = series_analyser.apply_PCA(config['PCA']['N_COMPONENTS'], df_returns)
    # Clustering
    clustered_series_all, clustered_series, counts, clf = series_analyser.apply_DBSCAN(config['clustering']['epsilon'],
                                                                                       config['clustering']['min_samples'],
                                                                                       X,
                                                                                       df_returns)

    # 3. Find good candidate pairs
    pairs, unique_tickers = series_analyser.get_candidate_pairs(clustered_series=clustered_series,
                                                                pricing_df=df_prices,
                                                                n_clusters=len(counts),
                                                                min_half_life=config['pair_restrictions']['min_half_life'],
                                                                min_zero_crosings=config['pair_restrictions']['min_zero_crossings'],
                                                                p_value_threshold=config['pair_restrictions']['p_value_threshold'],
                                                                hurst_threshold=config['pair_restrictions']['hurst_threshold']
                                                                )


