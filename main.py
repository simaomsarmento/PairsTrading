"""
This script does the following:
    1- Uploads relevant price series into a Dataframe (import class_DataReader)
    2- Performs dimensionality reduction and unsupervised learning on the time series (import class_SeriesAnalyser)
    3- Finds good candidates
    4- Implements a pairs trading strategy (import class_Trader)
"""


import numpy as np
import json
import class_SeriesAnalyser, class_Trader, class_DataProcessor

# just set the seed for the random number generator
np.random.seed(107)

if __name__ == "__main__":

    # Read configuration file
    with open('config.json', 'r') as f:
        config = json.load(f)

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
    try:
        range_n_components = config['PCA']['N_COMPONENTS'] # vaidates tuple input
        X, clustered_series_all, clustered_series, counts, clf = \
            series_analyser.clustering_for_optimal_PCA(range_n_components[0], range_n_components[1],
                                                       df_returns, config['clustering'])
    except:
        # PCA
        X, explained_variance = series_analyser.apply_PCA(config['PCA']['N_COMPONENTS'], df_returns)
        # Clustering
        clustered_series_all, clustered_series, counts, clf = \
            series_analyser.apply_DBSCAN(config['clustering']['epsilon'], config['clustering']['min_samples'],
                                         X, df_returns)

    # 3. Find good candidate pairs
    pairs, unique_tickers = series_analyser.get_candidate_pairs(clustered_series=clustered_series,
                                                                pricing_df=df_prices,
                                                                n_clusters=len(counts),
                                                                min_half_life=config['pair_restrictions']['min_half_life'],
                                                                min_zero_crosings=config['pair_restrictions']['min_zero_crossings'],
                                                                p_value_threshold=config['pair_restrictions']['p_value_threshold'],
                                                                hurst_threshold=config['pair_restrictions']['hurst_threshold']
                                                                )

    # 4. Apply trading strategy
    trader = class_Trader.Trader()

    # obtain trading strategy
    trading_strategy = config['trading']['strategy']

    # obtain trading filter info
    if config['trading_filter']['active'] == 1:
        trading_filter = config['trading_filter']
    else:
        trading_filter = None

    if 'bollinger' in trading_strategy:
        sharpe_results_bollinger, cum_returns_bollinger, performance = trader.apply_bollinger_strategy(
                                                                                                pairs=pairs,
                                                                                                lookback_multiplier=config['trading']['lookback_multiplier'],
                                                                                                entry_multiplier=config['trading']['entry_multiplier'],
                                                                                                exit_multiplier=config['trading']['exit_multiplier'],
                                                                                                trading_filter=trading_filter
                                                                                                )
    if 'kalman' in trading_strategy:
        sharpe_results_kalman, cum_returns_kalman = trader.apply_kalman_strategy(pairs,
                                                                                 entry_multiplier=config['trading']['entry_multiplier'],
                                                                                 exit_multiplier=config['trading']['exit_multiplier']
                                                                                 )




