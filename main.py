"""
This script does the following:
    1- Uploads relevant price series into a Dataframe (import class_DataReader)
    2- Performs dimensionality reduction and unsupervised learning on the time series (import class_SeriesAnalyser)
    3- Finds good candidates
    4- Implements a pairs trading strategy (import class_Trader)
"""

import pandas as pd
import numpy as np
import json
import sys
import class_SeriesAnalyser, class_Trader, class_DataProcessor

# just set the seed for the random number generator
np.random.seed(107)

if __name__ == "__main__":

    config_path = sys.argv[1]
    # Read configuration file
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 1. UPLOAD DATASET
    # initialize data processor
    data_processor = class_DataProcessor.DataProcessor(path=config['dataset']['path'])

    # get price series for tickers. First sees if df is already stored in pkl file
    dataset_name = config['dataset']['path'].replace("data/etfs/", "").replace(".xlsx", "")
    dataset_name = dataset_name + '_' + config['dataset']['training_initial_date'] + '_' + config['dataset']['testing_final_date']
    try:
        # try to retrieve from pickle if repeated file
        df_prices = pd.read_pickle('data/etfs/pickle/'+dataset_name)
    except:
        # read from original data source and save in pickle file
        _, df_tickers, tickers = data_processor.read_ticker_excel(
            ticker_attribute=config['dataset']['ticker_attribute'])
        # obtain prices
        ticker_prices_dict = data_processor.read_tickers_prices(tickers=tickers,
                                                       initial_date=config['dataset']['training_initial_date'],
                                                       final_date=config['dataset']['testing_final_date'],
                                                       data_source=config['dataset']['data_source']
                                                       )
        _, df_prices = data_processor.dict_to_df(ticker_prices_dict, config['dataset']['nan_threshold'])
        # save in pickle file
        df_prices.to_pickle('data/etfs/pickle/'+dataset_name)

    # get return series
    df_returns = data_processor.get_return_series(df_prices)

    # 2. APPLY PCA and CLUSTERING
    series_analyser = class_SeriesAnalyser.SeriesAnalyser()
    try:
        # validates list input
        range_n_components = config['PCA']['N_COMPONENTS']
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

    # 3. FIND GOOD CANDIDATE PAIRS
    #
    #
    #
    #CHANGE PRICING_DF_TEST
    #
    #
    #
    #
    pairs, unique_tickers = series_analyser.get_candidate_pairs(clustered_series=clustered_series,
                                                                pricing_df_train=df_prices,
                                                                pricing_df_test=df_prices,
                                                                n_clusters=len(counts),
                                                                min_half_life=config['pair_restrictions']['min_half_life'],
                                                                min_zero_crosings=config['pair_restrictions']['min_zero_crossings'],
                                                                p_value_threshold=config['pair_restrictions']['p_value_threshold'],
                                                                hurst_threshold=config['pair_restrictions']['hurst_threshold']
                                                                )

    # 4. APPLY TRADING STRATEGY
    trader = class_Trader.Trader()

    # obtain trading strategy
    trading_strategy = config['trading']['strategy']

    # obtain trading filter info
    if config['trading_filter']['active'] == 1:
        trading_filter = config['trading_filter']
    else:
        trading_filter = None

    if 'bollinger' in trading_strategy:
        sharpe_results, cum_returns, performance = trader.apply_bollinger_strategy(pairs=pairs,
                                                                                   lookback_multiplier=config['trading']['lookback_multiplier'],
                                                                                   entry_multiplier=config['trading']['entry_multiplier'],
                                                                                   exit_multiplier=config['trading']['exit_multiplier'],
                                                                                   trading_filter=trading_filter,
                                                                                   test_mode=False
                                                                                   )
        print('Avg sharpe Ratio using Bollinger: ', np.mean(sharpe_results))

    elif 'kalman' in trading_strategy:
        sharpe_results, cum_returns, performance = trader.apply_kalman_strategy(pairs,
                                                                                entry_multiplier=config['trading']['entry_multiplier'],
                                                                                exit_multiplier=config['trading']['exit_multiplier'],
                                                                                trading_filter=trading_filter,
                                                                                test_mode=False
                                                                                )
        print('Avg sharpe Ratio using kalman: ', np.mean(sharpe_results))
    else:
        print('Please insert valid trading strategy: 1. "bollinger" or 2."kalman"')
        exit()

    # get results
    results, pairs_summary = trader.summarize_results(sharpe_results, cum_returns, performance, pairs)

    # 5. DUMP RESULTS
    # - writes global pairs results in an excel file
    # - stores dataframe with info regarding every pair in pickle file
    data_processor.dump_results(dataset=config['dataset'], pca=config['PCA'], clustering=config['clustering'],
                                pair_restrictions=config['pair_restrictions'], trading=config['trading'],
                                trading_filter=config['trading_filter'], results=results,
                                pairs_summary_df=pairs_summary, filename=config['output']['filename'])
