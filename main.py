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

    # Read configuration file
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 1. UPLOAD DATA SET
    # This code assumes the data preprocessing has been done previously by running the notebook:
    # - PairsTrading_CommodityETFs-DataPreprocessing.ipynb
    # Therefore, we simply retrieve the data from a pickle file and select the dates to study

    # initialize data processor
    data_processor = class_DataProcessor.DataProcessor()

    # Read dataset and select dates
    dataset_path = config['dataset']['path']
    df_prices = pd.read_pickle(dataset_path)

    # split data in training and test
    df_prices_train, df_prices_test = data_processor.split_data(df_prices,
                                                                (config['dataset']['training_initial_date'],
                                                                 config['dataset']['training_final_date']),
                                                                (config['dataset']['testing_initial_date'],
                                                                 config['dataset']['testing_final_date']),
                                                                remove_nan=True)

    # 2. APPLY PCA and CLUSTERING
    # get return series
    df_returns_train = data_processor.get_return_series(df_prices_train)
    # initialize series analyser
    series_analyser = class_SeriesAnalyser.SeriesAnalyser()
    try:
        # validates list input from config file
        range_n_components = config['PCA']['N_COMPONENTS']
        X, clustered_series_all, clustered_series, counts, clf = \
            series_analyser.clustering_for_optimal_PCA(range_n_components[0], range_n_components[1],
                                                       df_returns_train, config['clustering'])
    except:
        # PCA
        X, _ = series_analyser.apply_PCA(config['PCA']['N_COMPONENTS'], df_returns_train)
        # Clustering
        clustered_series_all, clustered_series, counts, clf = \
            series_analyser.apply_DBSCAN(config['clustering']['epsilon'], config['clustering']['min_samples'],
                                         X, df_returns_train)

    # 3. FIND GOOD CANDIDATE PAIRS
    pairs, unique_tickers = series_analyser.get_candidate_pairs(clustered_series=clustered_series,
                                                                pricing_df_train=df_prices_train,
                                                                pricing_df_test=df_prices_test,
                                                                n_clusters=len(counts),
                                                                min_half_life=config['pair_restrictions']['min_half_life'],
                                                                min_zero_crosings=config['pair_restrictions']['min_zero_crossings'],
                                                                p_value_threshold=config['pair_restrictions']['p_value_threshold'],
                                                                hurst_threshold=config['pair_restrictions']['hurst_threshold']
                                                                )

    # 4. APPLY TRADING STRATEGY
    # we first apply the strategy to the training data, to discard the pairs that were not
    # profitable even in the training period
    trader = class_Trader.Trader()

    # obtain trading strategy
    trading_strategy = config['trading']['strategy']

    # obtain trading filter info
    if config['trading_filter']['active'] == 1:
        trading_filter = config['trading_filter']
    else:
        trading_filter = None

    # Run on TRAIN SET
    if 'bollinger' in trading_strategy:
        sharpe_results, cum_returns, performance = trader.apply_bollinger_strategy(pairs=pairs,
                                                                                   lookback_multiplier=config['trading']['lookback_multiplier'],
                                                                                   entry_multiplier=config['trading']['entry_multiplier'],
                                                                                   exit_multiplier=config['trading']['exit_multiplier'],
                                                                                   trading_filter=trading_filter,
                                                                                   test_mode=False
                                                                                   )
    elif 'kalman' in trading_strategy:
        sharpe_results, cum_returns, performance = trader.apply_kalman_strategy(pairs,
                                                                                entry_multiplier=config['trading']['entry_multiplier'],
                                                                                exit_multiplier=config['trading']['exit_multiplier'],
                                                                                trading_filter=trading_filter,
                                                                                test_mode=False
                                                                                )
    else:
        print('Please insert valid trading strategy: 1. "bollinger" or 2."kalman"')
        exit()

    # get ROI for training
    n_years_train = round(len(df_prices_train) / 240)
    print('Train years ', n_years_train)
    train_metrics = trader.calculate_metrics(sharpe_results, cum_returns, n_years_train)

    # filter pairs with positive results
    profitable_pairs = trader.filter_profitable_pairs(sharpe_results=sharpe_results, pairs=pairs)

    # Run on TEST SET
    if 'bollinger' in trading_strategy:
        sharpe_results, cum_returns, performance = trader.apply_bollinger_strategy(pairs=profitable_pairs,
                                                                                   lookback_multiplier=config['trading']['lookback_multiplier'],
                                                                                   entry_multiplier=config['trading']['entry_multiplier'],
                                                                                   exit_multiplier=config['trading']['exit_multiplier'],
                                                                                   trading_filter=trading_filter,
                                                                                   test_mode=True
                                                                                   )
        print('Avg sharpe Ratio using Bollinger in test set: ', np.mean(sharpe_results))

    elif 'kalman' in trading_strategy:
        sharpe_results, cum_returns, performance = trader.apply_kalman_strategy(profitable_pairs,
                                                                                entry_multiplier=config['trading']['entry_multiplier'],
                                                                                exit_multiplier=config['trading']['exit_multiplier'],
                                                                                trading_filter=trading_filter,
                                                                                test_mode=True
                                                                                )
        print('Avg sharpe Ratio using kalman in the test set: ', np.mean(sharpe_results))

    # get results
    results, pairs_summary = trader.summarize_results(sharpe_results, cum_returns, performance, pairs)

    # 5. DUMP RESULTS
    # - writes global pairs results in an excel file
    # - stores dataframe with info regarding every pair in pickle file
    data_processor.dump_results(dataset=config['dataset'], pca=config['PCA'], clustering=config['clustering'],
                                pair_restrictions=config['pair_restrictions'], trading=config['trading'],
                                trading_filter=config['trading_filter'], results=results, train_metrics=train_metrics,
                                pairs_summary_df=pairs_summary, filename=config['output']['filename'])
