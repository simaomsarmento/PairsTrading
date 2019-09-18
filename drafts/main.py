import pandas as pd
import numpy as np
import json
import sys
import pickle
from classes import class_Trader, class_ForecastingTrader, class_DataProcessor, class_SeriesAnalyser

# just set the seed for the random number generator
np.random.seed(107)

if __name__ == "__main__":

    # read inout parameters
    config_path = sys.argv[1]
    pairs_mode = int(sys.argv[2])
    trade_mode = int(sys.argv[3]) # 1. Benchmark 2.ML 3.Both
    with open(config_path, 'r') as f:
        config = json.load(f)

    ###################################################################################################################
    # 1. Upload Dataset
    # This code assumes the data preprocessing has been done previously by running the notebook:
    # - PairsTrading-DataPreprocessing.ipynb
    # Therefore, we simply retrieve the data from a pickle file and select the dates to study
    ###################################################################################################################

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

    ###################################################################################################################
    # 2. Pairs Filtering & Selection
    # As this part is very visual, the pairs filtering and selection can be obtained by running the notebook:
    # - 'PairsTrading-Clustering.ipynb'
    # This section uploads the pairs for each scenario
    ###################################################################################################################
    # initialize series analyser
    series_analyser = class_SeriesAnalyser.SeriesAnalyser()

    if pairs_mode == 1:
        with open('data/etfs/pickle/pairs_unfiltered.pickle', 'rb') as handle:
            pairs = pickle.load(handle)
    elif pairs_mode == 2:
        with open('data/etfs/pickle/pairs_category.pickle', 'rb') as handle:
            pairs = pickle.load(handle)
    elif pairs_mode == 3:
        with open('data/etfs/pickle/pairs_unsupervised_learning.pickle', 'rb') as handle:
            pairs = pickle.load(handle)

    ###################################################################################################################
    # 3. Apply trading
    # First apply the strategy to the training data, to discard the pairs that were not profitable not even in the
    # training period.
    # Secondly, apply the strategy on the test set
    ###################################################################################################################
    trader = class_Trader.Trader()

    # obtain trading strategy
    trading_strategy = config['trading']['strategy']

    # obtain trading filter info
    if config['trading_filter']['active'] == 1:
        trading_filter = config['trading_filter']
    else:
        trading_filter = None

    # ################################################ BENCHMARK #######################################################
    if (trade_mode == 1) or (trade_mode == 3):
        # Run on TRAIN SET
        if 'bollinger' in trading_strategy:
            sharpe_results, cum_returns, performance = \
                trader.apply_bollinger_strategy(pairs=pairs,
                                                lookback_multiplier=config['trading']['lookback_multiplier'],
                                                entry_multiplier=config['trading']['entry_multiplier'],
                                                exit_multiplier=config['trading']['exit_multiplier'],
                                                trading_filter=trading_filter,
                                                test_mode=False
                                                )
        elif 'kalman' in trading_strategy:
            sharpe_results, cum_returns, performance = \
                trader.apply_kalman_strategy(pairs,
                                             entry_multiplier=config['trading']['entry_multiplier'],
                                             exit_multiplier=config['trading']['exit_multiplier'],
                                             trading_filter=trading_filter,
                                             test_mode=False
                                             )
        else:
            print('Please insert valid trading strategy: 1. "bollinger" or 2."kalman"')
            exit()

        # get train metrics
        n_years_train = round(len(df_prices_train) / 240)
        train_metrics = trader.calculate_metrics(sharpe_results, cum_returns, n_years_train)

        # filter pairs with positive results
        profitable_pairs = trader.filter_profitable_pairs(sharpe_results=sharpe_results, pairs=pairs)

        # Run on TEST SET
        if 'bollinger' in trading_strategy:
            sharpe_results, cum_returns, performance = \
                trader.apply_bollinger_strategy(pairs=profitable_pairs,
                                                lookback_multiplier=config['trading']['lookback_multiplier'],
                                                entry_multiplier=config['trading']['entry_multiplier'],
                                                exit_multiplier=config['trading']['exit_multiplier'],
                                                trading_filter=trading_filter,
                                                test_mode=True
                                                )
            print('Avg sharpe Ratio using Bollinger in test set: ', np.mean(sharpe_results))

        elif 'kalman' in trading_strategy:
            sharpe_results, cum_returns, performance = \
                trader.apply_kalman_strategy(pairs=profitable_pairs,
                                             entry_multiplier=config['trading']['entry_multiplier'],
                                             exit_multiplier=config['trading']['exit_multiplier'],
                                             trading_filter=trading_filter,
                                             test_mode=True
                                             )
            print('Avg sharpe Ratio using kalman in the test set: ', np.mean(sharpe_results))

    # ################################################ ML BASED #######################################################
    if (trade_mode == 2) or (trade_mode == 3):

        forecasting_trader = class_ForecastingTrader.ForecastingTrader()

        # 1) get pairs spreads and train models
        mlp_config = config['mlp']
        mlp_config['train_val_split'] = int(config['mlp']['train_val_split']*len(pairs[0][2]['spread']))
        models = forecasting_trader.train_models(pairs[:2], model_config=mlp_config) # CHANGE LIMITATION OF PAIRS

        # 2) test models on training set and only keep profitable spreads
        print('Still under construction')
        exit()

        # 3) test spreads on test set

    ###################################################################################################################
    # 4. Get results
    # Obtain the results in the test set.
    # - writes global pairs results in an excel file
    # - stores dataframe with info regarding every pair in pickle file
    ###################################################################################################################
    with open(config['dataset']['ticker_segment_dict'], 'rb') as handle:
        ticker_segment_dict = pickle.load(handle)

    results, pairs_summary = trader.summarize_results(sharpe_results, cum_returns, performance, profitable_pairs,
                                                      ticker_segment_dict)

