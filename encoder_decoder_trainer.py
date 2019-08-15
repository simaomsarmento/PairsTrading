import class_ForecastingTrader
import class_DataProcessor
import numpy as np
np.random.seed(1) # NumPy
import random
random.seed(3) # Python
import tensorflow as tf
tf.set_random_seed(2) # Tensorflow
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
from keras import backend as K
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import pandas as pd
import pickle
import gc

forecasting_trader = class_ForecastingTrader.ForecastingTrader()
data_processor = class_DataProcessor.DataProcessor()

################################# READ PRICES AND PAIRS #################################
# read prices
df_prices = pd.read_pickle('/content/drive/PairsTrading/2009-2019/commodity_ETFs_intraday_interpolated_screened_no_outliers.pickle')
# split data in training and test
df_prices_train, df_prices_test = data_processor.split_data(df_prices,
                                                            ('01-01-2009',
                                                             '31-12-2017'),
                                                            ('01-01-2018',
                                                             '31-12-2018'),
                                                            remove_nan=True)
# load pairs
with open('/content/drive/PairsTrading/2009-2019/pairs_unsupervised_learning_optical_intraday.pickle', 'rb') as handle:
    pairs = pickle.load(handle)
n_years_train = round(len(df_prices_train) / (240 * 78))
print('Loaded {} pairs!'.format(len(pairs)))

################################# TRAIN MODELS #################################

combinations = [(24, [20, 20]), (24, [30, 30]), (24, [50, 50])]
hidden_nodes_names = ['20_20', '30_30', '50_50']

for i, configuration in enumerate(combinations):

    model_config = {"n_in": configuration[0],
                    "n_out": 2,
                    "epochs": 500,
                    "hidden_nodes": configuration[1],
                    "loss_fct": "mse",
                    "optimizer": "rmsprop",
                    "batch_size": 512,
                    "train_val_split": '2017-01-01',
                    "test_init": '2018-01-01'}
    models = forecasting_trader.train_models(pairs, model_config, model_type='encoder_decoder')

    # save models for this configuration
    with open('/content/drive/PairsTrading/encoder_decoder/models_n_in-' + str(configuration[0]) + '_hidden_nodes-' +
              hidden_nodes_names[i] + '.pkl', 'wb') as f:
        pickle.dump(models, f)

gc.collect()