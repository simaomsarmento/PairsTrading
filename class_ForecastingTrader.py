# set seeds
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

import matplotlib.pyplot as plt

import class_Trader

from sklearn.preprocessing import MinMaxScaler

# Import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, GRU, Dropout
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.callbacks import EarlyStopping
# just set the seed for the random number generator

import pickle



class ForecastingTrader:
    """


    """
    def __init__(self):
        """
        :initial elements
        """

    def destandardize(self, predictions, spread_mean, spread_std):
        """
        This function transforms the normalized predictions into the original space.
        """
        return predictions * spread_std + spread_mean

    def forecast_spread_trading(self, X, Y, spread_test, spread_train, beta, predictions, lag,
                                low_quantile=0.15, high_quantile=0.85):
        """
        This function will set the trading positions based on the forecasted spread.
        For each day, the function compares the predicted spread for that day with the
        true value of the spread in tha day before, giving the predicted spread pct change.
        In case it is larger than the threshold, a position is entered.
        Note: because a position entered in day n it is only accounted for on the day after,
        we shift the entered positions.
        : predictions: predictions should not be standardized, but with regular mean and variance.
        """
        # 1. Get predictions pct_change
        # we want to see the pct change of the prediction compared
        # to the true value but the previous instant time, because we
        # are interested in seeing the temporal % change
        predictions_pct_change = (((predictions - spread_test.shift(lag)) / abs(spread_test.shift(lag))) * 100).fillna(
            0)

        # 2. Calculate trading thresholds
        spread_train_pct_change = ((spread_train - spread_train.shift(lag)) / abs(spread_train.shift(lag))) * 100
        positive_changes = spread_train_pct_change[spread_train_pct_change > 0]
        negative_changes = spread_train_pct_change[spread_train_pct_change < 0]
        long_threshold = max(positive_changes.quantile(q=high_quantile, interpolation='linear'), 5)
        #long_threshold = min(long_threshold, 50)
        print('Long threshold: {:.2f}'.format(long_threshold))
        short_threshold = min(negative_changes.quantile(q=low_quantile, interpolation='linear'), -5)
        #short_threshold = max(short_threshold, -50)
        print('Short threshold: {:.2f}'.format(short_threshold))

        # 3. Define trading timings
        # Note: If we want to enter a position at the beginning of day N,
        # because of the way pnl is calculated the position is entered
        # in the previous day.
        # Example: In day 23 the percentage change is 55% (wrt day 22). If we were to enter the
        # position in day 23, the following code would not consider the gains during day 23, even if we had
        # set the position in the morning. (it conly considers the gains for the next days)
        # Thus, as a workaoround, we enter the position at day 22 (at night), and it considers the gains for day 23
        numUnits = pd.Series(data=[0.] * len(spread_test), index=spread_test.index, name='numUnits')
        longsEntry = predictions_pct_change > long_threshold
        longsEntry = longsEntry.shift(-1).fillna(False)
        numUnits[longsEntry] = 1.
        shortsEntry = predictions_pct_change < short_threshold
        shortsEntry = shortsEntry.shift(-1).fillna(False)
        numUnits[shortsEntry] = -1

        # ffill if applicable
        if lag == 1:
            pct_change_from_previous = predictions_pct_change
        else:
            pct_change_from_previous = predictions_pct_change = (((predictions - spread_test.shift(1)) /
                                                                  abs(spread_test.shift(1))) * 100).fillna(0)
        for i in range(1, len(numUnits) - 1):
            if numUnits[i] != 0:
                continue
            else:
                if numUnits[i - 1] == 0:
                    continue
                elif numUnits[i - 1] == 1.:
                    if pct_change_from_previous[i + 1] > 0:
                        numUnits[i] = 1
                        continue
                elif numUnits[i - 1] == -1.:
                    if pct_change_from_previous[i + 1] < 0:
                        numUnits[i] = -1.
                        continue

        # 4. Calculate P&L and Returns
        trader = class_Trader.Trader()

        # concatenate for positions with not enough data to be predicted
        lookback = len(Y)-len(spread_test)
        numUnits_not_predicted = pd.Series(data=[0.] * lookback, index=Y.index[:lookback])
        numUnits = pd.concat([numUnits_not_predicted, numUnits], axis=0)
        numUnits.name = 'numUnits'
        # add trade durations
        numUnits_df = pd.DataFrame(numUnits, index=Y.index)
        numUnits_df = numUnits_df.rename(columns={"positions": "numUnits"})
        trading_durations = trader.add_trading_duration(numUnits_df)
        # calculate balance
        balance_summary = trader.calculate_balance(Y, X, beta, numUnits.shift(1).fillna(0), trading_durations)

        # summarize
        ret_with_costs, cum_ret_with_costs = balance_summary.daily_return, (balance_summary.account_balance-1)
        bins = [-np.inf, -0.00000001, 0.00000001, np.inf]
        names = ['-1', '0', '1']
        summary = pd.DataFrame(data={'prediction(t)': predictions.values,
                                     'spread(t)': spread_test.values,
                                     'predicted_change(%)': predictions_pct_change,
                                     'position_during_day': numUnits.shift(1).fillna(0).values[lookback:],
                                     'trading_days': trading_durations[lookback:],
                                     'ret_with_costs': ret_with_costs[lookback:],
                                     'predicted_direction': pd.cut(predictions_pct_change, bins, labels=names),
                                     'true_direction': pd.cut(spread_test.diff(), bins, labels=names)
                                     },
                               index=spread_test.index)

        return ret_with_costs, cum_ret_with_costs, summary

    def series_to_supervised(self, data, index=None, n_in=1, n_out=1, dropnan=True):
        """
        Frame a time series as a supervised learning dataset.
        Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
        Returns:
            Pandas DataFrame of series framed for supervised learning.
        """
        n_vars = 1 if type(data) is list else data.shape[1]
        if index is None:
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame(data, index=index)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def plot_loss(self, history):
        """
        Function to plot loss function.
        Arguments:
        history: History object with data from training.
        title: Plot title.
        """

        plt.plot(history.history['loss'], label = "training")
        plt.plot(history.history['val_loss'], label = "validation")

    def prepare_train_data(self, spread, model_config):
        """

        :param spread: spread of the pair being considered
        :param model_config: dictionary with model parameters
        :return:
            tuple with training data
            tuple with validation data
            y_series in validation period (to compare with predictions later on)
        """
        train_val_split = model_config['train_val_split']

        # save data form original spread
        #standardization_dict = {'mean': spread[:train_val_split].mean(), 'std': np.std(spread[:train_val_split])}
        #spread = (spread - standardization_dict['mean']) / standardization_dict['std']
        scaler = MinMaxScaler(feature_range=(0, 1))
        spread_norm = scaler.fit_transform(spread.values.reshape(spread.shape[0], 1))
        spread_norm = pd.Series(data=spread_norm.flatten(), index=spread.index)
        forecasting_data = self.series_to_supervised(list(spread_norm), spread.index, model_config['n_in'],
                                                     model_config['n_out'], dropnan=True)
        # define dataset
        X_series = forecasting_data.drop(columns='var1(t)')
        y_series = forecasting_data['var1(t)']

        # split
        X_series_train = X_series[:train_val_split]
        X_series_val = X_series[train_val_split:]
        y_series_train = y_series[:train_val_split]
        y_series_val = y_series[train_val_split:]

        X_train = X_series_train.values
        X_val = X_series_val.values
        y_train = y_series_train.values
        y_val = y_series_val.values

        return (X_train, y_train), (X_val, y_val), y_series_val, scaler

    def prepare_test_data(self, spread, model_config, scaler):
        """
        """
        # normalize spread
        #spread = (spread - standardization_dict['mean']) / standardization_dict['std']
        spread_norm = scaler.transform(spread.values.reshape(spread.shape[0], 1))
        spread_norm = pd.Series(data=spread_norm.flatten(), index=spread.index)
        forecasting_data = self.series_to_supervised(list(spread_norm), spread.index, model_config['n_in'],
                                                     model_config['n_out'], dropnan=True)
        # define dataset
        X_series_test = forecasting_data.drop(columns='var1(t)')
        y_series_test = forecasting_data['var1(t)']

        X_test = X_series_test.values
        y_test = y_series_test.values

        return (X_test, y_test)

    def destandardize(self, predictions, spread_mean, spread_std):
        """
        This function transforms the normalized predictions into the original space.
        """
        return predictions * spread_std + spread_mean

    def train_models(self, pairs, model_config, model_type='mlp'):
        """
        This function trains the models for every pair identified.

        :param pairs: list with pairs and corresponding statistics
        :param model_config: dictionary with info for the model
        :return: all models
        """

        models = []
        for pair in pairs:

            # prepare train data
            spread = pair[2]['spread']
            train_data, validation_data, y_series_val, scaler = self.prepare_train_data(spread, model_config)
            # prepare test data
            spread_test = pair[2]['Y_test']-pair[2]['coint_coef']*pair[2]['X_test']
            test_data = self.prepare_test_data(spread_test, model_config, scaler)

            if model_type == 'mlp':
                # train model and get predictions
                model, history, score, predictions_val, predictions_test = self.apply_MLP(X=train_data[0],
                                                                                y=train_data[1],
                                                                                validation_data=validation_data,
                                                                                test_data=test_data,
                                                                                n_in=model_config['n_in'],
                                                                                hidden_nodes=model_config['hidden_nodes'],
                                                                                epochs=model_config['epochs'],
                                                                                optimizer=model_config['optimizer'],
                                                                                loss_fct=model_config['loss_fct'],
                                                                                batch_size=model_config['batch_size'])
                # save keras model
                nodes = model_config['hidden_nodes']
                nodes_name = str(nodes[0]) + '*2_' if len(nodes) > 1 else str(nodes[0])
                model.save('../mlp_models/keras_models/models_n_in-' + str(
                    model_config['n_in']) + '_hidden_nodes-' + nodes_name +
                           '_{}_{}'.format(pair[0], pair[1]) + '.h5')  # creates a HDF5 file 'my_model.h5'
                del model  # deletes the existing model

            elif model_type == 'rnn':
                model, history, score, predictions_val, predictions_test = self.apply_RNN(X=train_data[0],
                                                                             y=train_data[1],
                                                                             validation_data=validation_data,
                                                                             test_data=test_data,
                                                                             hidden_nodes=model_config['hidden_nodes'],
                                                                             epochs=model_config['epochs'],
                                                                             optimizer=model_config['optimizer'],
                                                                             loss_fct=model_config['loss_fct'],
                                                                             batch_size=model_config['batch_size'])
                # save keras model
                nodes = model_config['hidden_nodes']
                nodes_name = str(nodes[0]) + '*2_' if len(nodes) > 1 else str(nodes[0])
                model.save('../rnn_models/keras_models/models_n_in-' + str(
                    model_config['n_in']) + '_hidden_nodes-' + nodes_name +
                           '_{}_{}'.format(pair[0], pair[1]) + '.h5')  # creates a HDF5 file 'my_model.h5'
                del model  # deletes the existing model

            # transform predictions to series
            predictions_val = scaler.inverse_transform(predictions_val)
            predictions_test = scaler.inverse_transform(predictions_test)
            predictions_val = pd.Series(data=predictions_val.flatten(), index=y_series_val.index)
            predictions_test = pd.Series(data=predictions_test.flatten(), index=spread_test[-len(test_data[1]):].index)

            #predictions_val_destandardized = self.destandardize(predictions_val, standardization_dict['mean'],
            #                                                    standardization_dict[ 'std'])
            #predictions_test_destandardized = self.destandardize(predictions_test, standardization_dict['mean'],
            #                                                     standardization_dict['std'])

            # save all info
            # check epochs
            if len(history.history['val_loss']) == 500:
                epoch_stop = 500
            else:
                epoch_stop = len(history.history['val_loss']) - 50 # patience=50

            model_info = {'leg1': pair[0],
                          'leg2': pair[1],
                          'standardization_dict': 'scaler',
                          'history': history.history,
                          'score': score,
                          'epoch_stop': epoch_stop,
                          'predictions_val': predictions_val.copy(),
                          'predictions_test': predictions_test.copy()
                          }
            models.append(model_info)

        # append model configuration on last position
        models.append(model_config)

        return models

    # ################################### MLP ############################################
    def apply_MLP(self, X, y, validation_data, test_data, n_in, hidden_nodes, epochs, optimizer, loss_fct,
                  batch_size=128):
        # reset seed
        # np.random.seed(0) # NumPy
        # tf.set_random_seed(2) # Tensorflow
        # random.seed(3) # Python

        # define validation set
        X_val = validation_data[0]
        y_val = validation_data[1]

        # define test set
        X_test = test_data[0]
        y_test = test_data[1]

        model = Sequential()
        for i in range(len(hidden_nodes)):
            model.add(Dense(hidden_nodes[i], activation='relu', input_dim=n_in))
            # model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer=optimizer, loss=loss_fct, metrics=['mae'])
        model.summary()

        # simple early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        # pb = ProgbarLogger(count_mode='samples', stateful_metrics=None)

        history = model.fit(X, y, epochs=epochs, verbose=2, validation_data=validation_data,
                            shuffle=False, batch_size=batch_size, callbacks=[es]) # , callbacks=[pb, es])

        train_score = model.evaluate(X, y, verbose=0)
        val_score = model.evaluate(X_val, y_val, verbose=0)
        test_score = model.evaluate(X_test, y_test, verbose=0)
        score = {'train': train_score, 'val': val_score, 'test': test_score}

        predictions_validation = model.predict(X_val)
        predictions_test = model.predict(X_test)

        print('------------------------------------------------------------')
        print('The mse train loss is: ', train_score[0])
        print('The mae train loss is: ', train_score[1])
        print('The mse test loss is: ', val_score[0])
        print('The mae test loss is: ', val_score[1])
        print('------------------------------------------------------------')

        return model, history, score, predictions_validation, predictions_test

    # ################################### RNN ############################################
    def apply_RNN(self, X, y, validation_data, test_data, hidden_nodes, epochs, optimizer, loss_fct,
                  batch_size=256):
        # reshape
        X = X.reshape((X.shape[0], X.shape[1], 1))
        X_val = validation_data[0].reshape((validation_data[0].shape[0], validation_data[0].shape[1], 1))
        y_val = validation_data[1]
        X_test = test_data[0].reshape((test_data[0].shape[0], test_data[0].shape[1], 1))
        y_test = test_data[1]

        # define model
        model = Sequential()
        # add GRU layers
        if len(hidden_nodes) == 1:
            model.add(GRU(hidden_nodes[0], activation='relu', input_shape=(X.shape[1], 1)))
        else:
            for i in range(len(hidden_nodes)-1):
                if i == 0:
                    model.add(GRU(hidden_nodes[0], activation='relu', input_shape=(X.shape[1], 1),
                                  return_sequences=True))
                else:
                    model.add(GRU(hidden_nodes[i], activation='relu', return_sequences=True))
                # add dropout in between
                model.add(Dropout(0.2))

            model.add(GRU(hidden_nodes[-1], activation='relu')) # last layer does not return sequences
        # add dense layer for output
        model.add(Dense(1))
        model.compile(optimizer=optimizer, loss=loss_fct, metrics=['mae'])
        model.summary()

        # simple early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

        # fit model
        history = model.fit(X, y, epochs=epochs, verbose=2, validation_data=(X_val, y_val), shuffle=False,
                            batch_size=batch_size, callbacks=[es])#,stateful=True

        train_score = model.evaluate(X, y, verbose=0)
        val_score = model.evaluate(X_val, y_val, verbose=0)
        test_score = model.evaluate(X_test, y_test, verbose=0)
        score = {'train': train_score, 'val': val_score, 'test': test_score}

        predictions_validation = model.predict(X_val)
        predictions_test = model.predict(X_test)

        print('------------------------------------------------------------')
        print('The mse train loss is: ', train_score[0])
        print('The mae train loss is: ', train_score[1])
        print('The mse test loss is: ', val_score[0])
        print('The mae test loss is: ', val_score[1])
        print('------------------------------------------------------------')

        return model, history, score, predictions_validation, predictions_test

    def display_forecasting_score(self, models):

        # initialize storage variables
        best_score = 99999
        best_model = None

        all_models = models
        for configuration in all_models:
            config = configuration[-1]
            print('\nNEW CONFIGURATION:')
            print('Configuration: ', config)
            mae_train, mse_train = list(), list()
            mae_val, mse_val = list(), list()
            for pair_i in range(len(configuration) - 1):
                score_train = configuration[pair_i]['score']['train']
                score_val = configuration[pair_i]['score']['val']
                # print('MAE: {:.2f}%'.format(score[1]))
                mse_train.append(score_train[0])
                mae_train.append(score_train[1])
                mse_val.append(score_val[0])
                mae_val.append(score_val[1])
                print('\nPair loaded: {}_{}: Epochs: {} Val_MSE: {}'.format(configuration[pair_i]['leg1'],
                                                                            configuration[pair_i]['leg2'],
                                                                            configuration[pair_i]['epoch_stop'],
                                                                            score_val[0]
                                                                            ))
            if (np.mean(mse_val)) < best_score:
                best_score = np.mean(mse_val)
                best_model = config

            print('\nCONFIGURATION TRAIN MSE ERROR: {:.4f}E-4'.format(np.mean(mse_train) * 10000))
            print('CONFIGURATION TRAIN MAE ERROR: {:.4f}'.format(np.mean(mae_train)))
            print('\nCONFIGURATION VAL MSE ERROR: {:.4f}E-4'.format(np.mean(mse_val) * 10000))
            print('CONFIGURATION VAL MAE ERROR: {:.4f}'.format(np.mean(mae_val)))

        return (best_model, best_score)

    def run_specific_model(self, n_in, hidden_nodes, pairs, path='models/', train_val_split='2017-01-01', lag=1,
                           low_quantile=0.10, high_quantile=0.90):

        nodes_name = str(hidden_nodes[0]) + '*2' if len(hidden_nodes) > 1 else str(hidden_nodes[0])
        file_name = 'models_n_in-' + str(n_in) + '_hidden_nodes-' + nodes_name + '.pkl'

        with open(path + file_name, 'rb') as f:
            model = pickle.load(f)

        model_cumret, model_sharpe_ratio = list(), list()
        summaries = list()
        for pair_i in range(len(model) - 1):
            # print('\nPair loaded: {}_{}:'.format(model[pair_i]['leg1'], model[pair_i]['leg2']))
            # print('Check pairs: {}_{}.'.format(pairs[pair_i][0], pairs[pair_i][1]))
            predictions = model[pair_i]['predictions_val']
            ret, cumret, summary = self.forecast_spread_trading(
                                                            X=pairs[pair_i][2]['X_train'][train_val_split:],
                                                            Y=pairs[pair_i][2]['Y_train'][train_val_split:],
                                                            spread_test=pairs[pair_i][2]['spread'][train_val_split:],
                                                            spread_train=pairs[pair_i][2]['spread'][:train_val_split],
                                                            beta=pairs[pair_i][2]['coint_coef'],
                                                            predictions=predictions,
                                                            lag=lag,
                                                            low_quantile=low_quantile,
                                                            high_quantile=high_quantile)
            print('Accumulated return: {:.2f}%'.format(cumret[-1] * 100))

            if np.std(ret) != 0:
                sharpe_ratio = np.sqrt(252*1*78) * np.mean(ret) / np.std(ret)
            else:
                sharpe_ratio = 0
            print('Sharpe Ratio:', sharpe_ratio)

            model_cumret.append(cumret[-1] * 100)
            model_sharpe_ratio.append(sharpe_ratio)
            summaries.append(summary)

        print('\nModel mean ROI: {:.2f}%'.format(np.mean(model_cumret)))
        print('Model mean Sharpe Ratio: {:.2f}'.format(np.mean(model_sharpe_ratio)))

        return model, model_cumret, summaries

    def test_specific_model(self, n_in, hidden_nodes, pairs, path, train_test_split='2018-01-01', lag=1,
                            low_quantile=0.10, high_quantile=0.90, profitable_pairs_indices=None):

        nodes_name = str(hidden_nodes[0]) + '*2' if len(hidden_nodes) > 1 else str(hidden_nodes[0])
        file_name = 'models_n_in-' + str(n_in) + '_hidden_nodes-' + nodes_name + '.pkl'

        with open(path + file_name, 'rb') as f:
            model = pickle.load(f)

        model_cumret, model_sharpe_ratio = list(), list()
        summaries = list()
        for pair_i in range(len(model) - 1):
            if pair_i in profitable_pairs_indices:
                print('\nPair loaded: {}_{}:'.format(model[pair_i]['leg1'], model[pair_i]['leg2']))
                #print('Check pairs: {}_{}.'.format(pairs[pair_i][0], pairs[pair_i][1]))
                predictions = model[pair_i]['predictions_test']
                spread_test = pairs[pair_i][2]['Y_test'] - pairs[pair_i][2]['coint_coef'] * pairs[pair_i][2]['X_test']
                ret, cumret, summary = self.forecast_spread_trading(
                                                            X=pairs[pair_i][2]['X_test'],
                                                            Y=pairs[pair_i][2]['Y_test'],
                                                            spread_test=spread_test[-len(predictions):],
                                                            spread_train=pairs[pair_i][2]['spread'][:train_test_split],
                                                            beta=pairs[pair_i][2]['coint_coef'],
                                                            predictions=predictions,
                                                            lag=lag,
                                                            low_quantile=low_quantile,
                                                            high_quantile=high_quantile)
                print('Accumulated return: {:.2f}%'.format(cumret[-1] * 100))

                if np.std(ret) != 0:
                    sharpe_ratio = np.sqrt(252 * 1 * 78) * np.mean(ret) / np.std(ret)
                else:
                    sharpe_ratio = 0
                print('Sharpe Ratio:', sharpe_ratio)

                model_cumret.append(cumret[-1] * 100)
                model_sharpe_ratio.append(sharpe_ratio)
                summaries.append(summary)

        print('\nModel mean ROI on test set: {:.2f}%'.format(np.mean(model_cumret)))
        print('Model mean Sharpe Ratio on test set: {:.2f}'.format(np.mean(model_sharpe_ratio)))

        return model, model_cumret, summaries


