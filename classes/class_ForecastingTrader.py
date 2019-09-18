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

from classes import class_Trader

from sklearn.preprocessing import StandardScaler

# Import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, TimeDistributed, CuDNNLSTM
from keras.callbacks import EarlyStopping
from keras.initializers import glorot_normal
from keras.layers import RepeatVector
from keras.utils import plot_model
#from keras_sequential_ascii import keras2ascii
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
                                low_quantile=0.15, high_quantile=0.85, multistep=0):
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
        if multistep == 0:
            predictions_1 = predictions
            predictions_2 = pd.Series(data=[0]*len(predictions), index=predictions.index)
            predictions_pct_change = (((predictions_1 - spread_test.shift(lag)) /
                                       abs(spread_test.shift(lag))) * 100).fillna(0)
            # true_change = spread_test.diff().fillna(0)
        else:
            predictions_1, predictions_2 = predictions['t'], predictions['t+1']
            predictions_pct_change = (((predictions_2 - spread_test.shift(lag)) /
                                       abs(spread_test.shift(lag))) * 100).fillna(0)
            # need to add last row and first row correspondingly
            predictions_1 = predictions_1.append(pd.Series(data=predictions_2[-1], index=spread_test[-1:].index))
            predictions_2 = pd.concat([pd.Series(data=predictions_1[0], index=predictions_1[:1].index), predictions_2])

        # 2. Calculate trading thresholds
        spread_train_pct_change = ((spread_train - spread_train.shift(lag+multistep)) /
                                   abs(spread_train.shift(lag+multistep))) * 100
        positive_changes = spread_train_pct_change[spread_train_pct_change > 0]
        negative_changes = spread_train_pct_change[spread_train_pct_change < 0]
        long_threshold = positive_changes.quantile(q=high_quantile, interpolation='linear')
        #print('Long threshold: {:.2f}'.format(long_threshold))
        short_threshold = negative_changes.quantile(q=low_quantile, interpolation='linear')
        #print('Short threshold: {:.2f}'.format(short_threshold))

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
            pct_change_from_previous = predictions_pct_change = (((predictions_1 - spread_test.shift(1)) /
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
        # calculate return per position
        position_ret, _, _ = trader.calculate_position_returns(Y, X, beta, numUnits)
        df = pd.DataFrame({'position_return':position_ret.values,
                           'trading_duration':trading_durations,
                           'position_during_day': numUnits.shift(1).fillna(0).values},
                          index = position_ret.index)
        position_ret_with_costs = trader.add_transaction_costs(df, beta)
        balance_summary['position_ret_with_costs']=position_ret_with_costs

        # summarize
        ret_with_costs, cum_ret_with_costs = balance_summary.returns, (balance_summary.account_balance-1)
        bins = [-np.inf, -0.00000001, 0.00000001, np.inf]
        names = ['-1', '0', '1']
        summary = pd.DataFrame(data={'prediction(t)': predictions_1.values,
                                     'prediction(t+1)': predictions_2.values,
                                     'spread(t)': spread_test.values,
                                     'predicted_change(%)': predictions_pct_change,
                                     'position_during_day': numUnits.shift(1).fillna(0).values[lookback:],
                                     'position_return': position_ret,
                                     'position_ret_with_costs': position_ret_with_costs,
                                     'trading_days': trading_durations[lookback:],
                                     'ret_with_costs': ret_with_costs[lookback:],
                                     'predicted_direction': pd.cut(predictions_pct_change, bins, labels=names),
                                     'true_direction': pd.cut(spread_test.diff(), bins, labels=names)
                                     },
                               index=spread_test.index)

        return ret_with_costs, cum_ret_with_costs, summary, balance_summary

    def returns_forecasting_trading(self, Y, X, beta, predictions, test):
        """
        This function implements Dunis methodology.
        It tracks big changes in returns, and opens a position when the change in the returns is significant.
        """
        # track positions for which the expected p&l overweights the transaction costs
        numUnits = pd.Series(data=[0.] * len(predictions), index=predictions.index, name='numUnits')
        long_opportunities = predictions > 0.0056
        short_opportunities = predictions < -0.0056
        longsEntry = long_opportunities.shift(-1).fillna(False)
        numUnits[longsEntry] = 1.
        shortsEntry = short_opportunities.shift(-1).fillna(False)
        numUnits[shortsEntry] = -1.

        # Calculate P&L and Returns
        trader = class_Trader.Trader()
        # concatenate positions with not enough data to be predicted
        lookback = len(Y) - len(predictions)
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
        ret_with_costs, cum_ret_with_costs = balance_summary.returns, (balance_summary.account_balance - 1)
        summary = pd.DataFrame(data={'predicted_pnl(t)': predictions.values,
                                     'pnl(t)': test.values,
                                     'position_during_day': numUnits.shift(1).fillna(0).values[lookback:],
                                     'Y': Y[lookback:],
                                     'X': X[lookback:],
                                     'Y_pct_change': Y.pct_change()[lookback:],
                                     'X_pct_change': X.pct_change()[lookback:],
                                     'trading_days': trading_durations[lookback:],
                                     'ret_with_costs': ret_with_costs[lookback:]
                                     },
                               index=test.index)

        return ret_with_costs, cum_ret_with_costs, summary, balance_summary

    def spread_trading(self, X, Y, spread_test, spread_train, beta, predictions, lag, low_quantile=0.10,
                       high_quantile=0.90, multistep=0):
        """
        This function will set the trading positions based on the forecasted spread.
        For each day, the function compares the predicted spread for that day with the
        true value of the spread in tha day before, giving the predicted spread pct change.
        In case it is larger than the threshold, a position is entered.
        Note: because a position entered in day n it is only accounted for on the day after,
        we shift the entered positions.
        : predictions: predictions should not be standardized, but with regular mean and variance.
        """
        # 1. Get predictions change
        if multistep == 0:
            predictions_1 = predictions
            predictions_change = predictions.diff().fillna(0)
            true_change = spread_test.diff().fillna(0)
        else:
            predictions_1, predictions_2 = predictions
            predictions_change = (predictions_2 - predictions_1.shift(lag)).fillna(0)
            # need to add last row
            predictions_change = predictions_change.append(pd.Series(data=[0], index=spread_test[-1:].index))
            predictions_1 = predictions_1.append(pd.Series(data=predictions_2[-1], index=spread_test[-1:].index))

        # 2. Calculate trading threshold
        spread_train_change = (spread_train - spread_train.shift(lag+multistep)).fillna(0)
        positive_changes = spread_train_change[spread_train_change > 0]
        negative_changes = spread_train_change[spread_train_change < 0]
        long_threshold = positive_changes.quantile(q=high_quantile, interpolation='linear')
        print('Long threshold: {:.2f}'.format(long_threshold))
        short_threshold = negative_changes.quantile(q=low_quantile, interpolation='linear')
        print('Short threshold: {:.2f}'.format(short_threshold))

        # 3. Define trading timings
        numUnits = pd.Series(data=[0.] * len(spread_test), index=spread_test.index, name='numUnits')
        longsEntry = (predictions_change > long_threshold) & (true_change.shift() > 0)
        longsEntry = longsEntry.shift(-1).fillna(False)
        numUnits[longsEntry] = 1.
        shortsEntry = (predictions_change < short_threshold) & (true_change.shift() < 0)
        shortsEntry = shortsEntry.shift(-1).fillna(False)
        numUnits[shortsEntry] = -1.

        # ffill if applicable
        if lag == 1:
            change_from_previous = predictions_change
        else:
            change_from_previous = (predictions_1 - spread_test.shift(1)).fillna(0)
        for i in range(1, len(numUnits) - 1):
            if numUnits[i] != 0:
                continue
            else:
                if numUnits[i - 1] == 0:
                    continue
                elif numUnits[i - 1] == 1.:
                    if change_from_previous[i + 1] > 0:
                        numUnits[i] = 1
                        continue
                elif numUnits[i - 1] == -1.:
                    if change_from_previous[i + 1] < 0:
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
        ret_with_costs, cum_ret_with_costs = balance_summary.returns, (balance_summary.account_balance-1)
        summary = pd.DataFrame(data={'prediction(t)': predictions_1.values,
                                     'spread(t)': spread_test.values,
                                     'predicted_change': predictions_change,
                                     'true_change': spread_test.diff().fillna(0).values,
                                     'position_during_day': numUnits.shift(1).fillna(0).values[lookback:],
                                     'Y': Y[lookback:],
                                     'X': X[lookback:],
                                     'trading_days': trading_durations[lookback:],
                                     'ret_with_costs': ret_with_costs[lookback:]
                                     },
                               index=spread_test.index)
        print('Accuracy of time series forecasting: {:.2f}%'.format(self.calculate_direction_accuracy(spread_test,
                                                                                                  predictions_1)))

        return ret_with_costs, cum_ret_with_costs, summary, balance_summary

    def momentum_trading(self, X, Y, spread_test, spread_train, beta, predictions, lag, low_quantile=0.10,
                         high_quantile=0.90, multistep=0):
        """
        This function will set the trading positions based on the forecasted spread.
        For each day, the function compares the predicted spread for that day with the
        true value of the spread in tha day before, giving the predicted spread pct change.
        In case it is larger than the threshold, a position is entered.
        Note: because a position entered in day n it is only accounted for on the day after,
        we shift the entered positions.
        : predictions: predictions should not be standardized, but with regular mean and variance.
        """
        # 1. Get predictions change
        if multistep == 0:
            predictions_1 = predictions
            predictions_change = spread_test - predictions_1
        else:
            predictions_1, predictions_2 = predictions
            predictions_change = (predictions_2 - predictions_1.shift(lag)).fillna(0)
            # need to add last row
            predictions_change = predictions_change.append(pd.Series(data=[0], index=spread_test[-1:].index))
            predictions_1 = predictions_1.append(pd.Series(data=predictions_2[-1], index=spread_test[-1:].index))

        # 2. Calculate trading threshold
        spread_train_change = (spread_train - spread_train.shift(lag+multistep)).fillna(0)
        positive_changes = spread_train_change[spread_train_change > 0]
        negative_changes = spread_train_change[spread_train_change < 0]
        long_threshold = positive_changes.quantile(q=high_quantile, interpolation='linear')
        print('Long threshold: {:.2f}'.format(long_threshold))
        short_threshold = negative_changes.quantile(q=low_quantile, interpolation='linear')
        print('Short threshold: {:.2f}'.format(short_threshold))

        # 3. Define trading timings
        numUnits = pd.Series(data=[0.] * len(spread_test), index=spread_test.index, name='numUnits')
        longsEntry = predictions_change > long_threshold
        numUnits[longsEntry] = 1.
        shortsEntry = predictions_change < short_threshold
        numUnits[shortsEntry] = -1.

        # ffill if applicable
        if lag == 1:
            change_from_previous = predictions_change
        else:
            change_from_previous = (predictions_1 - spread_test.shift(1)).fillna(0)
        for i in range(1, len(numUnits) - 1):
            if numUnits[i] != 0:
                continue
            else:
                if numUnits[i - 1] == 0:
                    continue
                elif numUnits[i - 1] == 1.:
                    if change_from_previous[i] > 0:
                        numUnits[i] = 1
                        continue
                elif numUnits[i - 1] == -1.:
                    if change_from_previous[i] < 0:
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
        ret_with_costs, cum_ret_with_costs = balance_summary.returns, (balance_summary.account_balance-1)
        summary = pd.DataFrame(data={'prediction(t)': predictions_1.values,
                                     'spread(t)': spread_test.values,
                                     'spread_predicted_change': predictions_change.values,
                                     'position_during_day': numUnits.shift(1).fillna(0).values[lookback:],
                                     '{}'.format(Y.name): Y[lookback:],
                                     '{}'.format(X.name): X[lookback:],
                                     'trading_days': trading_durations[lookback:],
                                     'ret_with_costs': ret_with_costs[lookback:]
                                     },
                               index=spread_test.index)
        print('Accuracy of time series forecasting: {:.2f}%'.format(self.calculate_direction_accuracy(spread_test,
                                                                                                  predictions_1)))

        return ret_with_costs, cum_ret_with_costs, summary, balance_summary

    def calculate_direction_accuracy(self, true, predictions):

        bins = [-np.inf, -0.00000001, 0.00000001, np.inf]
        names = ['-1', '0', '1']
        predictions_change = predictions.diff().fillna(0)

        predicted_direction = pd.cut(predictions_change, bins, labels=names)
        true_direction = pd.cut(true.diff().fillna(0), bins, labels=names)
        #accuracy = len(predicted_direction[predicted_direction == true_direction])/len(predicted_direction) * 100

        predicted_direction_subset = predicted_direction[true_direction != '0']
        true_direction_subset = true_direction[true_direction != '0']
        accuracy = len(predicted_direction_subset[predicted_direction_subset == true_direction_subset]) / \
                   len(predicted_direction_subset) * 100

        return accuracy

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

        scaler = StandardScaler()
        spread_norm = scaler.fit_transform(spread.values.reshape(spread.shape[0], 1))
        spread_norm = pd.Series(data=spread_norm.flatten(), index=spread.index)
        forecasting_data = self.series_to_supervised(list(spread_norm), spread.index, model_config['n_in'],
                                                     model_config['n_out'], dropnan=True)
        # define dataset
        if model_config['n_out'] == 1:
            X_series = forecasting_data.drop(columns='var1(t)')
            y_series = forecasting_data['var1(t)']
        elif model_config['n_out'] == 2:
            X_series = forecasting_data.drop(columns=['var1(t)', 'var1(t+1)'])
            y_series = forecasting_data[['var1(t)', 'var1(t+1)']]

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
        spread_norm = scaler.transform(spread.values.reshape(spread.shape[0], 1))
        spread_norm = pd.Series(data=spread_norm.flatten(), index=spread.index)
        forecasting_data = self.series_to_supervised(list(spread_norm), spread.index, model_config['n_in'],
                                                     model_config['n_out'], dropnan=True)
        # define dataset
        if model_config['n_out'] == 1:
            X_series_test = forecasting_data.drop(columns='var1(t)')
            y_series_test = forecasting_data['var1(t)']
        elif model_config['n_out'] == 2:
            X_series_test = forecasting_data.drop(columns=['var1(t)', 'var1(t+1)'])
            y_series_test = forecasting_data[['var1(t)', 'var1(t+1)']]

        X_test = X_series_test.values
        y_test = y_series_test.values

        return (X_test, y_test), y_series_test

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
            test_data, y_series_test = self.prepare_test_data(spread_test, model_config, scaler)

            # train model and get predictions
            if model_type == 'mlp':
                model, history, score, predictions_train, predictions_val, predictions_test = \
                                                              self.apply_MLP(X=train_data[0],
                                                                            y=train_data[1],
                                                                            validation_data=validation_data,
                                                                            test_data=test_data,
                                                                            n_in=model_config['n_in'],
                                                                            hidden_nodes=model_config['hidden_nodes'],
                                                                            epochs=model_config['epochs'],
                                                                            optimizer=model_config['optimizer'],
                                                                            loss_fct=model_config['loss_fct'],
                                                                            batch_size=model_config['batch_size'])

            elif model_type == 'rnn':
                model, history, score, predictions_train, predictions_val, predictions_test = \
                                                              self.apply_RNN(X=train_data[0],
                                                                             y=train_data[1],
                                                                             validation_data=validation_data,
                                                                             test_data=test_data,
                                                                             hidden_nodes=model_config['hidden_nodes'],
                                                                             epochs=model_config['epochs'],
                                                                             optimizer=model_config['optimizer'],
                                                                             loss_fct=model_config['loss_fct'],
                                                                             batch_size=model_config['batch_size'])
            elif model_type == 'encoder_decoder':
                model, history, score, predictions_train, predictions_val, predictions_test = \
                                                self.apply_encoder_decoder(X=train_data[0],
                                                                           y=train_data[1],
                                                                           validation_data=validation_data,
                                                                           test_data=test_data,
                                                                           n_in=model_config['n_in'],
                                                                           n_out=model_config['n_out'],
                                                                           hidden_nodes=model_config['hidden_nodes'],
                                                                           epochs=model_config['epochs'],
                                                                           optimizer=model_config['optimizer'],
                                                                           loss_fct=model_config['loss_fct'],
                                                                           batch_size=model_config['batch_size'])

                # validation
                predictions_val = pd.DataFrame({'t': predictions_val.reshape(predictions_val.shape[0],
                                                                             predictions_val.shape[1])[:, 0],
                                                't+1': predictions_val.reshape(predictions_val.shape[0],
                                                                               predictions_val.shape[1])[:, 1]},
                                               index=y_series_val.index)
                predictions_val['t'] = scaler.inverse_transform(np.array(predictions_val['t']))
                predictions_val['t+1'] = scaler.inverse_transform(np.array(predictions_val['t+1']))

                # test
                predictions_test = pd.DataFrame({'t': predictions_test.reshape(predictions_test.shape[0],
                                                                               predictions_test.shape[1])[:, 0],
                                                't+1': predictions_test.reshape(predictions_test.shape[0],
                                                                                predictions_test.shape[1])[:, 1]},
                                                index=y_series_test.index)
                predictions_test['t'] = scaler.inverse_transform(np.array(predictions_test['t']))
                predictions_test['t+1'] = scaler.inverse_transform(np.array(predictions_test['t+1']))

                # train
                predictions_train = predictions_val.copy()  # not relevant, just to fill up

            # transform predictions to series
            if model_type != 'encoder_decoder':
                predictions_train = scaler.inverse_transform(predictions_train)
                predictions_val = scaler.inverse_transform(predictions_val)
                predictions_test = scaler.inverse_transform(predictions_test)
                predictions_train = pd.Series(data=predictions_train.flatten(),
                                              index=spread[model_config['n_in']:-len(y_series_val)].index)
                predictions_val = pd.Series(data=predictions_val.flatten(), index=y_series_val.index)
                predictions_test = pd.Series(data=predictions_test.flatten(),
                                             index=spread_test[-len(test_data[1]):].index)

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
                          'predictions_train': predictions_train.copy(),
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

        # define validation set
        X_val = validation_data[0]
        y_val = validation_data[1]

        # define test set
        X_test = test_data[0]
        y_test = test_data[1]

        model = Sequential()
        glorot_init = glorot_normal(seed=None)
        for i in range(len(hidden_nodes)):
            model.add(Dense(hidden_nodes[i], activation='relu', input_dim=n_in, kernel_initializer=glorot_init))
        #model.add(Dropout(0.1))
        model.add(Dense(1))
        model.compile(optimizer=optimizer, loss=loss_fct, metrics=['mae'])
        model.summary()
        if len(hidden_nodes)>1:
            plot_model(model, to_file='/content/drive/PairsTrading/mlp_models/model_{}-{}_{}.png'.format(str(n_in),
                       str(hidden_nodes[0]), str(hidden_nodes[1]), show_shapes=True, show_layer_names=False))
        else:
            plot_model(model, to_file='/content/drive/PairsTrading/mlp_models/model_{}-{}.png'.format(str(n_in),
                       str(hidden_nodes[0])), show_shapes=True, show_layer_names=False)
        #print(keras2ascii(model))

        # simple early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

        history = model.fit(X, y, epochs=epochs, verbose=1, validation_data=validation_data,
                            shuffle=False, batch_size=batch_size, callbacks=[es])

        # scores
        if len(history.history['loss']) < 500:
            train_score = [min(history.history['loss']), min(history.history['mean_absolute_error'])]
            val_score = [min(history.history['val_loss']),min(history.history['val_mean_absolute_error'])]
        else:
            train_score = [history.history['loss'][-1], history.history['mean_absolute_error'][-1]]
            val_score = [history.history['val_loss'][-1], history.history['val_mean_absolute_error'][-1]]
        score = {'train': train_score, 'val': val_score}

        # predictions
        predictions_train = model.predict(X, verbose=1)
        predictions_validation = model.predict(X_val, verbose=1)
        predictions_test = model.predict(X_test, verbose=1)

        print('------------------------------------------------------------')
        print('The mse train loss is: ', train_score[0])
        print('The mae train loss is: ', train_score[1])
        print('The mse test loss is: ', val_score[0])
        print('The mae test loss is: ', val_score[1])
        print('------------------------------------------------------------')

        return model, history, score, predictions_train, predictions_validation, predictions_test

    # ################################### RNN ############################################
    def apply_RNN(self, X, y, validation_data, test_data, hidden_nodes, epochs, optimizer, loss_fct,
                  batch_size=256):
        """
        Note: CuDNNLSTM provides a faster implementation on GPU than regular LSTM
        :param X:
        :param y:
        :param validation_data:
        :param test_data:
        :param hidden_nodes:
        :param epochs:
        :param optimizer:
        :param loss_fct:
        :param batch_size:
        :return:
        """
        # reshape
        X = X.reshape((X.shape[0], X.shape[1], 1))
        X_val = validation_data[0].reshape((validation_data[0].shape[0], validation_data[0].shape[1], 1))
        y_val = validation_data[1]
        X_test = test_data[0].reshape((test_data[0].shape[0], test_data[0].shape[1], 1))
        y_test = test_data[1]

        # define model
        model = Sequential()
        glorot_init = glorot_normal(seed=None)
        # add GRU layers
        if len(hidden_nodes) == 1:
            #model.add(LSTM(hidden_nodes[0], activation='relu', input_shape=(X.shape[1], 1),
            #               kernel_initializer=glorot_init))
            model.add(CuDNNLSTM(hidden_nodes[0], input_shape=(X.shape[1], 1), kernel_initializer=glorot_init))
        else:
            for i in range(len(hidden_nodes)-1):
                if i == 0:
                    #model.add(LSTM(hidden_nodes[0], activation='relu', input_shape=(X.shape[1], 1),
                    #              return_sequences=True, kernel_initializer=glorot_init))
                    model.add(CuDNNLSTM(hidden_nodes[0], input_shape=(X.shape[1], 1),
                                   return_sequences=True, kernel_initializer=glorot_init))
                else:
                    #model.add(LSTM(hidden_nodes[i], activation='relu', return_sequences=True,
                    #               kernel_initializer=glorot_init))
                    model.add(CuDNNLSTM(hidden_nodes[i], return_sequences=True,
                                   kernel_initializer=glorot_init))
                # add dropout in between
                model.add(Dropout(0.1))

            #model.add(LSTM(hidden_nodes[-1], activation='relu', kernel_initializer=glorot_init)) # last layer does not return sequences
            model.add(CuDNNLSTM(hidden_nodes[-1], kernel_initializer=glorot_init))# last layer does not return sequences
        # add regularization
        #model.add(Dropout(0.1))
        # add dense layer for output
        model.add(Dense(1, kernel_initializer=glorot_init))
        model.compile(optimizer=optimizer, loss=loss_fct, metrics=['mae'])
        model.summary()
        plot_model(model, to_file='/content/drive/PairsTrading/rnn_models/model.png', show_shapes=True,
                   show_layer_names=False)

        # simple early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

        # fit model
        history = model.fit(X, y, epochs=epochs, verbose=1, validation_data=(X_val, y_val), shuffle=False,
                            batch_size=batch_size, callbacks=[es])

        # scores
        if len(history.history['loss']) < 500:
            train_score = [min(history.history['loss']), min(history.history['mean_absolute_error'])]
            val_score = [min(history.history['val_loss']),min(history.history['val_mean_absolute_error'])]
        else:
            train_score = [history.history['loss'][-1], history.history['mean_absolute_error'][-1]]
            val_score = [history.history['val_loss'][-1], history.history['val_mean_absolute_error'][-1]]

        score = {'train': train_score, 'val': val_score}

        # removed test score calculation to save time
        #test_score = model.evaluate(X_test, y_test, verbose=1)
        # , 'test': test_score}

        predictions_train = model.predict(X, verbose=1)
        predictions_validation = model.predict(X_val, verbose=1)
        predictions_test = model.predict(X_test, verbose=1)

        print('------------------------------------------------------------')
        print('The mse train loss is: ', train_score[0])
        print('The mae train loss is: ', train_score[1])
        print('The mse test loss is: ', val_score[0])
        print('The mae test loss is: ', val_score[1])
        print('------------------------------------------------------------')

        return model, history, score, predictions_train, predictions_validation, predictions_test

    # ################################### ENCODER DECODER ############################################
    def apply_encoder_decoder(self, X, y, validation_data, test_data, n_in, n_out, hidden_nodes,
                              epochs, optimizer, loss_fct, batch_size=512):

        # reshape from [samples, timesteps] into [samples, timesteps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))
        y = y.reshape((y.shape[0], y.shape[1], 1))
        X_val = validation_data[0].reshape((validation_data[0].shape[0], validation_data[0].shape[1], 1))
        y_val = validation_data[1].reshape((validation_data[1].shape[0], validation_data[1].shape[1], 1))
        X_test = test_data[0].reshape((test_data[0].shape[0], test_data[0].shape[1], 1))
        y_test = test_data[1].reshape((test_data[1].shape[0], test_data[1].shape[1], 1))

        # define model
        glorot_init = glorot_normal(seed=None)
        model = Sequential()

        # CuDNNLSTM provides a faster implementation on GPU
        #model.add(LSTM(hidden_nodes[0], activation='relu', input_shape=(n_in, 1),  kernel_initializer=glorot_init))
        model.add(CuDNNLSTM(hidden_nodes[0], input_shape=(n_in, 1), kernel_initializer=glorot_init))
        model.add(RepeatVector(n_out))

        # CuDNNLSTM provides a faster implementation on GPU
        #model.add(LSTM(hidden_nodes[1], activation='relu', return_sequences=True,  kernel_initializer=glorot_init))
        model.add(CuDNNLSTM(hidden_nodes[1], return_sequences=True, kernel_initializer=glorot_init))

        #model.add(Dropout(0.1))
        model.add(TimeDistributed(Dense(1, kernel_initializer=glorot_init)))
        model.compile(optimizer=optimizer, loss=loss_fct, metrics=['mae'])
        model.summary()
        plot_model(model, to_file='/content/drive/PairsTrading/encoder_decoder/model.png', show_shapes=True,
                   show_layer_names=False)

        # fit model
        # simple early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

        # fit model
        history = model.fit(X, y, epochs=epochs, verbose=1, validation_data=(X_val, y_val), shuffle=False,
                            batch_size=batch_size, callbacks=[es])

        # scores
        if len(history.history['loss']) < 500:
            train_score = [min(history.history['loss']), min(history.history['mean_absolute_error'])]
            val_score = [min(history.history['val_loss']), min(history.history['val_mean_absolute_error'])]
        else:
            train_score = [history.history['loss'][-1], history.history['mean_absolute_error'][-1]]
            val_score = [history.history['val_loss'][-1], history.history['val_mean_absolute_error'][-1]]

        score = {'train': train_score, 'val': val_score}

        predictions_train = model.predict(X, verbose=1)
        predictions_validation = model.predict(X_val, verbose=1)
        predictions_test = model.predict(X_test, verbose=1)

        print('------------------------------------------------------------')
        print('The mse train loss is: ', train_score[0])
        print('The mae train loss is: ', train_score[1])
        print('The mse test loss is: ', val_score[0])
        print('The mae test loss is: ', val_score[1])
        print('------------------------------------------------------------')

        return model, history, score, predictions_train, predictions_validation, predictions_test

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
                           multistep=0, low_quantile=0.10, high_quantile=0.90):

        nodes_name = str(hidden_nodes[0]) + '_' + str(hidden_nodes[1]) if len(hidden_nodes) > 1 else str(hidden_nodes[0])
        file_name = 'models_n_in-' + str(n_in) + '_hidden_nodes-' + nodes_name + '.pkl'

        with open(path + file_name, 'rb') as f:
            model = pickle.load(f)

        model_cumret, model_sharpe_ratio = list(), list()
        balance_summaries, summaries = list(), list()
        for pair_i in range(len(model) - 1):
            #print('\nPair loaded: {}_{}:'.format(model[pair_i]['leg1'], model[pair_i]['leg2']))
            #print('Check pairs: {}_{}.'.format(pairs[pair_i][0], pairs[pair_i][1]))
            predictions = model[pair_i]['predictions_val']

            ret, cumret, summary, balance_summary = self.forecast_spread_trading(
                                                            X=pairs[pair_i][2]['X_train'][train_val_split:],
                                                            Y=pairs[pair_i][2]['Y_train'][train_val_split:],
                                                            spread_test=pairs[pair_i][2]['spread'][train_val_split:],
                                                            spread_train=pairs[pair_i][2]['spread'][:train_val_split],
                                                            beta=pairs[pair_i][2]['coint_coef'],
                                                            predictions=predictions,
                                                            lag=lag,
                                                            low_quantile=low_quantile,
                                                            high_quantile=high_quantile,
                                                            multistep=multistep)

            #print('Accumulated return: {:.2f}%'.format(cumret[-1] * 100))

            trader = class_Trader.Trader()
            if np.std(ret) != 0:
                sharpe_ratio = trader.calculate_sharpe_ratio(1, 252, ret)
            else:
                sharpe_ratio = 0
            #print('Sharpe Ratio:', sharpe_ratio)

            model_cumret.append(cumret[-1] * 100)
            model_sharpe_ratio.append(sharpe_ratio)
            summaries.append(summary)
            balance_summaries.append(balance_summary)

        return model, model_cumret, model_sharpe_ratio, summaries, balance_summaries

    def test_specific_model(self, n_in, hidden_nodes, pairs, path, train_test_split='2018-01-01', lag=1,
                            low_quantile=0.10, high_quantile=0.90, multistep=0, profitable_pairs_indices=None):

        nodes_name = str(hidden_nodes[0]) + '_' + str(hidden_nodes[1]) if len(hidden_nodes) > 1 else str(
            hidden_nodes[0])
        file_name = 'models_n_in-' + str(n_in) + '_hidden_nodes-' + nodes_name + '.pkl'

        with open(path + file_name, 'rb') as f:
            model = pickle.load(f)

        model_cumret, model_sharpe_ratio = list(), list()
        summaries, balance_summaries = list(), list()
        for pair_i in range(len(model) - 1):
            if pair_i in profitable_pairs_indices:
                #print('\nPair loaded: {}_{}:'.format(model[pair_i]['leg1'], model[pair_i]['leg2']))
                #print('Check pairs: {}_{}.'.format(pairs[pair_i][0], pairs[pair_i][1]))
                predictions = model[pair_i]['predictions_test']
                spread_test = pairs[pair_i][2]['Y_test'] - pairs[pair_i][2]['coint_coef'] * pairs[pair_i][2]['X_test']

                ret, cumret, summary, balance_summary = self.forecast_spread_trading(
                                                            X=pairs[pair_i][2]['X_test'],
                                                            Y=pairs[pair_i][2]['Y_test'],
                                                            spread_test=spread_test[-len(predictions)-multistep:],
                                                            spread_train=pairs[pair_i][2]['spread'][:train_test_split],
                                                            beta=pairs[pair_i][2]['coint_coef'],
                                                            predictions=predictions,
                                                            lag=lag,
                                                            low_quantile=low_quantile,
                                                            high_quantile=high_quantile,
                                                            multistep=multistep)

                #print('Accumulated return: {:.2f}%'.format(cumret[-1] * 100))

                trader = class_Trader.Trader()
                if np.std(ret) != 0:
                    sharpe_ratio = trader.calculate_sharpe_ratio(1, 252, ret)
                else:
                    sharpe_ratio = 0
                #print('Sharpe Ratio:', sharpe_ratio)

                model_cumret.append(cumret[-1] * 100)
                model_sharpe_ratio.append(sharpe_ratio)
                summaries.append(summary)
                balance_summaries.append(balance_summary)

        return model, model_cumret, model_sharpe_ratio, summaries, balance_summaries


