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

# Import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Dropout
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.callbacks import EarlyStopping
# just set the seed for the random number generator




class ForecastingTrader:
    """


    """
    def __init__(self):
        """
        :initial elements
        """

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

    def prepare_data(self, spread, model_config):
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
        standardization_dict = {'mean': spread[:train_val_split].mean(), 'std': np.std(spread[:train_val_split])}
        spread = (spread - standardization_dict['mean']) / standardization_dict['std']
        forecasting_data = self.series_to_supervised(list(spread), spread.index, model_config['n_in'],
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

        return (X_train, y_train), (X_val, y_val), y_series_val, standardization_dict

    def destandardize(self, predictions, spread_mean, spread_std):
        """
        This function transforms the normalized predictions into the original space.
        """
        return predictions * spread_std + spread_mean

    def train_models(self, pairs, model_config):
        """
        This function trains the models for every pair identified.

        :param pairs: list with pairs and corresponding statistics
        :param model_config: dictionary with info for the model
        :return: all models
        """

        models = []
        for pair in pairs:

            # prepare data
            spread = pair[2]['spread']
            train_data, validation_data, y_series_val, standardization_dict = self.prepare_data(spread, model_config)

            # train model and get predictions
            model = None # ensure new var is created
            model, history, score, predictions = self.apply_MLP(X=train_data[0],
                                                                y=train_data[1],
                                                                validation_data=validation_data,
                                                                n_in=model_config['n_in'],
                                                                hidden_nodes=model_config['hidden_nodes'],
                                                                epochs=model_config['epochs'],
                                                                optimizer=model_config['optimizer'],
                                                                loss_fct=model_config['loss_fct'])

            predictions = pd.Series(data=predictions.flatten(), index=y_series_val.index)
            predictions_destandardized = self.destandardize(predictions,
                                                            standardization_dict['mean'],
                                                            standardization_dict['std'])

            # save all info
            model_info = {'leg1': pair[0],
                          'leg2': pair[1],
                          'info': pair[2].copy(),
                          'standardization_dict': standardization_dict,
                          'model': model,
                          'history': history,
                          'score': score,
                          'predictions': predictions_destandardized.copy(),
                          }

            models.append(model_info)

        # append model configuration on last position
        models.append(model_config)

        return models

    # ################################### MLP ############################################
    def apply_MLP(self, X, y, validation_data, n_in, hidden_nodes, epochs, optimizer, loss_fct):
        # reset seed
        # np.random.seed(0) # NumPy
        # tf.set_random_seed(2) # Tensorflow
        # random.seed(3) # Python

        # define validation set
        X_val = validation_data[0]
        y_val = validation_data[1]

        model = Sequential()
        for i in range(len(hidden_nodes)):
            model.add(Dense(hidden_nodes[i], activation='relu', input_dim=n_in))
            # model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer=optimizer, loss=loss_fct, metrics=['mae'])
        model.summary()

        # simple early stopping
        # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        # pb = ProgbarLogger(count_mode='samples', stateful_metrics=None)

        history = model.fit(X, y, epochs=epochs, verbose=2, validation_data=validation_data,
                            shuffle=False, batch_size=128)  # , callbacks=[pb, es])

        train_score = model.evaluate(X, y, verbose=0)
        val_score = model.evaluate(X_val, y_val, verbose=0)
        score = {'train': train_score, 'val': val_score}

        predictions = model.predict(X_val)

        print('------------------------------------------------------------')
        print('The mse train loss is: ', train_score[0])
        print('The mae train loss is: ', train_score[1])
        print('The mse test loss is: ', val_score[0])
        print('The mae test loss is: ', val_score[1])
        print('------------------------------------------------------------')

        return model, history, score, predictions

