import numpy as np
import pandas as pd
import sys
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import timedelta

# just set the seed for the random number generator
np.random.seed(107)

class Trader:
    """
    This class contains a set of pairs trading strategies along
    with some auxiliary functions
    """

    def __init__(self):
        """
        :initial elements
        """

    def rolling_regression(self, y, x, window):
        """
        y and x must be pandas.Series
        y is the dependent variable
        x is the independent variable
        spread: y - b*x
        Source: https://stackoverflow.com/questions/37317727/deprecated-rolling-window-
                option-in-ols-from-pandas-to-statsmodels/39704930#39704930
        """
        # Clean-up
        x = x.dropna()
        y = y.dropna()
        # Trim acc to shortest
        if x.index.size > y.index.size:
            x = x[y.index]
        else:
            y = y[x.index]
        # Verify enough space
        if x.index.size < window:
            return None
        else:
            # Add a constant if needed
            X_name = x.name
            X = x.to_frame()
            X['c'] = 1
            # Loop... this can be improved
            estimate_data = []
            for i in range(window, len(X)):
                X_slice = X.iloc[i - window:i, :]  # always index in np as opposed to pandas, much faster
                y_slice = y.iloc[i - window:i]
                coeff = sm.OLS(y_slice, X_slice).fit()
                estimate_data.append(coeff.params[X_name])

            # Assemble
            estimate = pd.Series(data=np.nan, index=x.index[:window])
            # add nan values for first #lookback indices
            estimate = estimate.append(pd.Series(data=estimate_data, index=x.index[window:]))
            return estimate

    def rolling_zscore(self, Y, X, lookback):
        """
        This function calculates the normalized moving spread
        Note that moving average and moving std will have the first 39 values as np.Nan, because
        the spread is only valid after 20 points, and the moving averages still need 20 points more
        to define its value.
        """
        # Calculate moving parameters
        # 1.beta:
        rolling_beta = self.rolling_regression(Y, X, window=lookback)
        # 2.spread:
        rolling_spread = Y - rolling_beta * X
        # 3.moving average
        rolling_avg = rolling_spread.rolling(window=lookback, center=False).mean()
        rolling_avg.name = 'spread_' + str(lookback) + 'mavg'
        # 4. rolling standard deviation
        rolling_std = rolling_spread.rolling(window=lookback, center=False).std()
        rolling_std.name = 'rolling_std_' + str(lookback)

        # z-score
        zscore = (rolling_spread - rolling_avg) / rolling_std

        return zscore, rolling_beta

    def threshold_strategy(self, y, x, beta, entry_level=1.0, exit_level=1.0, stabilizing_threshold=5,
                           predictions=None):
        """
        This function implements a threshold filter strategy with a fixed beta, found from cointegration test.
        :param y:
        :param x:
        :param entry_multiplier:
        :param exit_multiplier:
        :param stabilizing_threshold:
        :return:
        """

        spread = y - beta * x
        norm_spread = (spread - spread.mean()) / np.std(spread)
        norm_spread = np.asarray(norm_spread.values)

        longs_entry = norm_spread < -entry_level
        longs_exit = norm_spread > -exit_level
        shorts_entry = norm_spread > entry_level
        shorts_exit = norm_spread < exit_level

        num_units_long = pd.Series([np.nan for i in range(len(y))])
        num_units_short = pd.Series([np.nan for i in range(len(y))])

        # remove trades while the spread is stabilizing
        longs_entry[:stabilizing_threshold] = False
        longs_exit[:stabilizing_threshold] = False
        shorts_entry[:stabilizing_threshold] = False
        shorts_exit[:stabilizing_threshold] = False

        num_units_long[longs_entry] = 1.
        num_units_long[longs_exit] = 0
        num_units_short[shorts_entry] = -1.
        num_units_short[shorts_exit] = 0
        # shift to simulate delay in real life trading
        # please comment if no need to simulate delay
        num_units_long = num_units_long.shift(1)
        num_units_short = num_units_short.shift(1)
        # initialize with zero
        num_units_long[0] = 0.
        num_units_short[0] = 0.
        # finally, fill in between
        num_units_long = num_units_long.fillna(method='ffill')
        num_units_short = num_units_short.fillna(method='ffill')

        num_units = num_units_long + num_units_short
        num_units = pd.Series(data=num_units.values, index=y.index, name='numUnits')

        if predictions is not None:
            df = pd.DataFrame({'units': num_units.shift(1).fillna(0), 'predictions': predictions,
                               'predictions_change': predictions.diff().fillna(0)})
            df = self.update_positions(df, 'predictions_change', 0)
            num_units = df.units.shift(-1).fillna(0)
            num_units.name = 'numUnits'
        else:
            predictions = pd.Series(data=[0]*len(num_units), index=y.index)

        # position durations
        trading_durations = self.add_trading_duration(pd.DataFrame(num_units, index=y.index))


        # calculate return per position
        position_ret, _, ret_summary = self.calculate_position_returns(y, x, beta, num_units)
        # calculate balance in total
        balance_summary = self.calculate_balance(y, x, beta, num_units.shift(1).fillna(0), trading_durations)

        # add transaction costs and gather all info in df
        series_to_include = [(balance_summary.pnl, 'pnl'),
                             (balance_summary.pnl_y, 'pnl_y'),
                             (balance_summary.pnl_x, 'pnl_x'),
                             (balance_summary.account_balance, 'account_balance'),
                             (balance_summary.returns, 'returns'),
                             (position_ret, 'position_return'),
                             (predictions.diff().fillna(0), 'prediction_change'),
                             (y, y.name),
                             (x, x.name),
                             (pd.Series(norm_spread, index=y.index), 'norm_spread'),
                             (num_units, 'numUnits'),
                             (trading_durations, 'trading_duration')]

        summary = self.trade_summary(series_to_include, beta)

        # calculate sharpe ratio
        ret_w_costs = summary.returns
        n_years = round(len(y) / (240 * 78))
        n_days = 252
        if np.std(ret_w_costs) == 0:
            sharpe_no_costs, sharpe_w_costs = (0, 0)
        else:
            if np.std(position_ret) == 0:
                sharpe_no_costs=0
            else:
                sharpe_no_costs = self.calculate_sharpe_ratio(n_years, n_days, position_ret)
            sharpe_w_costs = self.calculate_sharpe_ratio(n_years, n_days, ret_w_costs)

        return summary, (sharpe_no_costs, sharpe_w_costs), balance_summary

    def apply_trading_strategy(self, pairs, strategy='fixed_beta', entry_multiplier=1, exit_multiplier=0,
                               test_mode=False, ai_support=False, train_val_split='2017-01-01'):
        """
        This function calls the kalman filter implementation for every pair.
        :param pairs: list with pairs identified in the training set information
        :param entry_multiplier: threshold that defines where to enter a position
        :param exit_multiplier: threshold that defines where to exit a position
        :param test_mode: flag to decide whether to apply strategy on the training set or in the test set
        :return: sharpe ratio results
        :return: cumulative returns
        """
        sharpe_results = []
        cum_returns = []
        sharpe_results_with_costs = []
        cum_returns_with_costs = []
        performance = []  # aux variable to store pairs' record
        print(' entry delay turned on.')
        for i, pair in enumerate(pairs):
            sys.stdout.write("\r"+'Pair: {}/{}'.format(i + 1, len(pairs)))
            sys.stdout.flush()
            pair_info = pair[2]

            if test_mode:
                y = pair_info['Y_test']
                x = pair_info['X_test']
            else:
                y = pair_info['Y_train'][train_val_split:]
                x = pair_info['X_train'][train_val_split:]

            if strategy == 'fixed_beta':

                if ai_support:
                    predictions = pair[2]['predictions']
                else:
                    predictions = None

                summary, sharpe, balance_summary = self.threshold_strategy(y=y, x=x, beta=pair_info['coint_coef'],
                                                                           entry_level=entry_multiplier,
                                                                           exit_level=exit_multiplier,
                                                                           predictions=predictions)
                # no costs
                cum_returns.append((np.cumprod(1 + summary.position_return) - 1).iloc[-1] * 100)
                sharpe_results.append(sharpe[0])
                # with costs
                # cum_returns_with_costs.append((np.cumprod(1 + summary.position_ret_with_costs) - 1).iloc[-1] * 100)
                cum_returns_with_costs.append((summary.account_balance[-1] - 1) * 100)
                sharpe_results_with_costs.append(sharpe[1])
                performance.append((pair, summary, balance_summary))

            elif strategy == 'kalman_filter':
                summary, ret_summary = self.kalman_filter(y=y, x=x,
                                                          entry_multiplier=entry_multiplier,
                                                          exit_multiplier=exit_multiplier)

                cum_returns.append((np.cumprod(1 + summary.position_return) - 1).iloc[-1] * 100)
                cum_returns_with_costs.append((np.cumprod(1 + summary.position_ret_with_costs) - 1).iloc[-1] * 100)
                sharpe_results, sharpe_results_with_costs = 0, 0
                performance.append((pair, summary))

            elif strategy == 'bollinger_bands':
                summary, ret_summary = self.bollinger_bands(y=y, x=x,
                                                          entry_multiplier=entry_multiplier,
                                                          exit_multiplier=exit_multiplier,
                                                          lookback=78*80)

                cum_returns.append((np.cumprod(1 + summary.position_return) - 1).iloc[-1] * 100)
                cum_returns_with_costs.append((np.cumprod(1 + summary.position_ret_with_costs) - 1).iloc[-1] * 100)
                print(' ',cum_returns_with_costs[-1])
                sharpe_results, sharpe_results_with_costs = 0, 0
                performance.append((pair, summary))

            else:
                print('3 strategies are available: \n1.Fixed Beta\n2.Bollinger Bands\n3.Kalman Filter')
                exit()

        return (sharpe_results, cum_returns), (sharpe_results_with_costs, cum_returns_with_costs), performance

    def kalman_filter(self, y, x, entry_multiplier=1.0, exit_multiplier=1.0, stabilizing_threshold=5):
        """
        This function implements a Kalman Filter for the estimation of
        the moving hedge ratio
        :param y:
        :param x:
        :param entry_multiplier:
        :param exit_multiplier:
        :param stabilizing_threshold:
        :return:
        """

        # store series for late usage
        x_series = x.copy()
        y_series = y.copy()

        # add constant
        x = x.to_frame()
        x['intercept'] = 1

        x = np.array(x)
        y = np.array(y)
        delta = 0.0001
        Ve = 0.001

        yhat = np.ones(len(y)) * np.nan
        e = np.ones(len(y)) * np.nan
        Q = np.ones(len(y)) * np.nan
        R = np.zeros((2, 2))
        P = np.zeros((2, 2))

        beta = np.matrix(np.zeros((2, len(y))) * np.nan)

        Vw = delta / (1 - delta) * np.eye(2)

        beta[:, 0] = 0.

        for t in range(len(y)):
            if (t > 0):
                beta[:, t] = beta[:, t - 1]
                R = P + Vw

            yhat[t] = np.dot(x[t, :], beta[:, t])

            tmp1 = np.matrix(x[t, :])
            tmp2 = np.matrix(x[t, :]).T
            Q[t] = np.dot(np.dot(tmp1, R), tmp2) + Ve

            e[t] = y[t] - yhat[t]  # plays spread role

            K = np.dot(R, np.matrix(x[t, :]).T) / Q[t]

            # print R;print x[t, :].T;print Q[t];print 'K',K;print;print

            beta[:, t] = beta[:, t] + np.dot(K, np.matrix(e[t]))

            tmp1 = np.matrix(x[t, :])
            P = R - np.dot(np.dot(K, tmp1), R)

        # if t==2:
        # print beta[0, :].T

        # plt.plot(beta[0, :].T)
        # plt.savefig('/tmp/beta1.png')
        # plt.hold(False)
        # plt.plot(beta[1, :].T)
        # plt.savefig('/tmp/beta2.png')
        # plt.hold(False)
        # plt.plot(e[2:], 'r')
        # plt.hold(True)
        # plt.plot(np.sqrt(Q[2:]))
        # plt.savefig('/tmp/Q.png')

        y2 = pd.concat([x_series, y_series], axis=1)

        longsEntry = e < -entry_multiplier * np.sqrt(Q)
        longsExit = e > -exit_multiplier * np.sqrt(Q)

        shortsEntry = e > entry_multiplier * np.sqrt(Q)
        shortsExit = e < exit_multiplier * np.sqrt(Q)

        numUnitsLong = pd.Series([np.nan for i in range(len(y))])
        numUnitsShort = pd.Series([np.nan for i in range(len(y))])
        # initialize with zero
        numUnitsLong[0] = 0.
        numUnitsShort[0] = 0.
        # remove trades while the spread is stabilizing
        longsEntry[:stabilizing_threshold] = False
        longsExit[:stabilizing_threshold] = False
        shortsEntry[:stabilizing_threshold] = False
        shortsExit[:stabilizing_threshold] = False

        numUnitsLong[longsEntry] = 1.
        numUnitsLong[longsExit] = 0
        numUnitsLong = numUnitsLong.fillna(method='ffill')

        numUnitsShort[shortsEntry] = -1.
        numUnitsShort[shortsExit] = 0
        numUnitsShort = numUnitsShort.fillna(method='ffill')

        numUnits = numUnitsLong + numUnitsShort
        numUnits = pd.Series(data=numUnits.values, index=y_series.index, name='numUnits')

        # position durations
        trading_durations = self.add_trading_duration(pd.DataFrame(numUnits, index=y_series.index))

        beta = pd.Series(data=np.squeeze(np.asarray(beta[0, :])), index=y_series.index).fillna(0)
        position_ret, _, ret_summary = self.calculate_sliding_position_returns(y_series, x_series, beta, numUnits)

        # add transaction costs and gather all info in df
        series_to_include = [(position_ret, 'position_return'),
                             (y_series, y_series.name),
                             (x_series, x_series.name),
                             (beta, 'beta_position'),
                             (pd.Series(e, index=y_series.index), 'e'),
                             (pd.Series(np.sqrt(Q), index=y_series.index), 'sqrt(Q)'),
                             (numUnits, 'numUnits'),
                             (trading_durations, 'trading_duration')]

        summary = self.trade_summary(series_to_include)

        return summary, ret_summary

    def bollinger_bands(self, y, x, lookback, entry_multiplier=1, exit_multiplier=0):
        """
        This function implements a pairs trading strategy based
        on bollinger bands.
        Source: Example 3.2 EC's book
        : Y & X: time series composing the spread
        : lookback : Lookback period
        : entry_multiplier : defines the multiple of std deviation used to enter a position
        : exit_multiplier: defines the multiple of std deviation used to exit a position
        """
        # print("Warning: don't forget lookback (halflife) must be at least 3.")

        entryZscore = entry_multiplier
        exitZscore = exit_multiplier

        # obtain zscore
        zscore, rolling_beta = self.rolling_zscore(y, x, lookback)
        zscore_array = np.asarray(zscore)

        # find long and short indices
        numUnitsLong = pd.Series([np.nan for i in range(len(y))])
        numUnitsLong.iloc[0] = 0.
        long_entries = self.cross_threshold(zscore_array, -entryZscore, 'down', 'entry')
        numUnitsLong[long_entries] = 1.0
        long_exits = self.cross_threshold(zscore_array, -exitZscore, 'up')
        numUnitsLong[long_exits] = 0.0
        numUnitsLong = numUnitsLong.fillna(method='ffill')
        numUnitsLong.index = zscore.index

        numUnitsShort = pd.Series([np.nan for i in range(len(y))])
        numUnitsShort.iloc[0] = 0.
        short_entries = self.cross_threshold(zscore_array, entryZscore, 'up', 'entry')
        numUnitsShort[short_entries] = -1.0
        short_exits = self.cross_threshold(zscore_array, exitZscore, 'down')
        numUnitsShort[short_exits] = 0.0
        numUnitsShort = numUnitsShort.fillna(method='ffill')
        numUnitsShort.index = zscore.index

        # concatenate all positions
        numUnits = numUnitsShort + numUnitsLong
        numUnits = pd.Series(data=numUnits.values, index=y.index, name='numUnits')

        # position durations
        trading_durations = self.add_trading_duration(pd.DataFrame(numUnits, index=y.index))

        beta = rolling_beta.copy()
        position_ret, _, ret_summary = self.calculate_sliding_position_returns(y, x, beta, numUnits)

        # get trade summary
        rolling_spread = y - rolling_beta * x

        # All series contain Date as index
        series_to_include = [(position_ret, 'position_return'),
                             (y, y.name),
                             (x, x.name),
                             (rolling_beta, 'beta_position'),
                             (rolling_spread, 'spread'),
                             (zscore, 'zscore'),
                             (numUnits, 'numUnits'),
                             (trading_durations, 'trading_duration')]
        summary = self.trade_summary(series_to_include)

        return summary, ret_summary

    def filter_profitable_pairs(self, sharpe_results, pairs):
        """
        This function discards pairs that were not profitable mantaining those for which a positive sharpe ratio was
        obtained.
        :param sharpe_results: list with sharpe resutls for every pair
        :param pairs: list with all pairs and their info
        :return: list with profitable pairs and their info
        """

        sharpe_results = np.asarray(sharpe_results)
        profitable_pairs_indices = np.argwhere(sharpe_results > 0)
        profitable_pairs = [pairs[i] for i in profitable_pairs_indices.flatten()]

        return profitable_pairs

    def trade_summary(self, series, beta=0):
        """
        This function receives a set of series containing information from the trade and
        returns a DataFrame containing the summary data.
        :param series: a list of tuples containing the time series and the corresponding names
        :return: summary dataframe with all series concatenated
        """
        for attribute, attribute_name in series:
            try:
                attribute.name = attribute_name
            except:
                continue

        summary = pd.concat([item[0] for item in series], axis=1)

        # change numUnits so that it corresponds to the position for the row's date,
        # instead of corresponding to the next position
        summary['numUnits'] = summary['numUnits'].shift().fillna(0)
        summary = summary.rename(columns={"numUnits": "position_during_day"})

        # add position costs
        summary['position_ret_with_costs'] = self.add_transaction_costs(summary, beta)
        # summary = summary.drop('position_return', axis=1)

        return summary

    def cross_threshold(self, array, threshold, direction='up', position='exit'):
        """
        This function returns the indices corresponding to the positions where a given threshold
        is crossed
        :param array: np.array with time series
        :param threshold: threshold to be crossed
        :param direction: going up or down
        :param mode: auxiliar variable indicating whether we are checking for a position entry or exit
        :return: indices where threshold is crossed going in the desired direction
        """

        # add index for first element transitioning from None value, in case its above/below threshold
        # only add when checking if position should be entered.
        initial_index = []
        first_index, first_element = next((item[0], item[1]) for item in enumerate(array) if not np.isnan(item[1]))
        if position == 'entry':
            if direction == 'up':
                if first_element > threshold:
                    initial_index.append(first_index)
            elif direction == 'down':
                if first_element < threshold:
                    initial_index.append(first_index)
            else:
                print('The series must be either going "up" or "down", please insert valid direction')
        initial_index = np.asarray(initial_index, dtype='int')

        # add small decimal case to consider only strictly larger/smaller
        if threshold > 0:
            threshold = threshold + 0.000000001
        else:
            threshold = threshold - 0.000000001
        array = array - threshold

        # add other indices
        indices = np.where(np.diff(np.sign(array)))[0] + 1
        # only consider indices after first element which is not Nan
        indices = indices[indices > first_index]

        direction_indices = indices
        for index in indices:
            if direction == 'up':
                if array[index] < array[index - 1]:
                    direction_indices = direction_indices[direction_indices != index]
            elif direction == 'down':
                if array[index] > array[index - 1]:
                    direction_indices = direction_indices[direction_indices != index]
        # concatenate
        direction_indices = np.concatenate((initial_index, direction_indices), axis=0)

        return direction_indices

    def apply_correlation_filter(self, lookback, lag, threshold, Y, X, units):
        """
        This function implements a filter proposed by Dunnis 2005.
        The main idea is tracking how the correlation is varying in a moving period, so that we
        are able to identify when the two legs of the spread are moving in opposing directions
        by analyzing how the correlation values are varying.
        :param lookback: lookback period
        :param lag: lag to compare the correlaiton evolution
        :param threshold: minimium difference to consider change
        :param Y: Y series
        :param X: X series
        :param units: positions taken
        :return: indices for position entry
        """

        # calculate correlation variations
        rolling_window = lookback
        returns_X = X.pct_change()
        returns_Y = Y.pct_change()
        correlation = returns_X.rolling(rolling_window).corr(returns_Y)
        diff_correlation = correlation.diff(periods=lag).fillna(0)

        # change positions accordingly
        diff_correlation.name = 'diff_correlation';
        units.name = 'units'
        units.index = diff_correlation.index
        df = pd.concat([diff_correlation, units], axis=1)
        new_df = self.update_positions(df, 'diff_correlation', threshold)

        units = new_df['units']

        return units

    def apply_zscorediff_filter(self, lag, threshold, zscore, units):
        """
        This function implements a filter which tracks how the zscore has been growing.
        The premise is that positions should not be entered while zscore is rising.
        :param lookback: lookback period
        :param lag: lag to compare the zscore evolution
        :param threshold: minimium difference to consider change
        :param Y: Y series
        :param X: X series
        :param units: positions taken
        :return: indices for position entry
        """

        # calculate zscore differences
        zscore_diff = zscore.diff(periods=lag).fillna(0)

        # change positions accordingly
        zscore_diff.name = 'zscore_diff';
        units.name = 'units'
        units.index = zscore_diff.index
        df = pd.concat([zscore_diff, units], axis=1)
        new_df = self.update_positions(df, 'zscore_diff', threshold)

        units = new_df['units']

        return units

    def update_positions(self, df, attribute, threshold):
        """
        The following function receives a dataframe containing the current positions
        along with the attribute column from which condition should be verified.
        A new df with positions updated accordingly is returned.
        :param df: df containing positions and column with attribute
        :param attribute: attribute name
        :param threshold: threshold that condition must verify
        :return: df with updated positions
        """
        previous_unit = 0
        for index, row in df.iterrows():
            if previous_unit == row['units']:
                continue  # no change in positions to verify
            else:
                if row['units'] == 0:
                    previous_unit = row['units']
                    continue  # simply close trade, nothing to verify
                else:
                    if (row[attribute] <= threshold and row['units'] < 0) or \
                       (row[attribute] > threshold and row['units'] > 0): # if criteria is met, continue
                        previous_unit = row['units']
                        continue
                    else:  # if criteria is not met, update row
                        df.loc[index, 'units'] = 0
                        previous_unit = 0
                        continue

        return df

    def add_trading_duration(self, df):
        """
        The following function adds a column containing the info concerning the last position
        returns
        :param df: Dataframe containing column with positions to enter in next day
        :return: df with extra column providing return information for each position
        """

        df['trading_duration'] = [0] * len(df)
        previous_unit = 0.
        new_position_counter = 0
        day = df.index[0].day
        for index, row in df.iterrows():
            if previous_unit == row['numUnits']:
                if previous_unit != 0.:
                    # update counter
                    if index.day != day:
                        new_position_counter += 1
                        day = index.day
                continue  # no change in positions to verify
            else:
                if previous_unit == 0.:
                    previous_unit = row['numUnits']
                    # begin counter
                    new_position_counter = 1
                    day = index.day
                    continue  # simply start the trade
                else:
                    df.loc[index, 'trading_duration'] = new_position_counter
                    previous_unit = row['numUnits']
                    # begin counter
                    new_position_counter = 1
                    day = index.day
                    continue

        return df['trading_duration']

    def add_transaction_costs(self, summary, beta=0, comission_costs=0.08, market_impact=0.2, short_rental=1):
        """
        Function to add transaction costs.
        :param summary: dataframe containing summary of all transactions
        :param comission_costs: commision costs, in percentage, per security, per trade
        :param market_impact: market impact costs, in percentage, per security, per trade
        :param short_rental: short rental costs, in annual percentage
        :return: series with returns after costs
        """
        fixed_costs_per_trade = (comission_costs + market_impact) / 100  # remove percentage
        short_costs_per_day = (short_rental / 252) / 100  # remove percentage

        costs = summary.apply(lambda row: self.apply_costs(row, fixed_costs_per_trade, short_costs_per_day, beta),
                              axis=1)

        ret_with_costs = summary['position_return'] - costs

        return ret_with_costs

    def apply_costs(self, row, fixed_costs_per_trade, short_costs_per_day, beta=0):

        if beta == 0:
            beta = row['beta_position']

        if row['position_during_day'] == 1. and row['trading_duration'] != 0:
            if beta >= 1:
                return fixed_costs_per_trade * (1 / beta) + fixed_costs_per_trade + short_costs_per_day * \
                       row['trading_duration']
            elif beta < 1:
                return fixed_costs_per_trade * beta + fixed_costs_per_trade + short_costs_per_day * \
                       row['trading_duration'] * beta

        elif row['position_during_day'] == -1. and row['trading_duration'] != 0:
            if beta >= 1:
                return fixed_costs_per_trade * (1 / beta) + fixed_costs_per_trade + short_costs_per_day * \
                       row['trading_duration'] * (1 / beta)
            elif beta < 1:
                return fixed_costs_per_trade * beta + fixed_costs_per_trade + short_costs_per_day * \
                       row['trading_duration']
        else:
            return 0

    def calculate_balance(self, y, x, beta, positions, trading_durations):
        """
        Function to calculate balance during a trading session.

        :param y: y series
        :param x: x series
        :param beta: pair's cointegration coefficient
        :param positions: position during the current day
        :param trading_durations: series with trading duration of each trade
        :return: balance dataframe containing summary info
        """
        y_returns = y.pct_change().fillna(0) * positions
        x_returns = -x.pct_change().fillna(0) * positions

        leg_y = [np.nan] * len(y)  # initial balance
        leg_x = [np.nan] * len(y)  # initial balance
        pnl_y = [np.nan] * len(y)
        pnl_x = [np.nan] * len(y)
        account_balance = [np.nan] * len(y)

        # auxiliary series to indicate beginning and end of position
        new_positions_idx = positions.diff()[positions.diff() != 0].index.values
        end_positions_idx = trading_durations[trading_durations != 0].index.values
        position_trigger = pd.Series([0] * len(y), index=y.index, name='position_trigger')
        # 2: new position
        # 1: new position which only lasts one day
        # -1: end of position that did not start on that day
        position_trigger[new_positions_idx] = 2.
        position_trigger[end_positions_idx] = position_trigger[end_positions_idx] - 1.
        position_trigger = position_trigger * positions.abs()
        position_trigger.name = 'position_trigger'

        for i in range(len(y)):
            if i == 0:
                pnl_y[0] = 0
                pnl_x[0] = 0
                account_balance[0] = 1
                if beta > 1:
                    leg_y[0] = 1 / beta
                    leg_x[0] = 1
                else:
                    leg_y[0] = 1
                    leg_x[0] = beta
            elif positions[i] == 0:
                pnl_y[i] = 0
                pnl_x[i] = 0
                leg_y[i] = leg_y[i - 1]
                leg_x[i] = leg_x[i - 1]
                account_balance[i] = account_balance[i-1]
            else:
                # add costs
                if position_trigger[i] == 1:
                    # every new position invest initial 1$ + acc in X + acc in Y
                    position_investment = account_balance[i-1]
                    # if new position, that most legs contain now the overall invested
                    if beta > 1:
                        pnl_y[i] = y_returns[i] * position_investment * (1 / beta)
                        pnl_x[i] = x_returns[i] * position_investment
                    else:
                        pnl_y[i] = y_returns[i] * position_investment
                        pnl_x[i] = x_returns[i] * position_investment * beta

                    # update legs
                    if beta > 1:
                        if positions[i] == 1:
                            leg_y[i] = position_investment * (1 / beta) + pnl_y[i]
                            leg_x[i] = position_investment - pnl_x[i]
                        else:
                            leg_y[i] = position_investment * (1 / beta) - pnl_y[i]
                            leg_x[i] = position_investment + pnl_x[i]
                    else:
                        if positions[i] == 1:
                            leg_y[i] = position_investment + pnl_y[i]
                            leg_x[i] = position_investment * beta - pnl_x[i]
                        else:
                            leg_y[i] = position_investment - pnl_y[i]
                            leg_x[i] = position_investment * beta + pnl_x[i]

                    # commission costs + market impact costs + short rental costs
                    if beta >= 1:
                        pnl_y[i] = pnl_y[i] - 0.0028*(1/beta)*position_investment # add commission + bid ask spread
                        pnl_x[i] = pnl_x[i] - 0.0028*position_investment # add commission + bid ask spread
                        if positions[i] == 1:
                            pnl_x[i] = pnl_x[i] - 1 * (0.01 / 252)*position_investment
                        elif positions[i] == -1:
                            pnl_y[i] = pnl_y[i] - 1 * (0.01 / 252)*(1/beta)*position_investment
                    elif beta < 1:
                        pnl_y[i] = pnl_y[i] - 0.0028 * position_investment  # add commission + bid ask spread
                        pnl_x[i] = pnl_x[i] - 0.0028 * beta * position_investment  # add commission + bid ask spread
                        if positions[i] == 1:
                            pnl_x[i] = pnl_x[i] - 1 * (0.01 / 252)*beta*position_investment
                        elif positions[i] == -1:
                            pnl_y[i] = pnl_y[i] - 1 * (0.01 / 252)*position_investment
                    # update balance
                    account_balance[i] = account_balance[i-1] + pnl_x[i] + pnl_y[i]

                elif position_trigger[i] == 2:
                    # every new position invest initial 1$ + acc in X + acc in Y
                    position_investment = account_balance[i-1]
                    # if new position, that most legs contain now the overall invested
                    if beta > 1:
                        pnl_y[i] = y_returns[i] * position_investment * (1 / beta)
                        pnl_x[i] = x_returns[i] * position_investment
                    else:
                        pnl_y[i] = y_returns[i] * position_investment
                        pnl_x[i] = x_returns[i] * position_investment * beta

                    # update legs
                    if beta > 1:
                        if positions[i] == 1:
                            leg_y[i] = position_investment * (1 / beta) + pnl_y[i]
                            leg_x[i] = position_investment - pnl_x[i]
                        else:
                            leg_y[i] = position_investment * (1 / beta) - pnl_y[i]
                            leg_x[i] = position_investment + pnl_x[i]
                    else:
                        if positions[i] == 1:
                            leg_y[i] = position_investment + pnl_y[i]
                            leg_x[i] = position_investment * beta - pnl_x[i]
                        else:
                            leg_y[i] = position_investment - pnl_y[i]
                            leg_x[i] = position_investment * beta + pnl_x[i]

                    # commission costs + market impact costs + short rental costs
                    if beta >= 1:
                        pnl_y[i] = pnl_y[i] - 0.0028*(1/beta)*position_investment # add commission + bid ask spread
                        pnl_x[i] = pnl_x[i] - 0.0028*position_investment # add commission + bid ask spread
                    elif beta < 1:
                        pnl_y[i] = pnl_y[i] - 0.0028 * position_investment  # add commission + bid ask spread
                        pnl_x[i] = pnl_x[i] - 0.0028 * beta * position_investment  # add commission + bid ask spread
                    # update balance
                    account_balance[i] = account_balance[i - 1] + pnl_x[i] + pnl_y[i]

                else:
                    # calculate trade pnl
                    pnl_y[i] = y_returns[i] * leg_y[i - 1]
                    pnl_x[i] = x_returns[i] * leg_x[i - 1]

                    # update legs
                    if positions[i] == 1:
                        leg_y[i] = leg_y[i - 1] + pnl_y[i]
                        leg_x[i] = leg_x[i - 1] - pnl_x[i]
                    else:
                        leg_y[i] = leg_y[i - 1] - pnl_y[i]
                        leg_x[i] = leg_x[i - 1] + pnl_x[i]

                    # add short costs
                    if position_trigger[i] == -1:
                        if positions[i]==1:
                            if beta > 1:
                                pnl_x[i] = pnl_x[i] - trading_durations[i] * (0.01 / 252) * position_investment
                            elif beta < 1:
                                pnl_x[i] = pnl_x[i] - trading_durations[i] * (0.01 / 252)*beta*position_investment
                        elif positions[i]==-1:
                            if beta > 1:
                                pnl_y[i] = pnl_y[i] - trading_durations[i] * (0.01 / 252)*(1/beta)*position_investment
                            elif beta < 1:
                                pnl_y[i] = pnl_y[i] - trading_durations[i] * (0.01 / 252) * position_investment

                    # update balance
                    account_balance[i] = account_balance[i - 1] + pnl_x[i] + pnl_y[i]
        pnl = [pnl_y[i] + pnl_x[i] for i in range(len(y))]

        # join everything in dataframe
        balance = pd.Series(data=account_balance, index=y.index, name='account_balance')
        returns = balance.pct_change().fillna(0)
        returns.name = 'returns'
        pnl = pd.Series(data=pnl, index=y.index, name='pnl')
        pnl_y = pd.Series(data=pnl_y, index=y.index, name='pnl_y')
        pnl_x = pd.Series(data=pnl_x, index=y.index, name='pnl_x')
        leg_y = pd.Series(data=leg_y, index=y.index, name='leg_y')
        leg_x = pd.Series(data=leg_x, index=y.index, name='leg_x')
        balance_summary = pd.concat(
                [balance, pnl, pnl_y, pnl_x, leg_y, leg_x, returns, position_trigger, positions, y, x,
                 trading_durations], axis=1)

        return balance_summary

    def calculate_sharpe_ratio(self, n_years, n_days, ret):
        """

        :param n_years: number of years being considered
        :param n_days: number of trading days per year
        :param ret: array containing returns per timestep
        :return: sharpe ratio
        """
        rf = {2015: 0.0005, 2016: 0.0032, 2017: 0.0093, 2018: 0.0194}
        time_in_market = n_years * n_days
        daily_index = ret.resample('D').last().dropna().index
        daily_ret = (ret + 1).resample('D').prod() - 1
        # remove added days from resample
        daily_ret = daily_ret.loc[daily_index]

        annualized_ret = (np.cumprod(1 + ret) - 1)[-1]
        year = ret.index[0].year
        if year in rf.keys():
            sharpe_ratio = (annualized_ret-rf[year]) / (np.std(daily_ret)*np.sqrt(time_in_market))
        else:
            print('Not considering risk-free rate')
            sharpe_ratio = annualized_ret / (np.std(daily_ret)*np.sqrt(time_in_market))

        return sharpe_ratio

    def calculate_portfolio_sharpe_ratio(self, performance, pairs):
        """
        Calculates the sharpe ratio based on the account balance of the total portfolio

        :param performance: df with summary statistics from strategy
        :param pairs: list with pairs
        :return: sharpe ratio
        """
        # calculate total daily account balance & df with returns
        total_account_balance = performance[0][1]['account_balance'].resample('D').last().dropna()
        portfolio_returns = total_account_balance.pct_change().fillna(0)
        for index in range(1, len(pairs)):
            pair_balance = performance[index][1]['account_balance'].resample('D').last().dropna()
            total_account_balance = total_account_balance + pair_balance
            portfolio_returns = pd.concat([portfolio_returns, pair_balance.pct_change().fillna(0)], axis=1)

        # add first day with total balance
        total_account_balance = pd.Series(data=[len(pairs)],
                                          index=[total_account_balance.index[0] - timedelta(days=1)]).append(
                                          total_account_balance)

        # calculate portfolio volatility
        weights = np.array([1 / len(pairs)] * len(pairs))
        vol = np.sqrt(np.dot(weights.T, np.dot(portfolio_returns.cov(), weights)))
        # calculate sharpe ratio
        rf = {2015: 0.00053, 2016: 0.0032, 2017: 0.0093, 2018: 0.0194}
        annualized_ret = (total_account_balance[-1]-len(pairs))/len(pairs)
        year = total_account_balance.index[-1].year
        if year in rf.keys():
            sharpe_ratio = (annualized_ret - rf[year]) / (vol*np.sqrt(252))
        else:
            print('Not considering risk-free rate')
            sharpe_ratio = annualized_ret / (vol*np.sqrt(252))

        return sharpe_ratio

    def calculate_maximum_drawdown(self, account_balance):
        """
        source: https://stackoverflow.com/questions/22607324/start-end-and-duration-of-maximum-drawdown-in-python
        """

        xs = np.asarray(account_balance.values)

        i = np.argmax(np.maximum.accumulate(xs) - xs)  # end of the period
        if i == 0:
            plt.plot(xs)
            return 0
        else:
            j = np.argmax(xs[:i])  # start of period
            plt.plot(xs)
            plt.plot([i, j], [xs[i], xs[j]], 'o', color='Red', markersize=10)

        #print(xs[i])
        #print(xs[j])

        return (xs[i]-xs[j])/xs[j] * 100

    def calculate_position_returns(self, y, x, beta, positions):
        """
        Y: price of ETF Y
        X: price of ETF X
        beta: cointegration ratio
        positions: array indicating position to enter in next day
        """
        # get copy of series
        y = y.copy()
        y.name = 'y'
        x = x.copy()
        x.name = 'x'

        # positions preceed the day when the position is actually entered!
        # get indices before entering position
        new_positions = positions.diff()[positions.diff() != 0].index.values
        # create variable for signalizing end of position
        end_position = pd.Series(data=[0] * len(y), index=y.index, name='end_position')
        end_position[new_positions] = 1.

        # get corresponding X and Y
        y_entry = pd.Series(data=[np.nan] * len(y), index=y.index, name='y_entry')
        x_entry = pd.Series(data=[np.nan] * len(y), index=y.index, name='x_entry')
        y_entry[new_positions] = y[new_positions]
        x_entry[new_positions] = x[new_positions]
        y_entry = y_entry.shift().fillna(method='ffill')
        x_entry = x_entry.shift().fillna(method='ffill')

        # name positions series
        positions.name = 'positions'

        # apply returns per trade
        # each row contain all the parameters to be applied in that position
        df = pd.concat([y, x, positions.shift().fillna(0), y_entry, x_entry, end_position], axis=1)
        returns = df.apply(lambda row: self.return_per_position(row, beta), axis=1).fillna(0)
        cum_returns = np.cumprod(returns + 1) - 1
        df['ret'] = returns
        returns.name = 'position_return'

        return returns, cum_returns, df

    def calculate_sliding_position_returns(self, y, x, beta, positions):
        """
        Y: price of ETF Y
        X: price of ETF X
        beta: moving cointegration ratio
        positions: array indicating position to enter in next day
        """
        # get copy of series
        y = y.copy()
        y.name = 'y'
        x = x.copy()
        x.name = 'x'

        # positions preceed the day when the position is actually entered!
        # get indices before entering position
        new_positions = positions.diff()[positions.diff() != 0].index.values
        # get corresponding betas
        beta_position = pd.Series(data=[np.nan] * len(y), index=y.index, name='beta_position')
        beta_position[new_positions] = beta[new_positions]
        # fill in between time slots with same beta
        beta_position = beta_position.fillna(method='ffill')
        # shift betas to match row when position is on
        beta_position = beta_position.shift().fillna(0)

        # create variable for signalizing end of position
        end_position = pd.Series(data=[0] * len(y), index=y.index, name='end_position')
        end_position[new_positions] = 1.

        # get corresponding X and Y
        y_entry = pd.Series(data=[np.nan] * len(y), index=y.index, name='y_entry')
        x_entry = pd.Series(data=[np.nan] * len(y), index=y.index, name='x_entry')
        y_entry[new_positions] = y[new_positions]
        x_entry[new_positions] = x[new_positions]
        y_entry = y_entry.shift().fillna(method='ffill')
        x_entry = x_entry.shift().fillna(method='ffill')

        # name positions series
        positions.name = 'positions'

        # apply returns per trade
        # each row contain all the parameters to be applied in that position
        df = pd.concat([y, x, beta_position, positions.shift().fillna(0), y_entry, x_entry, end_position], axis=1)
        returns = df.apply(lambda row: self.return_per_position(row, sliding=True), axis=1).fillna(0)
        cum_returns = np.cumprod(returns + 1) - 1
        df['ret'] = returns
        returns.name = 'position_return'

        return returns, cum_returns, df

    def return_per_position(self, row, beta=None, sliding=False):
        if row['end_position'] != 0:
            y_returns = (row['y'] - row['y_entry']) / row['y_entry']
            x_returns = (row['x'] - row['x_entry']) / row['x_entry']
            if sliding:
                beta = row['beta_position']
            if beta > 1.:
                return ((1 / beta) * y_returns - 1 * x_returns) * row['positions']
            else:
                return (y_returns - beta * x_returns) * row['positions']
        else:
            return 0

    def return_per_timestep(self, row):
        if row['beta_position'] > 1.:
            return ((1 / row['beta_position']) * row['y_returns'] - 1 * row['x_returns']) * row['positions']
        else:
            return (row['y_returns'] - row['beta_position'] * row['x_returns']) * row['positions']

    def calculate_metrics(self, sharpe_results, cum_returns, n_years):
        """
        Calculate common metrics on average over all pairs.
        :param sharpe_results: array with sharpe result of every pair
        :param cum_returns: array with cumulative returns of every pair
        :param n_years: numbers of yers of the trading strategy
        :return: average sharpe ratio
        :return: average average total roi
        :return: average annual roi
        :return: percentage of pairs with positive returns
        """
        # use below for fully invested capital
        # sharpe_results_filtered = [sharpe for sharpe in sharpe_results if sharpe != 0]
        # cum_returns_filtered = [cum for cum in cum_returns if cum != 0]
        # use below for commited capital
        sharpe_results_filtered = sharpe_results
        cum_returns_filtered = cum_returns

        avg_sharpe_ratio = np.mean(sharpe_results_filtered)
        print('Average result: ', avg_sharpe_ratio)

        avg_total_roi = np.mean(cum_returns_filtered)
        #print('avg_total_roi: ', avg_total_roi)

        avg_annual_roi = ((1 + (avg_total_roi / 100)) ** (1 / float(n_years)) - 1) * 100
        print('avg_annual_roi: ', avg_annual_roi)

        cum_returns_filtered = np.asarray(cum_returns_filtered)
        positive_pct = len(cum_returns_filtered[cum_returns_filtered > 0]) * 100 / len(cum_returns_filtered)
        print('{} % of the pairs had positive returns'.format(positive_pct))

        return avg_sharpe_ratio, avg_total_roi, avg_annual_roi, positive_pct

    def summarize_results(self, sharpe_results, cum_returns, performance, total_pairs, ticker_segment_dict, n_years):
        """
        This function summarizes interesting metrics to include in the final output
        :param sharpe_results: array containing sharpe results for each pair
        :param cum_returns: array containing cum returns for each pair
        :param performance: df containing a summary of each pair's trade
        :param total_pairs: list containing all the identified pairs
        :param ticker_segment_dict: dict containing segment for each ticker
        :return: dictionary with metrics of interest
        """

        avg_sharpe_ratio, avg_total_roi, avg_annual_roi, positive_pct = \
            self.calculate_metrics(sharpe_results, cum_returns, n_years)

        sorted_indices = np.flip(np.argsort(sharpe_results), axis=0)
        # print(sorted_indices)
        # initialize list of lists
        data = []
        for index in sorted_indices:
            # get number of positive and negative positions
            position_returns = performance[index][1]['position_return']
            positive_positions = len(position_returns[position_returns > 0])
            negative_positions = len(position_returns[position_returns < 0])
            data.append([total_pairs[index][0],
                         ticker_segment_dict[total_pairs[index][0]],
                         total_pairs[index][1],
                         ticker_segment_dict[total_pairs[index][1]],
                         total_pairs[index][2]['t_statistic'],
                         total_pairs[index][2]['p_value'],
                         total_pairs[index][2]['zero_cross'],
                         total_pairs[index][2]['half_life'],
                         total_pairs[index][2]['hurst_exponent'],
                         positive_positions,
                         negative_positions,
                         sharpe_results[index]
                         ])

        # Create the pandas DataFrame
        pairs_df = pd.DataFrame(data, columns=['Leg1', 'Leg1_Segmt', 'Leg2', 'Leg2_Segmt', 't_statistic', 'p_value',
                                               'zero_cross', 'half_life', 'hurst_exponent', 'positive_trades',
                                               'negative_trades', 'sharpe_result'])

        pairs_df['positive_trades_per_pair_pct'] = (pairs_df['positive_trades']) / \
                                                   (pairs_df['positive_trades'] + pairs_df['negative_trades']) * 100
        avg_positive_trades_per_pair_pct = pairs_df['positive_trades_per_pair_pct'].mean()

        results = {'n_pairs': len(sharpe_results),
                   'avg_sharpe_ratio': avg_sharpe_ratio,
                   'avg_total_roi': avg_total_roi,
                   'avg_annual_roi': avg_annual_roi,
                   'pct_positive_trades_per_pair': avg_positive_trades_per_pair_pct,
                   'pct_pairs_with_positive_results': positive_pct,
                   'avg_half_life': pairs_df['half_life'].mean(),
                   'avg_hurst_exponent': pairs_df['hurst_exponent'].mean()}

        return results, pairs_df