import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

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
                X_slice = X.iloc[i-window:i,:] # always index in np as opposed to pandas, much faster
                y_slice = y.iloc[i-window:i]
                coeff = sm.OLS(y_slice, X_slice).fit()
                estimate_data.append(coeff.params[X_name])

            # Assemble
            estimate = pd.Series(data=np.nan, index=x.index[:window])
            # add nan values for first #lookback indices
            estimate = estimate.append(pd.Series(data=estimate_data, index=x.index[window:]) )
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
        rolling_spread = Y - rolling_beta*X
        # 3.moving average
        rolling_avg = rolling_spread.rolling(window=lookback,center=False).mean()
        rolling_avg.name = 'spread_' + str(lookback) + 'mavg'
        # 4. rolling standard deviation
        rolling_std = rolling_spread.rolling(window=lookback, center=False).std()
        rolling_std.name = 'rolling_std_' + str(lookback)

        # z-score
        zscore = (rolling_spread - rolling_avg)/rolling_std

        return zscore, rolling_beta

    def kalman_filter(self, y, x, entry_multiplier=1.0, exit_multiplier=1.0, stabilizing_threshold=5,
                      trading_filter=None, rebalance=False):
        """
        This function implements a Kalman Filter for the estimation of
        the moving hedge ratio

        :param y:
        :param x:
        :param entry_multiplier:
        :param exit_multiplier:
        :param stabilizing_threshold:
        :param trading_filter:
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
        delta=0.0001
        Ve=0.001

        yhat = np.ones(len(y))*np.nan
        e = np.ones(len(y))*np.nan
        Q = np.ones(len(y))*np.nan
        R = np.zeros((2,2))
        P = np.zeros((2,2))

        beta = np.matrix(np.zeros((2,len(y)))*np.nan)

        Vw=delta/(1-delta)*np.eye(2)

        beta[:, 0]=0.

        for t in range(len(y)):
            if (t > 0):
                beta[:, t]=beta[:, t-1]
                R=P+Vw

            yhat[t]=np.dot(x[t, :],beta[:, t])

            tmp1 = np.matrix(x[t, :])
            tmp2 = np.matrix(x[t, :]).T
            Q[t] = np.dot(np.dot(tmp1,R),tmp2) + Ve

            e[t]=y[t]-yhat[t] # plays spread role

            K=np.dot(R,np.matrix(x[t, :]).T) / Q[t]

            #print R;print x[t, :].T;print Q[t];print 'K',K;print;print

            beta[:, t]=beta[:, t]+np.dot(K,np.matrix(e[t]))

            tmp1 = np.matrix(x[t, :])
            P=R-np.dot(np.dot(K,tmp1),R)

        #if t==2:
        #print beta[0, :].T

        #plt.plot(beta[0, :].T)
        #plt.savefig('/tmp/beta1.png')
        #plt.hold(False)
        #plt.plot(beta[1, :].T)
        #plt.savefig('/tmp/beta2.png')
        #plt.hold(False)
        #plt.plot(e[2:], 'r')
        #plt.hold(True)
        #plt.plot(np.sqrt(Q[2:]))
        #plt.savefig('/tmp/Q.png')

        y2 = pd.concat([x_series,y_series], axis = 1)

        longsEntry=e < -entry_multiplier*np.sqrt(Q)
        longsExit=e > -exit_multiplier*np.sqrt(Q)

        shortsEntry=e > entry_multiplier*np.sqrt(Q)
        shortsExit=e < exit_multiplier*np.sqrt(Q)

        numUnitsLong = pd.Series([np.nan for i in range(len(y))])
        numUnitsShort = pd.Series([np.nan for i in range(len(y))])
        # initialize with zero
        numUnitsLong[0]=0.
        numUnitsShort[0]=0.
        # remove trades while the spread is stabilizing
        longsEntry[:stabilizing_threshold] = False
        longsExit[:stabilizing_threshold] = False
        shortsEntry[:stabilizing_threshold] = False
        shortsExit[:stabilizing_threshold] = False

        numUnitsLong[longsEntry]=1.
        numUnitsLong[longsExit]=0
        numUnitsLong = numUnitsLong.fillna(method='ffill')

        numUnitsShort[shortsEntry]=-1.
        numUnitsShort[shortsExit]=0
        numUnitsShort = numUnitsShort.fillna(method='ffill')

        numUnits=numUnitsLong+numUnitsShort

        # apply trading filter
        if trading_filter is not None:
            if trading_filter['name'] == 'correlation':
                numUnits = self.apply_correlation_filter(lookback=trading_filter['lookback'],
                                                         lag=trading_filter['lag'],
                                                         threshold=trading_filter['diff_threshold'],
                                                         Y=y_series,
                                                         X=x_series,
                                                         units=numUnits
                                                         )
            elif trading_filter['name'] == 'zscore_diff':
                numUnits = self.apply_zscorediff_filter(lag=trading_filter['lag'],
                                                        threshold=trading_filter['diff_threshold'],
                                                        zscore=zscore,
                                                        units=numUnits
                                                        )

        #tmp1 = np.tile(np.matrix(numUnits).T, 2)
        #tmp2 = np.hstack((-1*beta[0, :].T, np.ones((len(y),1))))
        #positions = np.array(tmp1)*np.array(tmp2)*y2
        #positions = pd.DataFrame(positions)
        #tmp1 = np.array(positions.shift(1))
        #tmp2 = np.array(y2-y2.shift(1))
        #tmp3 = np.array(y2.shift(1))
        #pnl = np.sum(tmp1 * tmp2 / tmp3,axis=1)
        #ret = pnl / np.sum(np.abs(positions.shift(1)),axis=1)
        #ret = ret.fillna(0)
        #ret = ret.dropna()

        numUnits = pd.Series(data=numUnits.values, index=y_series.index)
        beta = pd.Series(data=np.squeeze(np.asarray(beta[0, :])), index=y_series.index).fillna(0)
        if not rebalance:
            ret, _ = self.calculate_position_returns_no_rebalance(y_series, x_series, beta, numUnits)
        else:
            print('WARNING: COSTS ARE NOT ADJUSTED FOR DAILY REBALANCING, THIS MUST BE REVISED')
            ret, _ = self.calculate_returns_adapted(y_series, x_series, beta, numUnits.shift(1).fillna(0))

        # add transaction costs and gather all info in df
        series_to_include = [(ret, 'ret'),
                             (y_series, y_series.name),
                             (x_series, x_series.name),
                             (beta, 'beta'),
                             (pd.Series(e, index=y_series.index), 'e'),
                             (pd.Series(np.sqrt(Q), index=y_series.index), 'sqrt(Q)'),
                             (numUnits, 'numUnits')]

        summary = self.trade_summary(series_to_include)

        # calculate sharpe ratio
        n_years = round(len(y)/240)
        time_in_market = 252.*n_years
        # apr = ((np.prod(1.+ret))**(time_in_market/len(ret)))-1
        if np.std(ret) == 0:
            sharpe = 0
        else:
            sharpe = np.sqrt(time_in_market) * np.mean(ret) / np.std(ret)

        return summary, sharpe

    def apply_kalman_strategy(self, pairs, entry_multiplier=1, exit_multiplier=0, trading_filter=None,
                              test_mode=False, rebalance=False):
        """
        This function calls the kalman filter implementation for every pair.

        :param pairs: list with pairs identified in the training set information
        :param entry_multiplier: threshold that defines where to enter a position
        :param exit_multiplier: threshold that defines where to exit a position
        :param trading_filter:  trading_flter dictionary with parameters or None object in case of no filter
        :param test_mode: flag to decide whether to apply strategy on the training set or in the test set

        :return: sharpe ratio results
        :return: cumulative returns
        """
        sharpe_results = []
        cum_returns = []
        sharpe_results_with_costs = []
        cum_returns_with_costs = []
        performance = []  # aux variable to store pairs' record
        for i,pair in enumerate(pairs):
            # start = time.time()
            #end = time.time()
            #print((end - start))
            print('Pair: {}/{}'.format(i+1, len(pairs)))
            pair_info = pair[2]
            if trading_filter is not None:
                trading_filter['lookback'] = min(trading_filter['filter_lookback_multiplier']*(pair_info['half_life']),
                                                 20)

            if test_mode:
                y = pair_info['Y_test']
                x = pair_info['X_test']
            else:
                y = pair_info['Y_train']
                x = pair_info['X_train']
            summary, sharpe = self.kalman_filter(y=y, x=x,
                                                 entry_multiplier=entry_multiplier,
                                                 exit_multiplier=exit_multiplier,
                                                 trading_filter=trading_filter,
                                                 rebalance=rebalance)
            # no costs
            cum_returns.append((np.cumprod(1 + summary.position_return) - 1).iloc[-1] * 100)
            sharpe_results.append(sharpe)
            # with costs
            cum_returns_with_costs.append((np.cumprod(1 + summary.position_ret_with_costs) - 1).iloc[-1] * 100)
            sharpe_results_with_costs = None
            performance.append((pair, summary))

        return (sharpe_results, cum_returns), (sharpe_results_with_costs, cum_returns_with_costs), performance

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

    def trade_summary(self, series):
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

        # add position returns
        summary = self.add_position_returns(summary)

        # change numUnits so that it corresponds to the position for the row's date,
        # instead of corresponding to the next position
        summary['numUnits'] = summary['numUnits'].shift().fillna(0)
        summary = summary.rename(columns={"numUnits": "position_during_day"})

        # add position costs
        summary['position_ret_with_costs'] = self.add_transaction_costs(summary)

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
        diff_correlation.name = 'diff_correlation'; units.name = 'units'
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
        zscore_diff.name = 'zscore_diff'; units.name = 'units'
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
                    if row[attribute] <= threshold:  # if criteria is met, continue
                        previous_unit = row['units']
                        continue
                    elif row[attribute] > threshold:  # if criteria is not met, update row
                        df.loc[index, 'units'] = 0
                        previous_unit = row['units']
                        continue

        return df

    def add_position_returns(self, df):
        """
        The following function adds a column containing the info concerning the last position
        returns

        :param df: Dataframe containing the trading summary
        :return: df with extra column providing return information for each position
        """

        df['position_return'] = 0
        df['trade_duration'] = 0
        previous_unit = 0.
        position_ret_acc = 1.
        new_position_counter = 0
        day = df.index[0].day
        for index, row in df.iterrows():
            if previous_unit == row['numUnits']:
                if previous_unit != 0.:
                    position_ret_acc = position_ret_acc * (1+row['ret'])
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
                    # update position returns
                    position_ret_acc = position_ret_acc * (1+row['ret'])
                    df.loc[index, 'position_return'] = (position_ret_acc-1)
                    df.loc[index, 'trade_duration'] = new_position_counter
                    position_ret_acc = 1.
                    previous_unit = row['numUnits']
                    # begin counter
                    new_position_counter = 1
                    day = index.day
                    continue

        return df

    def add_transaction_costs(self, summary, comission_costs=0.08, market_impact=0.2, short_rental=1):
        """
        Function to add transaction costs.

        :param summary: dataframe containing summary of all transactions
        :param comission_costs: commision costs, in percentage, per security, per trade
        :param market_impact: market impact costs, in percentage, per security, per trade
        :param short_rental: short rental costs, in annual percentage
        :return: series with returns after costs
        """
        fixed_costs_per_trade = (comission_costs*2 + market_impact*2)/100 # remove percentage
        short_costs_per_day = (short_rental / 252) / 100  # remove percentage

        costs = summary.apply(lambda row: self.apply_costs(row, fixed_costs_per_trade, short_costs_per_day), axis=1)

        ret_with_costs = summary['position_return']-costs

        return ret_with_costs

    def apply_costs(self, row, fixed_costs_per_trade, short_costs_per_day):
        if row['position_during_day'] == 1. and row['trade_duration']!= 0:
            return fixed_costs_per_trade + short_costs_per_day * row['trade_duration']
        elif row['position_during_day'] == -1. and row['trade_duration']!= 0:
            return fixed_costs_per_trade + short_costs_per_day * row['trade_duration']*(1/row['beta'])
        else:
            return 0

    def calculate_position_returns_no_rebalance(self, y, x, beta, positions):
        """
        Y: price of ETF Y
        X: price of ETF X
        beta: cointegration ratio
        positions: array indicating position to enter in next day
        """
        # get copy of series
        y = y.copy(); y.name = 'y'
        x = x.copy(); x.name = 'x'

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
        returns = df.apply(lambda row: self.return_per_position(row), axis=1).fillna(0)
        cum_returns = np.cumprod(returns + 1) - 1

        return returns, cum_returns

    def return_per_position(self, row):
        if row['end_position'] != 0:
            y_returns = (row['y']-row['y_entry'])/row['y_entry']
            x_returns = (row['x']-row['x_entry'])/row['x_entry']
            if row['beta_position'] > 1.:
                return ((1 / row['beta_position']) * y_returns - 1 * x_returns) * row['positions']
            else:
                return (y_returns - row['beta_position'] * x_returns) * row['positions']
        else:
            return 0

    def calculate_returns_adapted(self, y, x, beta, positions):
        """
        Y: price of ETF Y
        X: price of ETF X
        beta: cointegration ratio
        positions: array indicating when to take a position
        """
        # calculate each leg return
        y_returns = y.pct_change().fillna(0); y_returns.name = 'y_returns'
        x_returns = x.pct_change().fillna(0); x_returns.name = 'x_returns'

        # name positions series
        positions.name = 'positions'

        # beta must shift from row above
        beta_position = beta.shift().fillna(0)
        beta_position.name = 'beta_position'

        # apply returns per trade
        df = pd.concat([y_returns, x_returns, beta_position, positions], axis=1)
        returns = df.apply(lambda row: self.return_per_timestep(row), axis=1)
        cum_returns = np.cumprod(returns + 1) - 1

        return returns, cum_returns

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
        sharpe_results_filtered = [sharpe for sharpe in sharpe_results if sharpe != 0]
        cum_returns_filtered = [cum for cum in cum_returns if cum != 0]

        avg_sharpe_ratio = np.mean(sharpe_results_filtered)
        print('Average result: ', avg_sharpe_ratio)

        avg_total_roi = np.mean(cum_returns_filtered)
        #print('avg_total_roi: ', avg_total_roi)

        avg_annual_roi = ((1 + (avg_total_roi / 100)) ** (1 / float(n_years)) - 1) * 100
        print('avg_annual_roi: ', avg_annual_roi)

        sharpe_results_filtered = np.asarray(sharpe_results_filtered)
        positive_pct = len(sharpe_results_filtered[sharpe_results_filtered > 0]) * 100 / len(sharpe_results_filtered)
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

        #n_years = round(len(performance[0][1]) / 240)  # performance[0][1] contains time series index, thus true length
        avg_sharpe_ratio, avg_total_roi, avg_annual_roi, positive_pct = \
            self.calculate_metrics(sharpe_results, cum_returns, n_years)

        sorted_indices = np.flip(np.argsort(sharpe_results), axis=0)
        #print(sorted_indices)
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

        pairs_df['positive_trades_per_pair_pct'] = (pairs_df['positive_trades']) /\
                                                   (pairs_df['positive_trades']+pairs_df['negative_trades'])*100
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
