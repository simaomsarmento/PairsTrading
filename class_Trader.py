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

    def linear_strategy(self, Y, X, lookback):
        """
        This function applies a simple pairs trading strategy based on
        Ernie Chan's book: Algoritmic Trading.

        The number of shares for each position is set to be the negative
        z-score
        """

        # z-score
        zscore = self.rolling_zscore(Y, X, lookback)
        numUnits = -zscore

        # Define strategy
        # Multiply num positions inversely (-) proportionally to z-score
        # ATTENTION: in the book the signals are inverted. The author confirms it here:
        # http://epchan.blogspot.com/2013/05/my-new-book-on-algorithmic-trading-is.html
        X_positions = numUnits*(-rolling_beta*X)
        Y_positions = numUnits*Y

        # P&L:position (-spread value) * percentage of change
        # note that pnl is not a percentage. We multiply a position value by a percentage
        X_returns = (X - X.shift(periods=-1))/X.shift(periods=-1)
        Y_returns = (Y - Y.shift(periods=-1))/Y.shift(periods=-1)
        pnl = X_positions.shift(periods=-1)*X_returns + Y_positions.shift(periods=-1)*Y_returns
        total_pnl = (X_positions.shift(periods=-1)*(X - X.shift(periods=-1)) + \
                     Y_positions.shift(periods=-1)*(Y - Y.shift(periods=-1))).sum()
        ret=pnl/(abs(X_positions.shift(periods=-1))+abs(Y_positions).shift(periods=-1))

        return pnl, total_pnl, ret

    def bollinger_bands(self, Y, X, lookback, entry_multiplier=1, exit_multiplier=0, trading_filter=None):
        """
        This function implements a pairs trading strategy based
        on bollinger bands.
        Source: Example 3.2 EC's book
        : Y & X: time series composing the spread
        : lookback : Lookback period
        : entry_multiplier : defines the multiple of std deviation used to enter a position
        : exit_multiplier: defines the multiple of std deviation used to exit a position
        """
        #print("Warning: don't forget lookback (halflife) must be at least 3.")

        entryZscore = entry_multiplier
        exitZscore = exit_multiplier

        # obtain zscore
        zscore, rolling_beta = self.rolling_zscore(Y, X, lookback)
        zscore_array = np.asarray(zscore)

        # find long and short indices
        numUnitsLong = pd.Series([np.nan for i in range(len(Y))])
        numUnitsLong.iloc[0] = 0.
        long_entries = self.cross_threshold(zscore_array, -entryZscore, 'down', 'entry')
        numUnitsLong[long_entries] = 1.0
        long_exits = self.cross_threshold(zscore_array, -exitZscore, 'up')
        numUnitsLong[long_exits] = 0.0
        numUnitsLong = numUnitsLong.fillna(method='ffill')
        numUnitsLong.index = zscore.index

        numUnitsShort = pd.Series([np.nan for i in range(len(Y))])
        numUnitsShort.iloc[0] = 0.
        short_entries = self.cross_threshold(zscore_array, entryZscore, 'up', 'entry')
        numUnitsShort[short_entries] = -1.0
        short_exits = self.cross_threshold(zscore_array, exitZscore, 'down')
        numUnitsShort[short_exits] = 0.0
        numUnitsShort = numUnitsShort.fillna(method='ffill')
        numUnitsShort.index = zscore.index

        # concatenate all positions
        numUnits = numUnitsShort + numUnitsLong
        # apply trading filter
        if trading_filter is not None:
            if trading_filter['name'] == 'correlation':
                numUnits = self.apply_correlation_filter(lookback=trading_filter['lookback'],
                                                         lag=trading_filter['lag'],
                                                         threshold=trading_filter['diff_threshold'],
                                                         Y=Y,
                                                         X=X,
                                                         units=numUnits
                                                       )
            elif trading_filter['name'] == 'zscore_diff':
                numUnits = self.apply_zscorediff_filter(lag=trading_filter['lag'],
                                                        threshold=trading_filter['diff_threshold'],
                                                        zscore=zscore,
                                                        units=numUnits
                                                        )

        # define positions according to cointegration equation
        X_positions = (numUnits*(-rolling_beta*X)).fillna(0)
        Y_positions = (numUnits*Y)
        # discard positions up to window size
        X_positions[:lookback] = np.zeros(lookback); Y_positions[:lookback] = np.zeros(lookback)

        # P&L , Returns
        X_returns = (X - X.shift(periods=1))/X.shift(periods=1)
        Y_returns = (Y - Y.shift(periods=1))/Y.shift(periods=1)
        pnl_X = X_positions.shift(periods=1)*X_returns # beta * X *%return (beta included in X_positions)
        pnl_Y = Y_positions.shift(periods=1)*Y_returns
        pnl = pnl_X + pnl_Y
        pnl[0] = 0

        ret = (pnl/(np.abs(X_positions.shift(periods=1))+np.abs(Y_positions.shift(periods=1))))
        # use without fillna(0) according to git version, so that when position is not entered it is not taken into
        # account when calculating the avg return
        ret= ret.fillna(0)

        n_years = round(len(Y) / 240) # approx of # of years, as each year does not have exactly 252 days
        time_in_market = 252. * n_years
        # apr = ((np.prod(1.+ret))**(time_in_market/len(ret)))-1
        if np.std(ret) == 0:
            sharpe = 0
        else:
            sharpe = np.sqrt(time_in_market)*np.mean(ret)/np.std(ret) # should the mean include moments of no holding?
        #print('APR', apr)
        #print('Sharpe', sharpe)

        # get trade summary
        rolling_spread = Y - rolling_beta * X

        # All series contain Date as index
        series_to_include = [(pnl, 'pnl'),
                             (ret, 'ret'),
                             (Y, Y.name),
                             (X, X.name),
                             (rolling_beta,'beta'),
                             (rolling_spread, 'spread'),
                             (zscore, 'zscore'),
                             (numUnits, 'numUnits')]
        summary = self.trade_summary(series_to_include)

        return pnl, ret, summary, sharpe


    def bollinger_bands_ec(self, Y, X, lookback, entry_multiplier=1, exit_multiplier=0):

        df = pd.concat([Y, X], axis = 1)
        df = df.reset_index()
        df['hedgeRatio'] = np.nan
        for t in range(lookback, len(df)):
            x = np.array(X)[t - lookback:t]
            x = sm.add_constant(x)
            y = np.array(Y)[t - lookback:t]
            df.loc[t, 'hedgeRatio'] = sm.OLS(y, x).fit().params[1]

        cols = [X.name, Y.name]

        yport = np.ones(df[cols].shape);
        yport[:, 0] = -df['hedgeRatio']
        yport = yport * df[cols]

        yport = yport[X.name] + yport[Y.name]
        data_mean = pd.rolling_mean(yport, window=lookback)
        data_std = pd.rolling_std(yport, window=lookback)
        zScore = (yport - data_mean) / data_std

        entryZscore = entry_multiplier
        exitZscore = exit_multiplier

        longsEntry = zScore < -entryZscore
        longsExit = zScore > -exitZscore
        shortsEntry = zScore > entryZscore
        shortsExit = zScore < exitZscore

        numUnitsLong = pd.Series([np.nan for i in range(len(df))])
        numUnitsShort = pd.Series([np.nan for i in range(len(df))])
        numUnitsLong[0] = 0.
        numUnitsShort[0] = 0.

        numUnitsLong[longsEntry] = 1.0
        numUnitsLong[longsExit] = 0.0
        numUnitsLong = numUnitsLong.fillna(method='ffill')

        numUnitsShort[shortsEntry] = -1.0
        numUnitsShort[shortsExit] = 0.0
        numUnitsShort = numUnitsShort.fillna(method='ffill')
        df['numUnits'] = numUnitsShort + numUnitsLong

        tmp1 = np.ones(df[cols].shape) * np.array([df['numUnits']]).T
        tmp2 = np.ones(df[cols].shape)
        tmp2[:, 0] = -df['hedgeRatio']
        positions = pd.DataFrame(tmp1 * tmp2 * df[cols]).fillna(0)
        pnl = positions.shift(1) * (df[cols] - df[cols].shift(1)) / df[cols].shift(1)
        pnl = pnl.sum(axis=1)
        ret = pnl / np.sum(np.abs(positions.shift(1)), axis=1)
        ret = ret.fillna(0)
        apr = ((np.prod(1. + ret)) ** (252. / len(ret))) - 1
        print('APR', apr)
        if np.std(ret) == 0:
            sharpe = 0
        else:
            sharpe = np.sqrt(252.)*np.mean(ret)/np.std(ret) # should the mean include moments of no holding?
        print('Sharpe', sharpe)

        # checking results
        X = X.reset_index(drop=True)
        Y = Y.reset_index()
        pnl.name = 'pnl';
        rolling_spread = yport
        rolling_spread.name = 'spread'
        zScore.name = 'zscore'
        ret.name = 'ret'
        numUnits = df['numUnits']; numUnits.name = 'current_position'
        numUnits = numUnits.shift()
        summary = pd.concat([pnl, ret, X, Y, rolling_spread, zScore, numUnits], axis=1)
        summary.index = summary['Date']
        # new_df = new_df.loc[datetime(2006,7,26):]
        summary = summary[36:]

        return pnl, ret, summary, sharpe

    def kalman_filter(self, y, x, entry_multiplier=1.0, exit_multiplier=1.0, stabilizing_threshold=5,
                      trading_filter=None):
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

        tmp1 = np.tile(np.matrix(numUnits).T, 2)
        tmp2 = np.hstack((-1*beta[0, :].T, np.ones((len(y),1))))
        positions = np.array(tmp1)*np.array(tmp2)*y2

        positions = pd.DataFrame(positions)

        tmp1 = np.array(positions.shift(1))
        tmp2 = np.array(y2-y2.shift(1))
        tmp3 = np.array(y2.shift(1))
        pnl = np.sum(tmp1 * tmp2 / tmp3,axis=1)
        ret = pnl / np.sum(np.abs(positions.shift(1)),axis=1)
        ret = ret.fillna(0)
        #ret = ret.dropna()

        n_years = round(len(y)/240)
        time_in_market = 252.*n_years
        # apr = ((np.prod(1.+ret))**(time_in_market/len(ret)))-1
        if np.std(ret) == 0:
            sharpe = 0
        else:
            sharpe = np.sqrt(time_in_market) * np.mean(ret) / np.std(ret)

        # get summary df
        # No series should have Date as index
        series_to_include = [(pd.Series(pnl), 'pnl'), (ret.reset_index(drop=True), 'ret'),
                             (y_series.reset_index(drop=True), y_series.name),
                             (x_series.reset_index(), x_series.name),
                             (pd.Series(np.squeeze(np.asarray(beta[0, :]))), 'beta'),
                             (pd.Series(e), 'e'), (pd.Series(np.sqrt(Q)), 'sqrt(Q)'),
                             (numUnits.reset_index(drop=True), 'numUnits')]
        summary = self.trade_summary(series_to_include)

        return pnl, ret, summary, sharpe

    def apply_bollinger_strategy(self, pairs, lookback_multiplier, entry_multiplier=2, exit_multiplier=0.5,
                                 trading_filter=None, test_mode = False):
        """
        This function applies the bollinger strategy. We do not let the lookback extend further than 20 days,
        as this would be too long of a period for the time ranges we are dealing with

        :param pairs: pairs to trade
        :param lookback_multiplier: half life multiplier to define lookback period
        :param entry_multiplier: multiplier to define position entry level
        :param exit_multiplier: multiplier to define position exit level
        :param trading_filter: trading_flter dictionary with parameters or None object in case of no filter
        :param test_mode: flag to decide whether to apply strategy on the training set or in the test set

        :return: sharpe ratio results
        :return: cumulative returns
        :return: pairs which had a negative sharpe ratio
        """

        sharpe_results = []
        cum_returns = []
        performance = []  # aux variable to store pairs' record

        for pair in pairs:
            #print('\n\n{},{}'.format(pair[0], pair[1]))
            pair_info = pair[2]
            lookback = min(lookback_multiplier * (pair_info['half_life']), 20)
            if trading_filter is not None:
                trading_filter['lookback'] = min(trading_filter['filter_lookback_multiplier']*(pair_info['half_life']),
                                                 20)

            if lookback >= len(pair_info['Y_train']):
                print('Error: lookback is larger than length of the series')

            if test_mode:
                y = pair_info['Y_test']
                x = pair_info['X_test']
            else:
                y = pair_info['Y_train']
                x = pair_info['X_train']
            pnl, ret, summary, sharpe = self.bollinger_bands(Y=y,X=x,
                                                             lookback=lookback,
                                                             entry_multiplier=entry_multiplier,
                                                             exit_multiplier=int(exit_multiplier),
                                                             trading_filter=trading_filter)
            cum_returns.append((np.cumprod(1 + ret) - 1)[-1] * 100)
            sharpe_results.append(sharpe)
            performance.append((pair, summary))

        return sharpe_results, cum_returns, performance

    def apply_kalman_strategy(self, pairs, entry_multiplier=1, exit_multiplier=0, trading_filter=None,
                              test_mode=False):
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
        performance = []  # aux variable to store pairs' record
        for pair in pairs:
            # print('\n\n{},{}'.format(pair[0], pair[1]))
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
            pnl, ret, summary, sharpe = self.kalman_filter(y=y, x=x,
                                                           entry_multiplier=entry_multiplier,
                                                           exit_multiplier=exit_multiplier,
                                                           trading_filter=trading_filter)
            cum_returns.append((np.cumprod(1 + ret) - 1).iloc[-1] * 100)
            sharpe_results.append(sharpe)
            performance.append((pair, summary))

        return sharpe_results, cum_returns, performance

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
        summary = summary.rename(columns={"numUnits": "current_position"})
        if 'index' in summary.columns:
            summary = summary.rename(columns={"index": "Date"})
            summary = summary.set_index('Date')

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

        df['position_return_(%)'] = 0
        previous_unit = 0.
        position_ret_acc = 1.
        for index, row in df.iterrows():
            if previous_unit == row['numUnits']:
                if previous_unit != 0.:
                    position_ret_acc = position_ret_acc * (1+row['ret'])
                continue  # no change in positions to verify
            else:
                if previous_unit == 0.:
                    previous_unit = row['numUnits']
                    continue  # simply start the trade
                else:
                    # update position returns
                    position_ret_acc = position_ret_acc * (1+row['ret'])
                    df.loc[index, 'position_return_(%)'] = (position_ret_acc-1)*100
                    position_ret_acc = 1.
                    previous_unit = row['numUnits']
                    continue

        return df

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
        print('avg_total_roi: ', avg_total_roi)

        avg_annual_roi = ((1 + (avg_total_roi / 100)) ** (1 / float(n_years)) - 1) * 100
        print('avg_annual_roi: ', avg_annual_roi)

        sharpe_results_filtered = np.asarray(sharpe_results_filtered)
        positive_pct = len(sharpe_results_filtered[sharpe_results_filtered > 0]) * 100 / len(sharpe_results_filtered)
        print('{} % of the pairs had positive returns'.format(positive_pct))

        return avg_sharpe_ratio, avg_total_roi, avg_annual_roi, positive_pct

    def summarize_results(self, sharpe_results, cum_returns, performance, total_pairs, ticker_segment_dict):
        """
        This function summarizes interesting metrics to include in the final output

        :param sharpe_results: array containing sharpe results for each pair
        :param cum_returns: array containing cum returns for each pair
        :param performance: df containing a summary of each pair's trade
        :param total_pairs: list containing all the identified pairs
        :param ticker_segment_dict: dict containing segment for each ticker

        :return: dictionary with metrics of interest
        """

        n_years = round(len(performance[0][1]) / 240)  # performance[0][1] contains time series index, thus true length
        avg_sharpe_ratio, avg_total_roi, avg_annual_roi, positive_pct = \
            self.calculate_metrics(sharpe_results, cum_returns, n_years)

        sorted_indices = np.flip(np.argsort(sharpe_results), axis=0)
        #print(sorted_indices)
        # initialize list of lists
        data = []
        for index in sorted_indices:
            # get number of positive and negative positions
            position_returns = performance[index][1]['position_return_(%)']
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
