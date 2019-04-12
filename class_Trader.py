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

    def bollinger_bands(self, Y, X, lookback, entry_multiplier=1, exit_multiplier=0):
        """
        This function implements a pairs trading strategy based
        on bollinger bands.
        Source: Example 3.2 EC's book
        : Y & X: time series composing the spread
        : lookback : Lookback period
        : entry_multiplier : defines the multiple of std deviation used to enter a position
        : exit_multiplier: defines the multiple of std deviation used to exit a position
        """
        print("Warning: don't forget lookback (halflife) must be at least 3.")

        entryZscore = entry_multiplier
        exitZscore = exit_multiplier

        # obtain zscore
        zscore, rolling_beta = self.rolling_zscore(Y, X, lookback)

        # find long and short indices
        numUnitsLong = zscore.copy();
        numUnitsLong.iloc[0] = 0.
        numUnitsLong[zscore < -entryZscore]=1.0
        numUnitsLong[zscore > -exitZscore]=0.0
        numUnitsLong[(numUnitsLong!=0.0) & (numUnitsLong!=1.0)]= np.nan
        numUnitsLong = numUnitsLong.fillna(method='ffill')

        numUnitsShort = zscore.copy();
        numUnitsShort.iloc[0] = 0.
        numUnitsShort[zscore > entryZscore]=-1.0
        numUnitsShort[zscore < exitZscore]=0.0
        numUnitsShort[(numUnitsShort!=0.0) & (numUnitsShort!=-1.0)]= np.nan
        numUnitsShort = numUnitsShort.fillna(method='ffill')

        # concatenate all positions
        numUnits = numUnitsShort + numUnitsLong
        X_positions = (numUnits*(-rolling_beta*X)).fillna(0)
        Y_positions = (numUnits*Y)
        # discard positions up to window size
        X_positions[:lookback] = np.zeros(lookback); Y_positions[:lookback] = np.zeros(lookback)

        # P&L , Returns
        X_returns = (X - X.shift(periods=1))/X.shift(periods=1)
        Y_returns = (Y - Y.shift(periods=1))/Y.shift(periods=1)
        pnl_X = X_positions.shift(periods=1)*X_returns
        pnl_Y = Y_positions.shift(periods=1)*Y_returns
        pnl = pnl_X + pnl_Y
        pnl[0] = 0

        ret = (pnl/(np.abs(X_positions.shift(periods=1))+np.abs(Y_positions.shift(periods=1))))
        # use without fillna(0) according to git version, so that when position is not entered it is not taken into
        # account when calculating the avg return
        ret_0 = ret.fillna(0)
        ret = ret_0

        APR = ((np.prod(1.+ret))**(252./len(ret)))-1
        if np.std(ret) == 0:
            sharpe = 0
        else:
            sharpe = np.sqrt(252.)*np.mean(ret)/np.std(ret) # should the mean include moments of no holding?
        print('APR', APR)
        print('Sharpe', sharpe)


        # checking results
        pnl.name = 'pnl' ;  pnl_X.name = 'pnl_X'; pnl_Y.name = 'pnl_Y'
        rolling_spread = Y-rolling_beta*X
        rolling_spread.name = 'spread'
        ret_0.name = 'ret'
        zscore.name = 'zscore'
        numUnits.name = 'units'
        summary = pd.concat([pnl_X, pnl_Y, pnl, ret_0, X, Y, rolling_spread, zscore, numUnits], axis=1)
        #new_df = new_df.loc[datetime(2006,7,26):]
        summary = summary[36:]

        return pnl, ret_0, summary, sharpe


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
        data_mean = pd.rolling_mean(yport, window=20)
        data_std = pd.rolling_std(yport, window=20)
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
        summary = pd.concat([pnl, X, Y, rolling_spread, zScore], axis=1)
        summary.index = summary['Date']
        # new_df = new_df.loc[datetime(2006,7,26):]
        summary = summary[36:]

        return pnl, ret, summary, sharpe


    def kalman_filter(self, y, x, entry_multiplier=1.0, exit_multiplier=1.0):
        '''
        This function implements a Kalman Filter for the estimation of
        the moving hedge ratio
        '''

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
        numUnitsLong[0]=0.
        numUnitsShort[0]=0.

        numUnitsLong[longsEntry]=1.
        numUnitsLong[longsExit]=0
        numUnitsLong = numUnitsLong.fillna(method='ffill')

        numUnitsShort[shortsEntry]=-1.
        numUnitsShort[shortsExit]=0
        numUnitsShort = numUnitsShort.fillna(method='ffill')

        numUnits=numUnitsLong+numUnitsShort

        tmp1 = np.tile(np.matrix(numUnits).T, 2)
        tmp2 = np.hstack((-1*beta[0, :].T,np.ones((len(y),1))))
        positions = np.array(tmp1)*np.array(tmp2)*y2

        positions = pd.DataFrame(positions)

        tmp1 = np.array(positions.shift(1))
        tmp2 = np.array(y2-y2.shift(1))
        tmp3 = np.array(y2.shift(1))
        pnl = np.sum(tmp1 * tmp2 / tmp3,axis=1)
        ret = pnl / np.sum(np.abs(positions.shift(1)),axis=1)
        ret_0 = ret.fillna(0)
        ret = ret_0
        #ret = ret.dropna()


        apr = ((np.prod(1.+ret))**(252./len(ret)))-1
        sharpe = np.sqrt(252.) * np.mean(ret) / np.std(ret)
        print('APR', apr)
        print('Sharpe', sharpe)

        # get summary df
        threshold_Q = pd.Series(np.sqrt(Q)); threshold_Q.name = 'sqrt(Q)'
        pnl = pd.Series(pnl); pnl.name = 'PNL'
        e = pd.Series(e); e.name = 'e'
        ret.name = 'ret'; numUnits.name = 'numUnits'
        x_series = x_series.reset_index(drop=True); y_series = y_series.reset_index(drop=True); ret = ret.reset_index()
        summary = pd.concat([pnl, ret, x_series, y_series, e, threshold_Q, numUnits], axis=1)
        summary = summary[30:]

        return pnl, ret_0, summary, sharpe

    def apply_bollinger_strategy(self, pairs, lookback_multiplier, entry_multiplier=2, exit_multiplier=0.5,
                                 implementation='standard'):
        """

        :param pairs:
        :param lookback_multiplier:
        :param entry_multiplier:
        :param exit_multiplier:
        :param implementation:
        :return:
        """

        sharpe_results = []
        cum_returns = []
        negative_performance = []  # aux variable to store non profitable pairs

        for pair in pairs:
            print('\n\n{},{}'.format(pair[0], pair[1]))
            coint_result = pair[2]
            lookback = lookback_multiplier * (coint_result['half_life'])

            if lookback >= len(coint_result['Y']):
                print('Error: lookback is larger than length of the series')

            # run 1 of 2 possible implementations
            if implementation == 'standard':
                pnl, ret, summary, sharpe = self.bollinger_bands(coint_result['Y'], coint_result['X'],
                                                                   lookback, entry_multiplier, exit_multiplier)
                cum_returns.append((np.cumprod(1 + ret) - 1)[-1] * 100)
            else:
                pnl, ret, summary, sharpe = self.bollinger_bands_ec(coint_result['Y'], coint_result['X'],
                                                                      lookback, entry_multiplier, exit_multiplier)
                cum_returns.append((np.cumprod(1 + ret) - 1).iloc[-1] * 100)

            sharpe_results.append(sharpe)
            if sharpe < 0:
                negative_performance.append((pair, summary))

        return sharpe_results, cum_returns, negative_performance

    def apply_kalman_strategy(self, pairs, entry_multiplier=2, exit_multiplier=0.5):
        """
        This function caals the kalman filter implementation for every pair.

        :param entry_multiplier: threshold that defines where to enter a position
        :param exit_multiplier: threshold that defines where to exit a position

        :return: sharpe ratio results
        :return: cumulative returns
        """
        sharpe_results = []
        cum_returns = []
        for pair in pairs:
            print('\n\n{},{}'.format(pair[0], pair[1]))
            coint_result = pair[2]
            pnl, ret, summary, sharpe = self.kalman_filter(y=coint_result['Y'], x=coint_result['X'],
                                                           entry_multiplier=entry_multiplier,
                                                           exit_multiplier=exit_multiplier
                                                           )
            cum_returns.append((np.cumprod(1 + ret) - 1)[-1] * 100)
            sharpe_results.append(sharpe)

        return sharpe_results, cum_returns
