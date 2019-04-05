import numpy as np
import pandas as pd

import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

import matplotlib.pyplot as plt

# just set the seed for the random number generator
np.random.seed(107)


class SeriesAnalyser:
    """
    This class contains a set of pairs trading strategies along
    with some auxiliary functions

    """

    def __init__(self):
        """
        :initial elements
        """

    def check_for_stationarity(self, X, cutoff=0.01):
        """
        Receives as input a time series and a cutoff value.
        H_0 in adfuller is unit root exists (non-stationary).
        We must observe significant p-value to convince ourselves that the series is stationary.
        """

        result = adfuller(X)
        # result contains:
        # 0: t-statistic
        # 1: p-value
        # others: please see https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html

        return {'t_statistic': result[0], 'p_value': result[1], 'critical_values': result[4]}

    def check_for_cointegration(self, X, Y):
        """
        Gets two time series as inputs and provides information concerning cointegration stasttics
        Y - b*X : Y is dependent, X is independent
        """

        # for some reason is not giving right results
        # t_statistic, p_value, crit_value = coint(X,Y, method='aeg')

        # perform test manally in both directions
        pairs = [(X, Y), (Y, X)]
        coint_stats = [0] * 2

        for i, pair in enumerate(pairs):
            S1 = pair[0];
            S2 = pair[1];

            series_name = S1.name
            S1 = sm.add_constant(S1)
            # Y = bX + c
            # ols: (Y, X)
            results = sm.OLS(S2, S1).fit()
            S1 = S1[series_name]
            b = results.params[S1.name]

            spread = S2 - b * S1
            stats = self.check_for_stationarity(pd.Series(spread, name='Spread'))
            coint_stats[i] = {'t_statistic': stats['t_statistic'],
                              'critical_val': stats['critical_values'],
                              'p_value': stats['p_value'],
                              'coint_coef': b,
                              'spread': spread,
                              'Y': S2,
                              'X': S1
                              }

        # select lowest t-statistic as representative test
        if abs(coint_stats[0]['t_statistic']) > abs(coint_stats[1]['t_statistic']):
            coint_result = coint_stats[0]
        # print('Spread: Y - b*X')
        else:
            coint_result = coint_stats[1]
        # print('Spread: X - b*Y')

        return coint_result

    def find_cointegrated_pairs(self, data, threshold, min_half_life=5):
        """
        This function receives a df with the different securities as columns, and aims to find cointegrated
        pairs within this world.
        : data - df with price data as columns
        : threshold - pvalue threshold for a pair to be cointegrated
        : min_half_life - minimium half life value of the spreadto consider the pair
        """
        n = data.shape[1]
        keys = data.keys()
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                S1 = data[keys[i]]
                S2 = data[keys[j]]
                result = self.check_for_cointegration(S1, S2)
                pvalue = result['p_value']
                if pvalue < threshold:
                    hl = self.calculate_half_life(result['spread'])
                    if hl > min_half_life:
                        pairs.append((keys[i], keys[j], result))
        return pairs

    def zscore(self, series):
        """
        Returns the nromalized time series assuming a normal distribution
        """
        return (series - series.mean()) / np.std(series)

    def calculate_half_life(self, z_array):
        """
        This function calculates the half life parameter of a
        mean reversion series
        """
        z_lag = np.roll(z_array, 1)
        z_lag[0] = 0
        z_ret = z_array - z_lag
        z_ret[0] = 0

        # adds intercept terms to X variable for regression
        z_lag2 = sm.add_constant(z_lag)

        model = sm.OLS(z_ret[1:], z_lag2[1:])
        res = model.fit()

        halflife = -np.log(2) / res.params[1]

        # print(res.params)
        # print('\nEstimated lambda:',res.params[1])
        # print('Estimated miu:', res.params[0])
        # print('Estimated half-life:', halflife)

        return halflife

    def hurst(self, ts):
        """
        Returns the Hurst Exponent of the time series vector ts.
        Series vector ts should be a price series.
        Source: https://www.quantstart.com/articles/Basics-of-Statistical-Mean-Reversion-Testing"""
        # Create the range of lag values
        lags = range(2, 100)

        # Calculate the array of the variances of the lagged differences
        # Here it calculates the variances, but why it uses
        # standard deviation and then make a root of it?
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]

        # Use a linear fit to estimate the Hurst Exponent
        poly = np.polyfit(np.log(lags), np.log(tau), 1)

        # Return the Hurst exponent from the polyfit output
        return poly[0] * 2.0

    def variance_ratio(self, ts, lag=2):
        """
        Returns the variance ratio test result
        Source: https://gist.github.com/jcorrius/56b4983ca059e69f2d2df38a3a05e225#file-variance_ratio-py
        """
        # make sure we are working with an array, convert if necessary
        ts = np.asarray(ts)

        # Apply the formula to calculate the test
        n = len(ts)
        mu = sum(ts[1:n] - ts[:n - 1]) / n;
        m = (n - lag + 1) * (1 - lag / n);
        b = sum(np.square(ts[1:n] - ts[:n - 1] - mu)) / (n - 1)
        t = sum(np.square(ts[lag:n] - ts[:n - lag] - lag * mu)) / m
        return t / (lag * b);
