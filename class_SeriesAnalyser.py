import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn import preprocessing

import matplotlib.pyplot as plt

# just set the seed for the random number generator
np.random.seed(107)


class SeriesAnalyser:
    """
    This class contains a set of functions to deal with time series analysis.
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
            S1 = pair[0]
            S2 = pair[1]

            series_name = S1.name
            S1 = sm.add_constant(S1)
            # Y = bX + c
            # ols: (Y, X)
            results = sm.OLS(S2, S1).fit()
            S1 = S1[series_name]
            b = results.params[S1.name]

            spread = S2 - b * S1
            stats = self.check_for_stationarity(pd.Series(spread, name='Spread'))
            zero_cross = self.zero_crossings(spread)
            hl = self.calculate_half_life(spread)
            coint_stats[i] = {'t_statistic': stats['t_statistic'],
                              'critical_val': stats['critical_values'],
                              'p_value': stats['p_value'],
                              'coint_coef': b,
                              'zero_cross': zero_cross,
                              'half_life': int(round(hl)),
                              'spread': spread,
                              'Y': S2,
                              'X': S1
                              }

        # select lowest t-statistic as representative test
        if abs(coint_stats[0]['t_statistic']) > abs(coint_stats[1]['t_statistic']):
            coint_result = coint_stats[0]

        else:
            coint_result = coint_stats[1]

        return coint_result

    def find_pairs(self, data, p_value_threshold, min_half_life=5, min_zero_crossings=0, hurst_threshold=0.5):
        """
        This function receives a df with the different securities as columns, and aims to find tradable
        pairs within this world.
        Tradable pairs are those that verify:
            - cointegration
            - minimium half life
            - minimium zero crossings
        : data - df with price data as columns
        : threshold - pvalue threshold for a pair to be cointegrated
        : min_half_life - minimium half life value of the spread to consider the pair
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
                if pvalue < p_value_threshold: # verifies required pvalue
                    hl = self.calculate_half_life(result['spread'])
                    if hl >= min_half_life: # verifies required half life
                        if result['zero_cross'] >= min_zero_crossings: # verifies required zero crossings
                            if self.hurst(result['spread'])<hurst_threshold: # verifies hurst exponent
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
        mu = sum(ts[1:n] - ts[:n - 1]) / n
        m = (n - lag + 1) * (1 - lag / n)
        b = sum(np.square(ts[1:n] - ts[:n - 1] - mu)) / (n - 1)
        t = sum(np.square(ts[lag:n] - ts[:n - lag] - lag * mu)) / m
        return t / (lag * b)

    def zero_crossings(self, x):
        """
        Function that counts the number of zero crossings of a given signal
        :param x: the signal to be analyzed
        """
        x = x - x.mean()
        zero_crossings = sum(1 for i, _ in enumerate(x) if (i + 1 < len(x)) if ((x[i] * x[i + 1] < 0) or (x[i] == 0)))

        return zero_crossings

    def apply_PCA(self, n_components, df):
        """
        This function applies Principal Component Analysis to the df given as
        parameter

        :param n_components: number of principal components
        :param df: dataframe containing time series for analysis
        :return: reduced normalized and transposed df
        """

        if n_components > df.shape[1]:
            print("ERROR: number of components larger than samples...")
            exit()

        pca = PCA(n_components=n_components)
        pca.fit(df)
        explained_variance = pca.explained_variance_

        # standardize
        X = preprocessing.StandardScaler().fit_transform(pca.components_.T)
        print('New shape: ', X.shape)

        return X, explained_variance

    def apply_DBSCAN(self, eps, min_samples, X, df_returns):
        """
        This function applies a DBSCAN clustering algo

        :param eps: min distance for a sample to be within the cluster
        :param min_samples: min_samples to consider a cluster
        :param X: data

        :return: clustered_series_all: series with all tickers and labels
        :return: clustered_series: series with tickers belonging to a cluster
        :return: counts: counts of each cluster
        :return: clf object
        """
        clf = DBSCAN(eps=eps, min_samples=min_samples)
        print(clf)

        clf.fit(X)
        labels = clf.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print("\nClusters discovered: %d" % n_clusters_)

        clustered_series_all = pd.Series(index=df_returns.columns, data=labels.flatten())
        clustered_series = clustered_series_all[clustered_series_all != -1]

        counts = clustered_series.value_counts()
        print("Pairs to evaluate: %d" % (counts * (counts - 1) / 2).sum())

        return clustered_series_all, clustered_series, counts, clf

    def get_candidate_pairs(self, clustered_series, pricing_df, n_clusters, min_half_life=5,
                            min_zero_crosings=20, p_value_threshold=0.05, hurst_threshold=0.5):
        """
        This function looks for tradable pairs over the clusters formed previously.

        :param clustered_series: series with cluster label info
        :param pricing_df: df with price series
        :param n_clusters: number of clusters
        :param min_half_life: min half life of a time series to be considered as candidate
        :param min_zero_crosings: min number of zero crossings (or mean crossings)
        :param p_value_threshold: p_value to check during cointegration test
        :param hurst_threshold: max hurst exponent value

        :return: list of pairs and its info
        :return: list of unique tickers identified in the candidate pairs universe
        """

        total_pairs = []
        for clust in range(n_clusters):
            symbols = list(clustered_series[clustered_series == clust].index)
            cluster_pricing = pricing_df[symbols]
            pairs = self.find_pairs(cluster_pricing,
                                    p_value_threshold,
                                    min_half_life,
                                    min_zero_crosings,
                                    hurst_threshold)
            total_pairs.extend(pairs)

        print('Found {} pairs'.format(len(total_pairs)))

        unique_tickers = np.unique([(element[0], element[1]) for element in total_pairs])
        print('The pairs contain {} unique tickers'.format(len(unique_tickers)))

        return total_pairs, unique_tickers