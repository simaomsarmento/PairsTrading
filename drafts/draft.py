# This file contains old functions, not being used anymore but might turn out helpful at some point in time

# trader.py
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

def bollinger_bands_ec(self, Y, X, lookback, entry_multiplier=1, exit_multiplier=0):
    df = pd.concat([Y, X], axis=1)
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
        sharpe = np.sqrt(252.) * np.mean(ret) / np.std(ret)  # should the mean include moments of no holding?
    print('Sharpe', sharpe)

    # checking results
    X = X.reset_index(drop=True)
    Y = Y.reset_index()
    pnl.name = 'pnl';
    rolling_spread = yport
    rolling_spread.name = 'spread'
    zScore.name = 'zscore'
    ret.name = 'ret'
    numUnits = df['numUnits'];
    numUnits.name = 'position_during_day'
    numUnits = numUnits.shift()
    summary = pd.concat([pnl, ret, X, Y, rolling_spread, zScore, numUnits], axis=1)
    summary.index = summary['Date']
    # new_df = new_df.loc[datetime(2006,7,26):]
    summary = summary[36:]

    return pnl, ret, summary, sharpe

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

def calculate_returns_no_rebalance(self, y, x, beta, positions):
    """
    Y: price of ETF Y
    X: price of ETF X
    beta: cointegration ratio
    positions: array indicating position to enter in next day
    """

    # calculate each leg return
    y_returns = y.pct_change().fillna(0); y_returns.name = 'y_returns'
    x_returns = x.pct_change().fillna(0); x_returns.name = 'x_returns'

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
    # name positions series
    positions.name = 'positions'

    # apply returns per trade
    # each row contain all the parameters to be applied in that position
    df = pd.concat([y_returns, x_returns, beta_position, positions.shift().fillna(0)], axis=1)
    returns = df.apply(lambda row: self.return_per_timestep(row), axis=1)
    cum_returns = np.cumprod(returns + 1) - 1

    return returns, cum_returns

def calculate_returns_adapted(self, y, x, beta, positions):
    """
    Y: price of ETF Y
    X: price of ETF X
    beta: cointegration ratio
    positions: array indicating when to take a position
    """
    # calculate each leg return
    y_returns = y.pct_change().fillna(0);
    y_returns.name = 'y_returns'
    x_returns = x.pct_change().fillna(0);
    x_returns.name = 'x_returns'

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

def return_per_timestep(self, row):
    if row['beta_position'] > 1.:
        return ((1 / row['beta_position']) * row['y_returns'] - 1 * row['x_returns']) * row['positions']
    else:
        return (row['y_returns'] - row['beta_position'] * row['x_returns']) * row['positions']

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

# data_processor.py
def read_tickers_prices(self, tickers, initial_date, final_date, data_source, column='Adj Close'):
    """
    This function reads the price series for the requested tickers

    :param tickers: list with tickers from which to retrieve prices
    :param initial_date: start date to retrieve price series
    :param final_date: end point
    :param data_source: data source from where to retrieve data

    :return: dictionary with price series for each ticker
    """
    error_counter = 0
    dataset = {key: None for key in tickers}
    for ticker in tickers:
        try:
            df = data.DataReader(ticker, data_source, initial_date, final_date)
            series = df[column]
            series.name = ticker  # filter close price only
            dataset[ticker] = series.copy()
        except:
            error_counter = error_counter + 1
            print('Not Possible to retrieve information for ' + ticker)

    print('\nUnable to download ' + str(error_counter / len(tickers) * 100) + '% of the ETFs')

    return dataset



# forecasting notebook

def apply_ARIMA(series, p, d, q):
    # fit model
    model = ARIMA(series, order=(p,d,q))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    # plot residual errors
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    plt.show()
    residuals.plot(kind='kde')
    plt.show()
    print(residuals.describe())

def rolling_ARIMA(series, p, d, q, train_val_split):
    # standardize
    mean = series.mean()
    std = np.std(series)
    norm_series = (series - mean) / std

    train, val = norm_series[:train_val_split].values, norm_series[train_val_split:].values
    history = np.asarray([x for x in train])
    predictions = list()
    for t in range(len(val)):
        model = ARIMA(history, order=(p, d, q))
        model_fit = model.fit(transparams=False, trend='nc', tol=0.0001, disp=0)
        if t == 0:
            print(model_fit.summary())
            print(history[-5:])
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = val[t]
        history = np.append(history, obs)
        print('predicted=%f, expected=%f' % (yhat, obs))

    # destandardize
    val = val * std + mean
    predictions = np.asarray(predictions);
    predictions = predictions * std + mean
    error = mean_squared_error(val, predictions)
    print('Test MSE: {}'.format(error))
    # plot
    # plt.plot(val)
    # plt.plot(predictions, color='red')
    # plt.show()

    return error, predictions
