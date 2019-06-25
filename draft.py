# This file contains old functions, not being used anymore but might turn out helpful at some point in time

# trader.py

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

def apply_bollinger_strategy(self, pairs, lookback, entry_multiplier=2, exit_multiplier=0.5,
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

    for i,pair in enumerate(pairs):
        # start = time.time()
        #end = time.time()
        #print((end - start))
        print('\n{}/{}'.format(i+1, len(pairs)))
        pair_info = pair[2]

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

# data_processor
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
