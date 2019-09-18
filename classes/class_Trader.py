import numpy as np
import pandas as pd
import sys
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

    def threshold_strategy(self, y, x, beta, entry_level=1.0, exit_level=1.0, stabilizing_threshold=5):
        """
        This function implements a threshold filter strategy with a fixed beta, corresponding to the cointegration
        ratio.
        :param y: price series of asset y
        :param x: price series of asset x
        :param entry_level: abs of long and short threshold
        :param exit_multiplier: abs of exit threshold
        :param stabilizing_threshold: number of initial periods when no positions should be set
        """

        # calculate normalized spread
        spread = y - beta * x
        norm_spread = (spread - spread.mean()) / np.std(spread)
        norm_spread = np.asarray(norm_spread.values)

        # get indices for long and short positions
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

        # set threshold crossings with corresponding position
        num_units_long[longs_entry] = 1.
        num_units_long[longs_exit] = 0
        num_units_short[shorts_entry] = -1.
        num_units_short[shorts_exit] = 0

        # shift to simulate entry delay in real life trading
        # please comment if no need to simulate delay
        num_units_long = num_units_long.shift(1)
        num_units_short = num_units_short.shift(1)

        # initialize market position with zero
        num_units_long[0] = 0.
        num_units_short[0] = 0.
        # finally, fill in between
        num_units_long = num_units_long.fillna(method='ffill')
        num_units_short = num_units_short.fillna(method='ffill')
        num_units = num_units_long + num_units_short
        num_units = pd.Series(data=num_units.values, index=y.index, name='numUnits')

        # add position durations
        trading_durations = self.add_trading_duration(pd.DataFrame(num_units, index=y.index))

        # Method 1: calculate return per each position
        # This method receives the series with the positions and calculate the return at the end of each position, not
        # yet accounting for costs
        position_ret, _, ret_summary = self.calculate_position_returns(y, x, beta, num_units)
        # Method 2: calculate balance in total
        # This method constructs the portfolio during the entire trading session and calculates the returns every 5 min.
        # By compounding the returns during a position, we obtain the position return as given in method 1.
        # This method is necessary to obtain the daily returns from which to estimate the Sharpe Ratio
        balance_summary = self.calculate_balance(y, x, beta, num_units.shift(1).fillna(0), trading_durations)

        # add transaction costs and gather all info in a single dataframe
        series_to_include = [(balance_summary.pnl, 'pnl'),
                             (balance_summary.pnl_y, 'pnl_y'),
                             (balance_summary.pnl_x, 'pnl_x'),
                             (balance_summary.account_balance, 'account_balance'),
                             (balance_summary.returns, 'returns'),
                             (position_ret, 'position_return'),
                             (y, y.name),
                             (x, x.name),
                             (pd.Series(norm_spread, index=y.index), 'norm_spread'),
                             (num_units, 'numUnits'),
                             (trading_durations, 'trading_duration')]
        summary = self.trade_summary(series_to_include, beta)

        # calculate sharpe ratio for each pair separately
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
                               test_mode=False, train_val_split='2017-01-01'):
        """
        This function implements the standard fixed beta trading strategy.
        :param pairs: list with pairs identified in the training set
        :param strategy: currently, only fixed_beta is implemented
        :param entry_multiplier: threshold that defines where to enter a position
        :param exit_multiplier: threshold that defines where to exit a position
        :param test_mode: flag to decide whether to apply strategy on the validation set or in the test set
        :param train_val_split: split of training and validation data
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
                summary, sharpe, balance_summary = self.threshold_strategy(y=y, x=x, beta=pair_info['coint_coef'],
                                                                           entry_level=entry_multiplier,
                                                                           exit_level=exit_multiplier)
                # no costs
                cum_returns.append((np.cumprod(1 + summary.position_return) - 1).iloc[-1] * 100)
                sharpe_results.append(sharpe[0])
                # with costs
                # cum_returns_with_costs.append((np.cumprod(1 + summary.position_ret_with_costs) - 1).iloc[-1] * 100)
                cum_returns_with_costs.append((summary.account_balance[-1] - 1) * 100)
                sharpe_results_with_costs.append(sharpe[1])
                performance.append((pair, summary, balance_summary))

            else:
                print('Only one strategy currently available: \n1.Fixed Beta')
                exit()

        return (sharpe_results, cum_returns), (sharpe_results_with_costs, cum_returns_with_costs), performance

    def trade_summary(self, series, beta=0):
        """
        This function receives a set of series containing information from the trade and
        returns a DataFrame containing the summary data.
        :param series: a list of tuples containing the time series and the corresponding names
        :param beta: cointegration ratio. If moving beta, use beta=0.
        """
        for attribute, attribute_name in series:
            try:
                attribute.name = attribute_name
            except:
                continue
        summary = pd.concat([item[0] for item in series], axis=1)

        # change numUnits so that it corresponds to the position for the row's date,
        # instead of corresponding to the position entered in the end of that day.
        summary['numUnits'] = summary['numUnits'].shift().fillna(0)
        summary = summary.rename(columns={"numUnits": "position_during_day"})

        # add position costs
        summary['position_ret_with_costs'] = self.add_transaction_costs(summary, beta)

        return summary

    def add_trading_duration(self, df):
        """
        The following function adds a column containing the trading duration in days.
        :param df: Dataframe containing column with positions to enter in next day
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
                    # verify if it is last trading day
                    if index == df.index[-1]:
                        df.loc[index, 'trading_duration'] = new_position_counter
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
        Function to add transaction costs per position.
        :param summary: dataframe containing summary of all transactions
        :param beta: cointegration factor, use 0 if moving beta
        :param comission_costs: commision costs, in percentage, per security, per trade
        :param market_impact: market impact costs, in percentage, per security, per trade
        :param short_rental: short rental costs, in annual percentage
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
        Calculate sharpe ratio for one asset only.
        As an estimate of the expected value use the yearly return.
        :param n_years: number of years being considered
        :param n_days: number of trading days per year
        :param ret: array containing returns per timestep
        """
        rf = {2014: 0.00033, 2015: 0.00053, 2016: 0.0032, 2017: 0.0093, 2018: 0.0194}
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
        """
        # calculate total daily account balance & df with returns
        total_account_balance = performance[0][1]['account_balance'].resample('D').last().dropna()
        portfolio_returns = total_account_balance.pct_change().fillna(0)
        for index in range(1, len(pairs)):
            pair_balance = performance[index][1]['account_balance'].resample('D').last().dropna()
            total_account_balance = total_account_balance + pair_balance
            portfolio_returns = pd.concat([portfolio_returns, pair_balance.pct_change().fillna(0)], axis=1)

        # add first day with initial balance
        total_account_balance = pd.Series(data=[len(pairs)],
                                          index=[total_account_balance.index[0] - timedelta(days=1)]).append(
                                          total_account_balance)

        # calculate portfolio volatility
        weights = np.array([1 / len(pairs)] * len(pairs))
        vol = np.sqrt(np.dot(weights.T, np.dot(portfolio_returns.cov(), weights)))

        # calculate sharpe ratio
        rf = {2014: 0.00033, 2015: 0.00053, 2016: 0.0032, 2017: 0.0093, 2018: 0.0194}
        annualized_ret = (total_account_balance[-1]-len(pairs))/len(pairs)
        year = total_account_balance.index[-1].year
        if year in rf.keys():
            # assuming iid return's distributio, sr may be calculated as:
            sharpe_ratio = (annualized_ret - rf[year]) / (vol*np.sqrt(252))
            print('Sharpe Ratio assumming IID returns: ',sharpe_ratio)
            print('Autocorrelation: ', total_account_balance.pct_change().fillna(0).autocorr(lag=1))
            # accounting for non-zero autocorrelatio, daily sr should be calculated as:
            # the daily sharpe ratio is then multiplied by the annualization factor proposed by the paper: The
            # Statistics of Sharpe Ratios by Andrew W Lo
            annualized_ret = total_account_balance.pct_change().fillna(0).mean()
            rf_daily = (1+rf[year])**(1/252)-1
            sharpe_ratio = (annualized_ret-rf_daily) /vol
            print('Daily Sharpe Ratio', sharpe_ratio)
        else:
            print('Not considering risk-free rate')
            sharpe_ratio = annualized_ret / (vol*np.sqrt(252))

        return sharpe_ratio

    def calculate_maximum_drawdown(self, account_balance):
        """
        Function to calculate maximum drawdown w.r.t portfolio balance.

        source: https://stackoverflow.com/questions/22607324/start-end-and-duration-of-maximum-drawdown-in-python
        """

        # first calculate total drawdown period
        account_balance_drawdowns = account_balance.resample('D').last().dropna().diff().fillna(0).apply(lambda row: 0 if row >= 0 else 1)
        total_dd_duration = account_balance_drawdowns.sum()
        print('Total Drawdown Days: {} days'.format(total_dd_duration))

        xs = np.asarray(account_balance.values)

        i = np.argmax(np.maximum.accumulate(xs) - xs)  # end of the period
        if i == 0:
            plt.plot(xs)
            return 0
        else:
            j = np.argmax(xs[:i])  # start of period
            plt.figure(figsize=(10,7))
            plt.grid()
            plt.plot(xs, label='Total Account Balance')
            dates = account_balance.resample('BMS').first().dropna().index.date
            xi = np.arange(0, len(account_balance), len(account_balance)/12)
            plt.xticks(xi, dates, rotation=50)
            plt.xlim(0, len(account_balance))
            plt.plot([i, j], [xs[i], xs[j]], 'o', color='Red', markersize=10)
            plt.xlabel('Date', size=12)
            plt.ylabel('Capital($)', size=12)
            plt.legend()

        max_dd_period = round((i - j) / 78)
        print('Max DD period: {} days'.format(max_dd_period))
        #print('Max DD period: {} days'.format((account_balance.index[i]-account_balance.index[j]).days))

        return (xs[i]-xs[j])/xs[j] * 100, max_dd_period, total_dd_duration

    def calculate_position_returns(self, y, x, beta, positions):
        """
        This method receives the series with the positions and calculate the return at the end of each position, not
        yet accounting for costs

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
        # add end position if trading period is over and position is open
        if positions[-1] != 0:
            end_position[-1] = 1.

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

    def calculate_metrics(self, cum_returns, n_years):
        """
        Calculate common metrics on average over all pairs.
        :param cum_returns: array with cumulative returns of every pair
        :param n_years: numbers of yers of the trading strategy
        :return: average average total roi
        :return: average annual roi
        :return: percentage of pairs with positive returns
        """
        # use below for fully invested capital:
        # cum_returns_filtered = [cum for cum in cum_returns if cum != 0]
        # or use below for commited capital:
        cum_returns_filtered = cum_returns

        avg_total_roi = np.mean(cum_returns_filtered)

        avg_annual_roi = ((1 + (avg_total_roi / 100)) ** (1 / float(n_years)) - 1) * 100
        print('Annual ROI: ', avg_annual_roi)

        cum_returns_filtered = np.asarray(cum_returns_filtered)
        positive_pct = len(cum_returns_filtered[cum_returns_filtered > 0]) * 100 / len(cum_returns_filtered)
        print('{} % of the pairs had positive returns'.format(positive_pct))

        return avg_total_roi, avg_annual_roi, positive_pct

    def summarize_results(self, sharpe_results, cum_returns, performance, total_pairs, ticker_segment_dict, n_years):
        """
        This function summarizes interesting metrics to include in the final output
        :param sharpe_results: array containing sharpe results for each pair
        :param cum_returns: array containing cum returns for each pair
        :param performance: df containing a summary of each pair's trade
        :param total_pairs: list containing all the identified pairs
        :param ticker_segment_dict: dict containing segment for each ticker
        :param n_years: number of years the strategy is running
        :return: dictionary with metrics of interest
        """

        avg_total_roi, avg_annual_roi, positive_pct = self.calculate_metrics(cum_returns, n_years)

        portfolio_sharpe_ratio = self.calculate_portfolio_sharpe_ratio(performance, total_pairs)

        sorted_indices = np.flip(np.argsort(sharpe_results), axis=0)
        # print(sorted_indices)
        # initialize list of lists
        data = []
        for index in sorted_indices:
            # get number of positive and negative positions
            position_returns = performance[index][1]['position_ret_with_costs']
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

        print('Total number of trades: ', pairs_df.positive_trades.sum() + pairs_df.negative_trades.sum())
        print('Positive trades: ', pairs_df.positive_trades.sum())
        print('Negative trades: ', pairs_df.negative_trades.sum())

        avg_positive_trades_per_pair_pct = pairs_df['positive_trades_per_pair_pct'].mean()

        results = {'n_pairs': len(sharpe_results),
                   'portfolio_sharpe_ratio': portfolio_sharpe_ratio,
                   'avg_total_roi': avg_total_roi,
                   'avg_annual_roi': avg_annual_roi,
                   'pct_positive_trades_per_pair': avg_positive_trades_per_pair_pct,
                   'pct_pairs_with_positive_results': positive_pct,
                   'avg_half_life': pairs_df['half_life'].mean(),
                   'avg_hurst_exponent': pairs_df['hurst_exponent'].mean()}

        # Drawdown info
        total_account_balance = performance[0][1]['account_balance']
        for index in range(1, len(total_pairs)):
            total_account_balance = total_account_balance + performance[index][1]['account_balance']
        total_account_balance = total_account_balance.fillna(method='ffill')
        max_dd, max_dd_duration, total_dd_duration = self.calculate_maximum_drawdown(total_account_balance)
        print('Maximum drawdown of portfolio: {:.2f}%'.format(max_dd))

        return results, pairs_df

