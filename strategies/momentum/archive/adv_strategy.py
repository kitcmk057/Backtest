import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import mplfinance as mpf
import datetime
import sys

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)



# STOCK FILTER: 
# - So far only trade SP500 stocks
# - stock must about 100 MA
# - Rank according to momentum (Exponential Regression Slope in last 90 days * R^2)
# - No gap larger than 15% in the last 90 days

# BUY RULES:
# - Only Trade on wednesday
# - MARKET FILTER: SP500 Upper 200 MA

# STOP LOSS:
# - No longer top 20% of all stocks by momentum
# - its trading below 100 MA
# - It has gap larger than 15% in the last 90 days
# - If it left the index

# POSITION SIZING:
# - Scale according to volatility (ATR)
# - Shares = (Account value * Risk Factor) / ATR in 20 days
# If risk factor is 0.001, then the daily impact would be 0.1% of the portfolio

# POSITION REBALANCING:
# - Rebalance portfolio every wednesday


def ranking_momentum(calculated_df):
    calculated_df['rolling_slope'] = rolling_exp_regression(calculated_df, column='Close', days=90)
    calculated_df = average_true_range(calculated_df, window=20)
    return calculated_df


def rolling_exp_regression(df, column='Close', days=90):
    log_prices = np.log(df[column])
    X = np.arange(days).reshape(-1, 1)
    
    slopes = []
    for i in range(len(df) - days + 1):
        y = log_prices.iloc[i:i+days]
        model = LinearRegression().fit(X, y)
        
        slope = model.coef_[0]
        r_squared = model.score(X, y)
        
        adjusted_slope = (slope * 252) * r_squared  # Annualized slope (252 trading days) multiplied by R-squared
        slopes.append(adjusted_slope)

    slopes_series = pd.Series([np.nan] * (days - 1) + slopes, index=df.index)
    return slopes_series


def average_true_range(df, window=20):
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    
    df['ATR'] = atr
    return df






def backtest(calculated_df, initial_capital, sma_direction, sma_length, risk_factor, slope_threshold, profit_target, stop_loss, holding_period, std_ratio_threshold):

    calculated_df['sma'] = calculated_df['Close'].rolling(window=sma_length).mean()
    calculated_df['std'] = calculated_df['Close'].rolling(window=sma_length).std()
    calculated_df['std-ratio'] = (calculated_df['sma'] - calculated_df['Close']) / calculated_df['std']
    
    ### INITIALISATION ###
    open_date = datetime.datetime.now().date()
    open_price = 0
    num_of_shares = 0
    net_profit = 0
    num_of_trades = 0

    last_realised_capital = initial_capital

    equity_value = 0
    realised_pnl = 0
    unrealised_pnl = 0

    commission = 0
    platform_fee = 15
    commission_rate = 0.03 * 0.01
    min_commission = 3

    for i, row in calculated_df.iterrows():
        now_date = i.date()
        now_open = row['Open']
        now_high = row['High']
        now_low = row['Low']
        now_close = row['Close']

        now_slope = row['rolling_slope']
        now_atr = row['ATR']
        now_sma = row['sma']
        now_std_ratio = row['std-ratio']



        ### COMMISSION ### (Need to change to IBKR commission)
        if num_of_shares > 0:
            commission = (num_of_shares * now_close) * commission_rate
            if commission < min_commission: commission = min_commission
            commission += platform_fee
            commission *= 2
        else:
            commission = 0



        ### EQUAL VALUE ###
        unrealised_pnl = (now_close - open_price) * num_of_shares - commission
        equity_value = last_realised_capital + unrealised_pnl 
        net_profit = round(equity_value - initial_capital, 2)

        if sma_direction == 'above':
            trade_logic = now_slope > slope_threshold and (now_close > now_sma) and now_std_ratio < -1 * std_ratio_threshold
        elif sma_direction == 'below':
            trade_logic = now_slope < -slope_threshold and (now_close < now_sma) and now_std_ratio > std_ratio_threshold
        elif sma_direction == 'whatever':
            trade_logic = now_slope > slope_threshold

        close_logic = (now_date - open_date).days >= holding_period
        profit_target_condition = now_close - open_price > profit_target 
        stop_loss_condition = open_price - now_close > stop_loss
        last_index_condition = i == calculated_df.index[-1]
        min_cost_condition = last_realised_capital > now_close * num_of_shares



        ### OPEN POSITION ###
        if num_of_shares == 0 and trade_logic and min_cost_condition and not last_index_condition:               

            num_of_shares = (last_realised_capital * risk_factor) / now_atr

            open_date = now_date
            open_price = now_close

        ### CLOSE POSITION ###
        elif num_of_shares > 0 and (stop_loss_condition or profit_target_condition or last_index_condition or close_logic): 
            realised_pnl = unrealised_pnl
            last_realised_capital += realised_pnl

            num_of_trades += 1
            num_of_shares = 0
            

    return net_profit, num_of_trades




if __name__ == "__main__":

    stock_name = 'AAPL'
    ticker = yf.Ticker(stock_name)

    df = ticker.history(start='2002-01-01', end='2021-12-31', interval='1d')
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df[df['Volume'] > 0]
    preprocessed_df = df.copy()


    df['sma'] = df['Close'].rolling(window=20).mean()
    df['std'] = df['Close'].rolling(window=20).std()
    df['std-ratio'] = (df['sma'] - df['Close']) / df['std']
    # sys.exit()



    calculated_df = ranking_momentum(preprocessed_df)
    initial_capital = 1000000

    sma_direction_list = ['above', 'below', 'whatever']
    slope_threshold_list = [0.1, 0.2, 0.3]
    sma_length_list = [10, 20, 50]
    profit_target_list = [10, 15, 20]
    stop_loss_list = [7, 10]
    holding_period_list = [10]
    risk_factor_list = [0.001, 0.002, 0.003]
    std_ratio_threshold_list = [0.5, 1]


    result_dict = {}
    result_dict['sma_length'] = []
    result_dict['sma_direction'] = []
    result_dict['slope_threshold'] = []
    result_dict['profit_target'] = []
    result_dict['stop_loss'] = []
    result_dict['holding_period'] = []
    result_dict['risk_factor'] = []
    result_dict['std_ratio_threshold'] = []
    result_dict['net_profit'] = []
    result_dict['num_of_trades'] = []

    for sma_length in sma_length_list:
        for sma_direction in sma_direction_list:
            for slope_threshold in slope_threshold_list:
                for profit_target in profit_target_list:
                    for stop_loss in stop_loss_list:
                        for holding_period in holding_period_list:
                            for risk_factor in risk_factor_list:
                                for std_ratio_threshold in std_ratio_threshold_list:
                                    net_profit, num_of_trades = backtest(
                                        calculated_df, 
                                        initial_capital, 
                                        sma_direction,
                                        sma_length,
                                        risk_factor=risk_factor, 
                                        slope_threshold=slope_threshold, 
                                        profit_target=profit_target, 
                                        stop_loss=stop_loss, 
                                        holding_period=holding_period,
                                        std_ratio_threshold=std_ratio_threshold
                                    )
                                    print("net profit", round(net_profit, 2))
                                    print("num of trades", num_of_trades)
                                    print("sma_length", sma_length)
                                    print("sma_direction", sma_direction)
                                    print("slope_threshold", slope_threshold)
                                    print("profit_target", profit_target)
                                    print("stop_loss", stop_loss)
                                    print("holding_period", holding_period)
                                    print("risk_factor", risk_factor)
                                    print("std_ratio_threshold", std_ratio_threshold)
                                    print("--------------------------------")

                                    result_dict['sma_length'].append(sma_length)
                                    result_dict['sma_direction'].append(sma_direction)
                                    result_dict['slope_threshold'].append(slope_threshold)
                                    result_dict['profit_target'].append(profit_target)
                                    result_dict['stop_loss'].append(stop_loss)
                                    result_dict['holding_period'].append(holding_period)
                                    result_dict['risk_factor'].append(risk_factor)
                                    result_dict['std_ratio_threshold'].append(std_ratio_threshold)
                                    result_dict['net_profit'].append(net_profit)
                                    result_dict['num_of_trades'].append(num_of_trades)

    result_df = pd.DataFrame(result_dict).sort_values(by='net_profit', ascending=False)
    print(result_df)
