import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import mplfinance as mpf
import datetime

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



def backtest(calculated_df, slope_threshold, profit_target=10, stop_loss=20, holding_period=20):

    ### INITIALISATION ###
    open_date = datetime.datetime.now().date()
    open_price = 0
    num_of_shares = 0
    pnl = 0
    net_profit = 0
    num_of_trades = 0

    for i, row in calculated_df.iterrows():
        now_date = i.date()
        now_open = row['Open']
        now_high = row['High']
        now_low = row['Low']
        now_close = row['Close']

        now_slope = row['rolling_slope']
        now_atr = row['ATR']

        now_candle_size = round(now_close - now_open, 2)

        trade_logic = now_slope > slope_threshold
        close_logic = (now_date - open_date).days >= holding_period
        profit_target_condition = now_close - open_price > profit_target # * 0.01 * open_price
        stop_loss_condition = open_price - now_close > stop_loss # * 0.01 * open_price
        last_index_condition = i == calculated_df.index[-1]
        



        ### OPEN POSITION ###
        if num_of_shares == 0 and trade_logic:               
            num_of_shares = 1
            open_date = now_date
            open_price = now_close

        ### CLOSE POSITION ###
        elif num_of_shares > 0 and (stop_loss_condition or profit_target_condition or last_index_condition or close_logic): 
            num_of_shares = 0
            pnl = now_close - open_price


            net_profit += pnl
            num_of_trades += 1
            

    print("net profit", net_profit)
    print("num of trades", num_of_trades)





if __name__ == "__main__":

    stock_name = 'AAPL'
    ticker = yf.Ticker(stock_name)

    df = ticker.history(start='2002-01-01', end='2021-12-31', interval='1d')
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df[df['Volume'] > 0]
    preprocessed_df = df.copy()

    calculated_df = ranking_momentum(preprocessed_df)


    slope_threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    profit_target_list = [5, 10, 15, 20]
    stop_loss_list = [10, 20, 30, 40]
    holding_period_list = [10, 20, 30, 40]

    for slope_threshold in slope_threshold_list:
        for profit_target in profit_target_list:
            for stop_loss in stop_loss_list:
                for holding_period in holding_period_list:

                    backtest(calculated_df, slope_threshold, profit_target, stop_loss, holding_period)
