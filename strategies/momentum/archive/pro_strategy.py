import datetime
import time

import os
import sys

import hkfdb
import yfinance as yf


import pandas as pd
import numpy as np


from sklearn.linear_model import LinearRegression
import mplfinance as mpf
import plotguy
import itertools



pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

authToken = os.getenv('67399089')
client = hkfdb.Database(authToken)

data_folder = 'data'
secondary_data_folder = 'secondary_data'
backtest_output_folder = 'backtest_output'
signal_output_folder = 'signal_output'

[os.mkdir(folder) for folder in [data_folder, secondary_data_folder, backtest_output_folder, signal_output_folder] if not os.path.exists(folder)]

py_filename = os.path.basename(__file__).replace('.py', '')



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



def rolling_exp_regression(df, column='close', days=90):
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
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    
    df['ATR'] = atr
    return df


def get_secondary_data(df_dict):

    for code, df in df_dict.items():

        ######### CALCULATING EXPONENTIAL REGRESSION #########
        df['rolling_slope'] = rolling_exp_regression(df, column='close', days=90)

        ######### CALCULATING ATR #########
        df = average_true_range(df, window=20)

        df_dict[code] = df

    return df_dict


def get_sec_profile(code_list, market, sectype, initial_capital):
    sec_profile = {}
    lot_size_dict = {}

    if market == 'HK':
        if sectype == 'STK':
            info = client.get_basic_hk_stock_info()
            for code in code_list:
                lot_size = int(info[info['code'] == code]['lot_size'])
                lot_size_dict[code] = lot_size
        else:
            for code in code_list:
                lot_size_dict[code] = 1

    elif market == 'US':
        for code in code_list:
            lot_size_dict[code] = 1


    sec_profile['market'] = market
    sec_profile['sectype'] = sectype
    sec_profile['initial_capital'] = initial_capital
    sec_profile['lot_size_dict'] = lot_size_dict

    if sectype == 'STK':
        if market == 'HK':
            sec_profile['commission_rate'] = 0.03 * 0.01
            sec_profile['platform_fee'] = 15
            sec_profile['min_commission'] = 3

        elif market == 'US':
            sec_profile['commission_each_stock'] = 0.0049
            sec_profile['platform_fee_each_stock'] = 0.005
            sec_profile['min_commission'] = 0.99
            sec_profile['min_platform_fee'] = 1


    return sec_profile


def get_hist_data(code_list, start_date, end_date, freq, data_folder, file_format, update_data, market):
    
    df_dict = {}
    for code in code_list:

        file_path = os.path.join(data_folder, f'{code}_{freq}.{file_format}')

        if os.path.isfile(file_path) and not update_data:
            if file_format == 'csv':
                df = pd.read_csv(file_path)
                df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
                df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
                df.set_index('datetime')

            elif file_format == 'parquet':
                df = pd.read_parquet(file_path)
                df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')


            print(datetime.datetime.now(), "successfully read data")
        
        else:
            if market == 'HK':
                start_date = int(start_date.replace('-', ''))
                end_date = int(end_date.replace('-', ''))
                df = client.get_hk_stock_ohlc(code, start_date, end_date, freq, price_adj=True, vol_adj=True)
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            
            elif market == 'US':
                ticker = yf.Ticker(code)

                df = ticker.history(start='2022-01-01', end='2024-01-01', interval='1D')
                df = df[["Open", "High", "Low", "Close", "Volume"]]
                df = df[df['Volume'] > 0] 
                df.columns = map(str.lower, df.columns)
                df = df.rename_axis('datetime')
                df['date'] = df.index.date
                df = df[["date", "open", "high", "low", "close", "volume"]]


            time.sleep(1)
            
            if file_format == 'csv':
                df.to_csv(file_path)
            
            elif file_format == 'parquet':
                df.to_parquet(file_path)
                
            print(datetime.datetime.now(), "successfully get data from data source")



        df['percentage_change'] = df['close'].pct_change()
        df_dict[code] = df

    return df_dict


def backtest(para_combination):

    para_dict           = para_combination['para_dict']
    sec_profile         = para_combination['sec_profile']
    start_date          = para_combination['start_date']
    end_date            = para_combination['end_date']
    reference_index     = para_combination['reference_index']
    freq                = para_combination['freq']
    file_format         = para_combination['file_format']
    df                  = para_combination['df']
    intraday            = para_combination['intraday']
    output_folder       = para_combination['output_folder']
    data_folder         = para_combination['data_folder']
    py_filename         = para_combination['py_filename']
    run_mode            = para_combination['run_mode']
    summary_mode        = para_combination['summary_mode']


    ######### STRATEGY SPECIFIC PARAMETERS #########
    code                = para_combination['code']
    sma_direction       = para_combination['sma_direction']
    sma_length          = para_combination['sma_length']
    slope_threshold     = para_combination['slope_threshold']
    profit_target       = para_combination['profit_target']
    stop_loss           = para_combination['stop_loss']
    holding_period      = para_combination['holding_period']
    risk_factor         = para_combination['risk_factor']
    std_ratio_threshold = para_combination['std_ratio_threshold']


    df['sma'] = df['close'].rolling(sma_length).mean()
    df['std'] = df['close'].rolling(sma_length).std()
    df['std-ratio'] = (df['sma'] - df['close']) / df['std']


    ######### SEC_PROFILE #########
    market              = sec_profile['market']
    sectype             = sec_profile['sectype']
    initial_capital     = sec_profile['initial_capital']
    lot_size_dict       = sec_profile['lot_size_dict']
    lot_size            = lot_size_dict[code]

    if sectype == 'STK':
        if market == 'HK':
            commission_rate         = sec_profile['commission_rate']
            platform_fee            = sec_profile['platform_fee']
            min_commission          = sec_profile['min_commission']

        elif market == 'US':
            commission_each_stock       = sec_profile['commission_each_stock']
            platform_fee_each_stock     = sec_profile['platform_fee_each_stock']
            min_commission              = sec_profile['min_commission']
            min_platform_fee            = sec_profile['min_platform_fee']



    ### INITIALISATION ###

    df['action'] = ''
    df['num_of_shares'] = 0
    df['open_price'] = np.NaN
    df['close_price'] = np.NaN

    df['realised_pnl'] = np.NaN
    df['unrealised_pnl'] = 0
    df['net_profit'] = 0

    df['equity_value'] = initial_capital
    df['mdd_dollar'] = 0
    df['mdd_pct'] = 0

    df['commission'] = 0
    df['logic'] = None


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


    for i, row in df.iterrows():
        now_date = i.date()
        now_open = row['open']
        now_high = row['high']
        now_low = row['low']
        now_close = row['close']

        now_slope = row['rolling_slope']
        now_atr = row['ATR']
        now_sma = row['sma']
        now_std_ratio = row['std-ratio']



        ### COMMISSION ### (Need to change to IBKR commission)
        if sectype == 'STK':
            if market == 'HK':  
                if num_of_shares > 0:
                    commission = (num_of_shares * now_close) * commission_rate
                if commission < min_commission: commission = min_commission
                commission += platform_fee
                commission *= 2
            
            elif market == 'US':
                if num_of_shares > 0:
                    commission = num_of_shares * commission_each_stock
                if commission < min_commission: commission = min_commission
                platform_fee = num_of_shares * platform_fee_each_stock
                if platform_fee < min_platform_fee: platform_fee = min_platform_fee
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

        if trade_logic: df.at[i, 'logic'] = 'trade_logic'


        if run_mode == 'backtest':

            close_logic = (now_date - open_date).days >= holding_period
            profit_target_condition = open_price != 0 and ((now_close - open_price) > profit_target * 0.01 * open_price)
            stop_loss_condition = open_price != 0 and ((open_price - now_close) > stop_loss * 0.01 * open_price)    
            last_index_condition = i == df.index[-1]
            min_cost_condition = last_realised_capital > now_close * num_of_shares


            ### open POSITION ###
            if num_of_shares == 0 and trade_logic and min_cost_condition and not last_index_condition:               

                num_of_shares = (last_realised_capital * risk_factor) / now_atr

                open_date = now_date
                open_price = now_close

                df.at[i, 'action'] = 'open'
                df.at[i, 'open_price'] = open_price

            
            ### close POSITION ###
            elif num_of_shares > 0 and (stop_loss_condition or profit_target_condition or last_index_condition or close_logic): 
                realised_pnl = unrealised_pnl
                unrealised_pnl = 0
                last_realised_capital += realised_pnl

                num_of_trades += 1
                num_of_shares = 0

                if close_logic: df.at[i, 'logic'] = 'close_logic'

                if last_index_condition: df.at[i, 'action'] = 'last_index'
                if close_logic: df.at[i, 'action'] = 'close_logic'
                if profit_target_condition: df.at[i, 'action'] = 'profit_target'
                if stop_loss_condition: df.at[i, 'action'] = 'stop_loss'

                df.at[i, 'close_price'] = now_close
                df.at[i, 'realised_pnl'] = realised_pnl
                df.at[i, 'commission'] = commission


        ### record at last ###
        df.at[i, 'equity_value'] = equity_value
        df.at[i, 'num_of_shares'] = num_of_shares
        df.at[i, 'unrealised_pnl'] = unrealised_pnl
        df.at[i, 'net_profit'] = net_profit


    if summary_mode and run_mode == 'backtest':
        df = df[df['action'] != '']
    save_path = plotguy.generate_filepath(para_combination)
    print(save_path)

    if file_format == 'parquet':
        df.to_parquet(save_path)
    elif file_format == 'csv':
        df.to_csv(save_path)


def get_all_para_combinations(para_dict, df_dict, sec_profile, start_date, end_date, freq, file_format, data_folder, run_mode, summary_mode):
    
    para_values = list(para_dict.values())
    para_keys = list(para_dict.keys())
    para_list = list(itertools.product(*para_values))

    print("number of para combinations: ", len(para_list))

    intraday = True if freq != '1D' else False
    output_folder = backtest_output_folder if run_mode == 'backtest' else signal_output_folder

    all_para_combinations = []


    for reference_index in range(len(para_list)):
        para = para_list[reference_index]
        code = para[0]
        df = df_dict[code]

        para_combination = {}
        for i in range(len(para_keys)):
            key = para_keys[i]
            para_combination[key] = para[i]

        para_combination['para_dict'] = para_dict
        para_combination['sec_profile'] = sec_profile
        para_combination['start_date'] = start_date
        para_combination['end_date'] = end_date
        para_combination['reference_index'] = reference_index
        para_combination['freq'] = freq
        para_combination['file_format'] = file_format
        para_combination['df'] = df
        para_combination['intraday'] = intraday
        para_combination['output_folder'] = output_folder
        para_combination['data_folder'] = data_folder
        para_combination['py_filename'] = py_filename
        para_combination['run_mode'] = run_mode
        para_combination['summary_mode'] = summary_mode

        all_para_combinations.append(para_combination)

    return all_para_combinations



if __name__ == "__main__":



    initial_capital = 1000000
    start_date = '2022-01-01'
    end_date = '2024-10-01'
    freq = '1D'
    market = 'HK'
    sectype = 'STK'    #CAN BE FUTURES, OPTIONS, ETF, ETC
    file_format = 'parquet'
    update_data = False
    run_mode = 'backtest'
    summary_mode = False 
    number_of_core = 8



    code_list = ['00388']
    df_dict = get_hist_data(code_list, start_date, end_date, freq, data_folder, file_format, update_data, market)
    df_dict = get_secondary_data(df_dict)
    sec_profile = get_sec_profile(code_list, market, sectype, initial_capital)

    para_dict = {

        "code": code_list,
        "sma_direction" : ['above', 'below', 'whatever'],
        "slope_threshold" : [0.1],
        "sma_length" : [10],
        "profit_target" : [10],
        "stop_loss" : [7],
        "holding_period" : [10],
        "risk_factor" : [0.001],
        "std_ratio_threshold" : [0.5]
    
    }

    all_para_combinations = get_all_para_combinations(para_dict, df_dict, sec_profile, start_date, end_date, freq, file_format, data_folder, run_mode, summary_mode)

    for para_combination in all_para_combinations:
        backtest(para_combination)


    plotguy.generate_backtest_result(
        all_para_combination=all_para_combinations,
        number_of_core=number_of_core
    )

    app = plotguy.plot(
        mode='equity_curves',
        all_para_combination=all_para_combinations
    )

    app.run_server(port=8900)