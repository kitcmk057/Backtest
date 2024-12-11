import time
import datetime
import os
import sys
import multiprocessing as mp
import hkfdb
import yfinance as yf
import pandas as pd
import numpy as np
import plotguy
import itertools
import pandas_ta as ta
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from process.load_data.form_stock_list import get_stock_list
from process.load_data.get_hist_data import get_hist_data
from process.commission_slippery.get_sec_profile import get_sec_profile
from process.para_dict.para_dict import get_all_para_combination
from process.plotguy.run_plotguy import run_plotguy
from process.load_data.get_sp500 import get_sp500_symbols
from sklearn.linear_model import LinearRegression




pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',None)





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


def get_secondary_data(df_dict, start_date, end_date):


    sp500_df = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
    sp500_df['sma200'] = sp500_df['Close'].rolling(200).mean()
    sp500_df['sp500_above_sma200'] = sp500_df['Close'] > sp500_df['sma200']
    sp500_df = sp500_df[['sp500_above_sma200']]


    for code, df in df_dict.items():
        

        df = pd.concat([df, sp500_df], axis=1)
        print(df.columns)
        print(df)
        sys.exit()



        df['slope'] = rolling_exp_regression(df)
        df['sma100'] = df['close'].rolling(100).mean()


        df_dict[code] = df
        print(df)
        print(df.columns)
        sys.exit()

    return df_dict


def backtest(para_combination):

    para_dict       = para_combination['para_dict']
    sec_profile     = para_combination['sec_profile']
    start_date      = para_combination['start_date']
    end_date        = para_combination['end_date']
    reference_index = para_combination['reference_index']
    freq            = para_combination['freq']
    file_format     = para_combination['file_format']
    df              = para_combination['df']
    intraday        = para_combination['intraday']
    output_folder   = para_combination['output_folder']
    data_folder     = para_combination['data_folder']
    run_mode        = para_combination['run_mode']
    summary_mode    = para_combination['summary_mode']
    py_filename     = para_combination['py_filename']


    ##### stra specific #####
    code                = para_combination['code']
    profit_target       = para_combination['profit_target']
    stop_loss           = para_combination['stop_loss']
    holding_day         = para_combination['holding_day']

    sp500_above_sma200  = para_combination['sp500_above_sma200']

    ##### sec_profile #####
    market          = sec_profile['market']
    sectype         = sec_profile['sectype']
    initial_capital = sec_profile['initial_capital']
    lot_size_dict   = sec_profile['lot_size_dict']
    lot_size        = lot_size_dict[code]

    if sectype == 'STK':
        if market == 'HK':
            commission_rate = sec_profile['commission_rate']
            min_commission  = sec_profile['min_commission']
            platform_fee    = sec_profile['platform_fee']
        if market == 'US':
            commission_each_stock   = sec_profile['commission_each_stock']
            min_commission          = sec_profile['min_commission']
            platform_fee_each_stock = sec_profile['platform_fee_each_stock']
            min_platform_fee        = sec_profile['min_platform_fee']


    ##### stra specific #####

    df['trade_logic'] = df['close'] > df['sma100']     # stock must be above 100 MA 
    df['market_logic'] = df['sp500_above_sma200']      # SP500 must be above 200 MA



    ##### initialization #####

    df['action'] = ''
    df['num_of_share'] = 0
    df['open_price'] = np.NaN
    df['close_price'] = np.NaN
    df['realized_pnl'] = np.NaN
    df['unrealized_pnl'] = 0
    df['net_profit'] = 0
    df['equity_value'] = initial_capital
    df['mdd_dollar'] = 0
    df['mdd_pct'] = 0
    df['commission'] = 0
    df['logic'] = None
    open_date    = datetime.datetime.now().date()
    open_price   = 0
    num_of_share = 0
    net_profit   = 0
    num_of_trade = 0
    last_realized_capital = initial_capital
    equity_value = 0
    realized_pnl   = 0
    unrealized_pnl = 0
    commission = 0



    for i, row in df.iterrows():
        now_date  = i.date()
        now_open  = row['open']
        now_high  = row['high']
        now_low   = row['low']
        now_close = row['close']

        ##### stra specific #####
        trade_logic = row['trade_logic']

        ##### commission #####
        if sectype == 'STK':
            if market == 'HK':
                if num_of_share > 0:
                    commission = (now_close * num_of_share) * commission_rate
                    if commission < min_commission: commission = min_commission
                    commission += platform_fee
                    commission = 2 * commission
                else:
                    commission = 0
            elif market == 'US':
                if num_of_share > 0:
                    commission = num_of_share * commission_each_stock
                    if commission < min_commission: commission = min_commission
                    platform_fee = num_of_share * platform_fee_each_stock
                    if platform_fee < min_platform_fee: platform_fee = min_platform_fee
                    commission += platform_fee
                    commission = 2 * commission
                else:
                    commission = 0

        ##### equity value #####
        unrealized_pnl = num_of_share * (now_close - open_price) - commission
        equity_value   = last_realized_capital + unrealized_pnl
        net_profit     = round(equity_value - initial_capital,2)

        if trade_logic: df.at[i, 'logic'] = 'trade_logic'

        if run_mode == 'backtest':

            close_logic        = num_of_share != 0 and (now_date - open_date).days >= holding_day
            profit_target_cond = num_of_share != 0 and now_close - open_price > profit_target * open_price * 0.01
            stop_loss_cond     = num_of_share != 0 and open_price - now_close > stop_loss * open_price * 0.01
            last_index_cond    = i == df.index[-1]
            min_cost_cond      = last_realized_capital > now_close * lot_size

            ##### open position #####
            if num_of_share == 0 and not last_index_cond and min_cost_cond and trade_logic:

                num_of_lot = last_realized_capital // (lot_size * now_close)
                num_of_share = num_of_lot * lot_size

                open_price = now_close
                open_date  = now_date

                df.at[i, 'action'] = 'open'
                df.at[i, 'open_price'] = open_price

            ##### close position #####
            elif num_of_share > 0 and (profit_target_cond or stop_loss_cond or last_index_cond or close_logic):

                realized_pnl = unrealized_pnl
                unrealized_pnl = 0
                last_realized_capital += realized_pnl
                num_of_trade += 1
                num_of_share = 0


                if close_logic: df.at[i, 'logic'] = 'close_logic'
                if last_index_cond: df.at[i, 'action'] = 'last_index'
                if close_logic: df.at[i, 'action'] = 'close_logic'
                if profit_target_cond: df.at[i, 'action'] = 'profit_target'
                if stop_loss_cond: df.at[i, 'action'] = 'stop_loss'


                df.at[i, 'close_price']  = now_close
                df.at[i, 'realized_pnl'] = realized_pnl
                df.at[i, 'commission']   = commission




        ### record at last ###
        df.at[i, 'equity_value'] = equity_value
        df.at[i, 'num_of_share'] = num_of_share
        df.at[i, 'unrealized_pnl'] = unrealized_pnl
        df.at[i, 'net_profit'] = net_profit




    if summary_mode and run_mode == 'backtest':
        df = df[df['action'] != '']

    save_path = plotguy.generate_filepath(para_combination)
    print(save_path)

    if file_format == 'csv':
        df.to_csv(save_path)
    
    elif file_format == 'parquet':
        df.to_parquet(save_path)




if __name__ == '__main__':




    start_date  = '2018-01-01'
    end_date    = '2022-12-31'
    freq        = '1D'
    market      = 'US'
    sectype     = 'STK'
    file_format = 'parquet'

    initial_capital = 200000

    update_data = False
    run_mode = 'backtest'
    summary_mode = False
    read_only = False
    number_of_core = 4
    mp_mode = True

    # code_list = get_stock_list(num_stocks=3)
    code_list = get_sp500_symbols(num_stocks=5)


    para_dict = {
        'code'                 : code_list,
        'profit_target'        : [5, 10],
        'stop_loss'            : [2.5, 5],
        'holding_day'          : [5, 10]
    }


    ########################################################################
    ########################################################################


    df_dict                 = get_hist_data(code_list, start_date, end_date, freq, file_format, update_data, market)
    df_dict                 = get_secondary_data(df_dict, start_date, end_date)
    sec_profile             = get_sec_profile(code_list, market, sectype, initial_capital)
    all_para_combination    = get_all_para_combination(para_dict, df_dict, sec_profile, start_date, end_date, run_mode, summary_mode, freq, file_format)


    if not read_only:
        if mp_mode:
            pool = mp.Pool(processes=number_of_core)
            pool.map(backtest, all_para_combination)
            pool.close()
        
        else:
            for para_combination in all_para_combination:
                backtest(para_combination)



    ########################################################################
    ########################################################################


    run_plotguy(all_para_combination, number_of_core)