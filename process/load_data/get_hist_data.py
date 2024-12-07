import os
import sys
import time
import datetime
import pandas as pd
import yfinance as yf
import hkfdb

client = hkfdb.Database('67399089')



def get_hist_data(code_list, start_date, end_date, freq, file_format, update_data, market):

    base_folder = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_folder = os.path.join(base_folder, 'data')

    start_date_int = int(start_date.replace('-',''))
    end_date_int   = int(end_date.replace('-',''))

    df_dict ={}
    for code in code_list:

        file_path = os.path.join(data_folder, code + '_' + freq + '.' + file_format)

        if os.path.isfile(file_path) and not update_data:
            if file_format == 'csv':
                df = pd.read_csv(file_path)
                df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d')
                df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
                df = df.set_index('datetime')
            elif file_format == 'parquet':
                df = pd.read_parquet(file_path)

            print(datetime.datetime.now(), 'successfully read data', code)
        else:
            if market == 'HK':
                df = client.get_hk_stock_ohlc(code, start_date_int, end_date_int, freq, price_adj=True, vol_adj=True)
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

            elif market == 'US':
                ticker = yf.Ticker(code)
                df = ticker.history(start=start_date, end=end_date)
                df = df[['Open','High','Low','Close','Volume']]
                df = df[df['Volume'] > 0]
                df.columns = map(str.lower, df.columns)
                df = df.rename_axis('datetime')
                df['date'] = df.index.date
                df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

            time.sleep(1)
            if file_format == 'csv':
                df.to_csv(file_path)
            elif file_format == 'parquet':
                df.to_parquet(file_path)
            print(datetime.datetime.now(), 'successfully get data from data source', code)

        df['pct_change'] = df['close'].pct_change()

        df_dict[code] = df

    return df_dict