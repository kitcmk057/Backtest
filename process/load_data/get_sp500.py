import pandas as pd
import yfinance as yf
import requests



def get_sp500_symbols(num_stocks):

    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url, headers=headers, verify=False)
    tables = pd.read_html(response.text)
    df = tables[0]
    symbols = df['Symbol'].str.replace('.', '-').tolist()
    return symbols[:num_stocks]



def check_historical_sp500(symbol, start_date, end_date):

    try:
        sp500 = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
        # stock = yf.download(symbol, start=date, end=date, progress=False)
        return sp500

    except:
        return False
    


if __name__ == "__main__":
    print(check_historical_sp500('AAPL', start_date='2023-01-01', end_date='2024-01-01'))
