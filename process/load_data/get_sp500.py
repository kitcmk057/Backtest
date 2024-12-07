import pandas as pd
import yfinance as yf
import requests



def get_sp500_symbols():

    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url, headers=headers, verify=False)
    tables = pd.read_html(response.text)
    df = tables[0]
    symbols = df['Symbol'].str.replace('.', '-').tolist()
    return symbols


def check_historical_sp500(symbol, date):

    try:
        sp500 = yf.download('^GSPC', start=date, end=date, progress=False)
        stock = yf.download(symbol, start=date, end=date, progress=False)
        return not (sp500.empty or stock.empty)

    except:
        return False
    


if __name__ == "__main__":
    sp500_symbols = get_sp500_symbols()
    print(sp500_symbols)
