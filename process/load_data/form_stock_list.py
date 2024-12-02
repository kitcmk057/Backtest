import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

__all__ = ['get_stock_list']

def get_stock_list(num_stocks=3): # Current Logic is select the 3 stocks with the highest volume 
    try:
        df = pd.read_csv('process/load_data/Nasdaq_screener.csv')
        df = df.sort_values(by='Volume', ascending=False)
        df = df.head(num_stocks)
        stock_list = df['Symbol'].tolist()
        return stock_list
    

    except FileNotFoundError:
        print("Error: Nasdaq_screener.csv file not found")
        return []
    

    except Exception as e:
        print(f"Error reading stock list: {str(e)}")
        return []
    


if __name__ == "__main__":
    stocks = get_stock_list()

