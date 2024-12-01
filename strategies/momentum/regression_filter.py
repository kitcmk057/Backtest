import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from strategies.stock_list import get_stock_list


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


if __name__ == '__main__':
    code_list = get_stock_list()
    sys.exit()