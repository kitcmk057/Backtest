# Momentum Strategy
Reference from "Stock on the Move" by Joel Greenblatt


### STOCK FILTER: 
- So far only trade SP500 stocks ✅
- stock must be above 100 MA ✅
- Rank according to momentum (Exponential Regression Slope in last 90 days * R^2) 
- No gap larger than 15% in the last 90 days

### BUY RULES:
- Only Trade on wednesday
- MARKET FILTER: SP500 Upper 200 MA
- This strategy have **NO BUYING TIME FILTER**

### STOP LOSS:
- No longer top 20% of all stocks by momentum
- its trading below 100 MA
- It has gap larger than 15% in the last 90 days
- If it left the index
- This strategy have **NO STOP LOSS**

### POSITION SIZING:
- Scale according to volatility (ATR)
- Shares = (Account value * Risk Factor) / ATR in 20 days
- If risk factor is 0.001, then the daily impact would be 0.1% of the portfolio


### POSITION REBALANCING:
- Rebalance portfolio every wednesday

