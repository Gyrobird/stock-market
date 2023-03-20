import yfinance as yf
import pandas as pd

# Download stock data for Apple (AAPL)
#data = yf.download("AAPL", start="2022-02-01", end="2022-02-21")

# Display data in table form using pandas
#print(pd.DataFrame(data))


# Get a list of all tickers in yfinance
tickers = yf.tickers()

# Print the list of tickers
print(tickers.tickers)
# Download stock data for Apple (AAPL)
#data = yf.Ticker("AAPL").history(period="10y")

# Print the data
#print(data)