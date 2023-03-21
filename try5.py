import pandas as pd
import yfinance as yf

df = yf.download("AAPL", start="2021-01-01", end="2021-12-31")
df["date"] = df.index # create a new column "date" that contains the index values (which are dates)
