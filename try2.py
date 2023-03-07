import yfinance as yf
import pandas as pd
import ta

# Download stock data
data = yf.download("AAPL", start="2010-01-01", end="2022-02-21")

# Preprocess data
data.dropna(inplace=True)  # Remove missing values
data["SMA20"] = ta.trend.sma_indicator(data["Close"], window=20)  # Add 20-day simple moving average
data["SMA50"] = ta.trend.sma_indicator(data["Close"], window=50)  # Add 50-day simple moving average
data["RSI"] = ta.momentum.rsi(data["Close"], window=14)  # Add 14-day relative strength index
data_norm = (data - data.mean()) / data.std()  # Normalize data

# Conduct technical analysis
data_norm["Trend"] = "None"  # Add Trend column
data_norm.loc[data_norm["SMA20"] > data_norm["SMA50"], "Trend"] = "Up"  # Identify Up trend
data_norm.loc[data_norm["SMA20"] < data_norm["SMA50"], "Trend"] = "Down"  # Identify Down trend
data_norm["Oversold"] = False  # Add Oversold column
data_norm.loc[data_norm["RSI"] < 30, "Oversold"] = True  # Identify Oversold condition
data_norm["Overbought"] = False  # Add Overbought column
data_norm.loc[data_norm["RSI"] > 70, "Overbought"] = True  # Identify Overbought condition

# Conduct fundamental analysis
income_statement = yf.download("AAPL", period="ytd", interval="1d", group_by="ticker")["income"]  # Download income statement
balance_sheet = yf.download("AAPL", period="ytd", interval="1d", group_by="ticker")["balance"]  # Download balance sheet
cash_flow_statement = yf.download("AAPL", period="ytd", interval="1d", group_by="ticker")["cashflow"]  # Download cash flow statement
pe_ratio = income_statement["netIncome"]/income_statement["totalRevenue"][-1]  # Calculate price-to-earnings ratio
debt_to_equity_ratio = balance_sheet["totalLiab"]/balance_sheet["totalStockholderEquity"][-1]  # Calculate debt-to-equity ratio
free_cash_flow = cash_flow_statement["operatingCashflow"][-1] - cash_flow_statement["capitalExpenditures"][-1]  # Calculate free cash flow

# Make investment decision
if data_norm.iloc[-1]["Trend"] == "Up" and data_norm.iloc[-1]["Oversold"] == True and pe_ratio < 20 and debt_to_equity_ratio < 1 and free_cash_flow > 0:
    print("Buy")
elif data_norm.iloc[-1]["Trend"] == "Down" and data_norm.iloc[-1]["Overbought"] == True and pe_ratio > 30 and debt_to_equity_ratio > 2 and free_cash_flow < 0:
    print("Sell")
else:
    print("Hold")
