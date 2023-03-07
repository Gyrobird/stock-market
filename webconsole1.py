from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import ta

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Get form input
    ticker = request.form['ticker']
    
    # Download stock data
    data = download_stock_data(ticker)
    
    # Preprocess data
    data_norm = preprocess_data(data)
    
    # Conduct technical analysis
    data_analysis = conduct_technical_analysis(data_norm)
    
    # Conduct fundamental analysis
    fundamental_analysis = conduct_fundamental_analysis(ticker)
    
    # Make investment decision
    decision = make_investment_decision(data_analysis, fundamental_analysis)
    
    # Render results template
    return render_template('results.html', decision=decision)

# Define function to download stock data
def download_stock_data(ticker):
    data = yf.download(ticker, start="2010-01-01", end="2022-02-21")
    return data

# Define function to preprocess stock data
def preprocess_data(data):
    # Remove missing values
    data.dropna(inplace=True)
    
    # Add technical indicators
    data["SMA20"] = ta.trend.sma_indicator(data["Close"], window=20)
    data["SMA50"] = ta.trend.sma_indicator(data["Close"], window=50)
    data["RSI"] = ta.momentum.rsi(data["Close"], window=14)
    
    # Normalize data
    data_norm = (data - data.mean()) / data.std()
    
    return data_norm

# Define function to conduct technical analysis
def conduct_technical_analysis(data):
    # Use moving averages to identify trends
    data["Trend"] = "None"
    data.loc[data["SMA20"] > data["SMA50"], "Trend"] = "Up"
    data.loc[data["SMA20"] < data["SMA50"], "Trend"] = "Down"
    
    # Use RSI to identify oversold/overbought conditions
    data["Oversold"] = False
    data.loc[data["RSI"] < 30, "Oversold"] = True
    data["Overbought"] = False
    data.loc[data["RSI"] > 70, "Overbought"] = True
    
    return data

# Define function to conduct fundamental analysis
def conduct_fundamental_analysis(ticker):
    # Download financial statements
    income_statement = yf.download(ticker, period="ytd", interval="1d", group_by="ticker")["income"]
    balance_sheet = yf.download(ticker, period="ytd", interval="1d", group_by="ticker")["balance"]
    cash_flow_statement = yf.download(ticker, period="ytd", interval="1d", group_by="ticker")["cashflow"]
    
    # Conduct financial ratio analysis
    pe_ratio = income_statement["netIncome"]/income_statement["totalRevenue"]
    debt_to_equity_ratio = balance_sheet["totalLiab"]/balance_sheet["totalStockholderEquity"]
    free_cash_flow = cash_flow_statement["operatingCashflow"] - cash_flow_statement["capitalExpenditures"]
    
    return {"PE Ratio": pe_ratio[-1], "Debt to Equity Ratio": debt_to_equity_ratio[-1], "Free Cash Flow": free_cash_flow[-1]}

#
