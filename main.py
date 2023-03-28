from tkinter.font import names
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
df = yf.download('AAPL', start='2000-01-01', end='2020-12-31')
df['return'] = np.log(df.Close.pct_change()+1)
def lagit(df , lags):
    names =[]
    for i in range(1, lags+1):
        df['lag'+str(i)] = df['return'].shift(i)
        names.append('lag'+str(i))
    return names
lagnames = lagit(df, 5)
df
df.dropna(inplace=True)
model = LinearRegression()
model = model.fit(df[lagnames], df['return'])
df['prediction'] = model.predict(df[lagnames])
df['direction_LR'] = [1 if t > 0 else 0 for t in df['prediction']]
df['strat_LR'] = df['direction_LR'] * df['return']
np.exp(df[['return', 'strat_LR']].sum())
np.exp(df[['return', 'strat_LR']].cumsum()).plot()
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, shuffle=False, test_size=0.2, random_state=0)
train = train.copy()
test = test.copy()
model = LinearRegression()
model.fit(train[lagnames], train['return'])
test['prediction_LR'] = model.predict(test[lagnames])
test['direction_LR'] = [1 if t > 0 else 0 for t in test['prediction_LR']]
test['strat_LR'] = test['direction_LR'] * test['return']
np.exp(test[['return', 'strat_LR']].sum())
(test['direction_LR'].diff() != 0).value_counts()
np.exp(test[['return', 'strat_LR']].cumsum()).plot()
