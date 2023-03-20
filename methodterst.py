import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Download historical stock data for Apple (AAPL)
ticker = yf.Ticker("AAPL")
df = ticker.history(period="max")

# Preprocess the data using MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Split the data into training and testing sets
train_data = scaled_data[:int(len(scaled_data)*0.8)]
test_data = scaled_data[int(len(scaled_data)*0.8):]

# Create X and y arrays for training
X_train = []
y_train = []
for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Train a linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Create X and y arrays for testing
X_test = []
y_test = []
for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i,0])
    y_test.append(test_data[i,0])
X_test, y_test = np.array(X_test), np.array(y_test)

# Make predictions on the test data
y_pred = regressor.predict(X_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
