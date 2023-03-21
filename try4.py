import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Set the GPU to use for training (if available)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Download and preprocess the data
df = yf.download("AAPL", start="2021-01-01", end="2021-12-31")
df["Date"] = df.index 
df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
df.index = df['Date']
data = df.sort_index(ascending=True, axis=0)
new_dataset = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])
new_dataset['Date'] = data['Date'].values
new_dataset['Close'] = data['Close'].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(new_dataset[['Close']])

# Prepare the training and validation data
train_data = scaled_data[:5000, :]
valid_data = scaled_data[5000:, :]
x_train_data, y_train_data = [], []
for i in range(60, len(train_data)):
    x_train_data.append(train_data[i-60:i, 0])
    y_train_data.append(train_data[i, 0])
x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

# Define the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(1))

# Compile and train the LSTM model
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
history = lstm_model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose='2')

# Prepare the test data and make predictions
inputs_data = scaled_data[len(new_dataset) - len(valid_data) - 60:, :]
inputs_data = np.reshape(inputs_data, (1, inputs_data.shape[0], inputs_data.shape[1]))
predicted_closing_price = lstm_model.predict(inputs_data)
predicted_closing_price = scaler.inverse_transform(predicted_closing_price.reshape(-1, 1))

# Plot the results
train_data = new_dataset[:5000]
valid_data = new_dataset[5000:]
valid_data['Predictions'] = predicted_closing_price
plt.plot(train_data['Close'])
plt.plot(valid_data[['Close', 'Predictions']])
