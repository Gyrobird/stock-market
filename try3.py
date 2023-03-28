import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pylab import rcParams
import yfinance as yf
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler

rcParams['figure.figsize'] = 20, 10

df = yf.download('AAPL', start='2021-01-01', end='2021-12-31')
df.index = pd.to_datetime(df.index) # set the index to be datetime
new_dataset = pd.DataFrame(index=range(len(df)), columns=['Close'])
new_dataset['Close'] = df['Close'].values

scaler = MinMaxScaler(feature_range=(0, 1))
final_dataset = scaler.fit_transform(new_dataset.values)
train_data = final_dataset[:5000, :]
valid_data = final_dataset[5000:, :]

x_train_data, y_train_data = [], []
for i in range(60, len(train_data)):
    x_train_data.append(train_data[i-60:i, 0])
    y_train_data.append(train_data[i, 0])

x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')

lstm_model.fit(x_train_data, y_train_data, epochs=1, batch_size='2', verbose='2', validation_split=0.1)

# prepare test data for prediction
inputs_data = new_dataset[len(new_dataset)-len(valid_data)-60:].values
inputs_data = inputs_data.reshape(-1, 1)
inputs_data = scaler.transform(inputs_data)

# prepare test data with sliding window
X_test = []
for i in range(60, inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# load model and make predictions
lstm_model = load_model('saved_model.h5')
if lstm_model is not None:
    predicted_closing_price = lstm_model.predict(X_test)
    predicted_closing_price = scaler.inverse_transform(predicted_closing_price)

    # plot the predictions
    valid_data = pd.DataFrame(valid_data, columns=['Close'])
    valid_data['Predictions'] = predicted_closing_price
    plt.plot(valid_data[['Close']])
    plt.plot(valid_data[['Predictions']])
    plt.show()
else:
    print("Error: Model not found.")
