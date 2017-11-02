"""
LSTM_example.py: LSTM prediction example

Code from tutorial: https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/

"""
__author__ = "Nuria Oyaga"
__date__ = "2017/10/06"

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot as plt
import numpy as np


# Load dataset
def parser(x):
    date = x.split('-')
    year = 2013 + int(date[0])
    return datetime.strptime(str(year)+'-'+date[1], '%Y-%m')


# Frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]  # Using the observation from the last time step (t-1) as the input
                                                      # and the observation at the current time step (t) as the output.
                                                      # Push all values in a series down by a specified number places
    columns.append(df)
    df = concat(columns, axis=1)  #Concatenate two series together to create a DataFrame ready for supervised learning
    df.fillna(0, inplace=True)
    return df


# Create a differenced series, the changes to the observations from one time step to the next. Removes the trend
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# Invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# Scale train and test data to [-1, 1] (tanh range)
# experiment fair, the scaling coefficients (min and max) values must be calculated
# on the training dataset and applied to scale the test dataset and any forecasts
def scale(train, test):
    # Fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # Transform train
    train = train.reshape(train.shape[0], train.shape[1])  # MinMaxScaler requires data provided in a matrix format with
                                                           # rows and columns. Reshape before transforming.
    train_scaled = scaler.transform(train)
    # Transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# Inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


# Fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])  # Reshaped into the Samples/TimeSteps/Features format

    # 1 hidden layer
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Because the network is stateful, we must control when the internal state is reset.
    # We must manually manage the training process one epoch at a time across the desired number of epochs
    for i in range(nb_epoch):
        print(i)
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model


# Make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]


if __name__ == '__main__':
    # Load data
    series = read_csv('shampoo-sales.csv',
                      header=0,  # Row number(s) to use as the column names, and the start of the data.
                      parse_dates=[0],  # Try parsing column 0 as a separate date column. If contains an unparseable date,
                                        # returned unaltered as an object data type
                      index_col=0,  # Column to use as the row labels of the DataFrame.
                      squeeze=True,  # If the parsed data only contains one column then return a Series
                      date_parser=parser)  # Function to use for converting a sequence of string columns to an array
                                           # of datetime instances.

    # Transform data to be stationary
    raw_values = series.values
    diff_values = difference(raw_values, 1)

    # Transform data to supervised learning
    supervised = timeseries_to_supervised(diff_values, 1)
    supervised_values = supervised.values

    # Split data into train and test-sets
    train, test = supervised_values[0:-12], supervised_values[-12:]

    # Transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train, test)

    # Fit the model
    lstm_model = fit_lstm(train_scaled,
                          1,  # The batch_size must be set to 1.
                              # It must be a factor of the size of the training and test datasets.
                          1500,
                          1)

    # Forecast the entire training dataset to build up state for forecasting
    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    lstm_model.predict(train_reshaped, batch_size=1)

    # Walk-forward validation on the test data
    predictions = list()
    for i in range(len(test_scaled)):
        # Make one-step forecast
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        yhat = forecast_lstm(lstm_model, 1, X)
        # Invert scaling
        yhat = invert_scale(scaler, X, yhat)
        # Invert differencing
        yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
        # Store forecast
        predictions.append(yhat)
        expected = raw_values[len(train) + i + 1]
        print('Month=%d, Predicted=%f, Expected=%f' % (i + 1, yhat, expected))

    # Report performance
    rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
    print('Test RMSE: %.3f' % rmse)
    # Line plot of observed vs predicted
    plt.plot(raw_values[-12:])
    plt.plot(predictions)
    plt.show()