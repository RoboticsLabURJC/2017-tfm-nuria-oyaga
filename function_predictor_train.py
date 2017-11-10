
"""
function_predictor.py: A script to predict a value from a sequence

"""
__author__ = "Nuria Oyaga"
__date__ = "2017/11/02"


from keras.models import Sequential
from keras.layers import Dense

import numpy as np
import math
import data_utils


if __name__ == '__main__':

    n_epochs = 300
    batch_size = 2

    # Load data
    train_set = data_utils.read_data('functions_dataset/quadratic_train.txt')
    test_set = data_utils.read_data('functions_dataset/quadratic_test.txt')

    # Put the train data into the right shape
    print('Puting the train data into the right shape...')
    trainX, trainY = data_utils.reshape_data(train_set)
    print(trainX.shape)

    # Put the test data into the right shape
    print('Puting the test data into the right shape...')
    testX, testY = data_utils.reshape_data(test_set)

    # Create and fit Multilayer Perceptron model
    model = Sequential()
    model.add(Dense(8, input_dim=trainX.shape[1], activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    print('Training model...')
    model.fit(trainX, trainY,
              epochs=n_epochs,
              batch_size=batch_size,
              verbose=2)

    model.save('Models/QuadraticPredictorMultilayerPerceptron' + str(n_epochs) + '.h5')

    # Estimate model performance
    trainScore = model.evaluate(trainX, trainY, verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
    testScore = model.evaluate(testX, testY, verbose=0)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))