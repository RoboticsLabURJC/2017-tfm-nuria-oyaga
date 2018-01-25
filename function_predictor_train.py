
"""
function_predictor.py: A script to predict a value from a sequence

"""
__author__ = "Nuria Oyaga"
__date__ = "2017/11/02"


from keras.models import Sequential
from keras.layers import Dense, Dropout

import math
import data_utils
from time import time
import os


if __name__ == '__main__':
    if not os.path.exists('Models'):
        os.mkdir('Models')

    # Load data
    parameters, train_set = data_utils.read_data('functions_dataset/quadratic/quadratic_10_[None]_train.txt')
    _, test_set = data_utils.read_data('functions_dataset/quadratic/quadratic_10_[None]_val.txt')

    # Put the train data into the right shape
    print('Puting the train data into the right shape...')
    trainX, trainY = data_utils.reshape_data(train_set)

    # Put the test data into the right shape
    print('Puting the test data into the right shape...')
    testX, testY = data_utils.reshape_data(test_set)

    # Create and fit Multilayer Perceptron model
    n_layers = 1
    n_neurons = [10]
    activation = 'relu'
    loss = 'mean_squared_error'
    dropout = 'N'

    model = Sequential()
    model.add(Dense(n_neurons[0], input_dim=trainX.shape[1], activation=activation))
    if dropout == 'Y':
        drop = 0.2
        model.add(Dropout(drop))
    model.add(Dense(1))
    model.compile(loss=loss, optimizer='adam')

    n_epochs = 100
    batch_size = 2

    print('Training model...')
    start_time = time()
    model.fit(trainX, trainY,
              epochs=n_epochs,
              batch_size=batch_size,
              verbose=2)

    end_time = time()

    # Estimate model performance
    trainScore = model.evaluate(trainX, trainY, verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
    testScore = model.evaluate(testX, testY, verbose=0)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

    # Save data
    name = parameters[0][4] + '_' + parameters[0][3] + '_' + parameters[0][5] + '_PredictorMultilayerPerceptron' + \
           str(n_layers) + '_' + str(n_neurons) + '_' + activation + '_' + loss
    root = 'Models/' + name

    data_utils.check_dir(root)

    model.save(root + '/' + str(n_epochs) + '_' + str(batch_size) + '_' +
               dropout + '_' + activation + '_' + loss + '.h5')

    file = open(root + '/' + str(n_epochs) + '_' + str(batch_size) + '_' +
                dropout + '_' + activation + '_' + loss + '_info.txt', 'w')
    file.write('Architecture: ' + str(n_layers) + '_' + str(n_neurons) + '\n')
    file.write('Activation: ' + activation + '\n')
    file.write('Loss: ' + loss + '\n')
    file.write('Dropout: ' + dropout + '\n')
    file.write('Epochs: ' + str(n_epochs) + '\n')
    file.write('Batch size: ' + str(batch_size) + '\n')
    file.write('Train data: ' + parameters[0][4] + '_' + parameters[0][3] + '_' + parameters[0][5] + '\n')
    file.write('Train score: ' + str(round(trainScore,2)) + '\n')
    file.write('Test score: ' + str(round(testScore, 2)) + '\n')
    file.write('Execution time: ' + str(round(end_time-start_time, 2)) + '\n')
