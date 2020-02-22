"""

TFM - Network.py - Description

"""
__author__ = "Nuria Oyaga"
__date__ = "21/05/2018"

from Utils import utils, vect_utils, frame_utils, test_utils

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Flatten, LSTM, ConvLSTM2D, TimeDistributed
from keras.utils import vis_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
from time import time


class Net(object):

    def __init__(self, net_type, **kwargs):
        self.net_type = net_type
        if 'model_file' in kwargs.keys():
            self.model = load_model(kwargs['model_file'])
        else:
            self.model = Sequential()
            self.dropout = kwargs['dropout']
            if self.dropout:
                self.drop_percentage = kwargs['drop_percentage']
            self.loss = kwargs['loss']
            self.activation = kwargs['activation']
            self.input_shape = kwargs['input_shape']
            self.output_shape = kwargs['output_shape']

    def train(self, n_epochs, batch_size, patience, root, data_train, data_val):
        utils.check_dirs(root)
        name = root + '/' + str(batch_size) + '_' + str(self.dropout) + '_' + self.activation + '_' + \
            self.loss + '_' + str(patience)

        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=patience)
        checkpoint = ModelCheckpoint(name + '.h5', verbose=1, monitor='val_loss', save_best_only=True, mode='auto')

        print('Training model...')
        start_time = time()
        model_history = self.model.fit(data_train[0], data_train[1], batch_size=batch_size,
                                       epochs=n_epochs, validation_data=data_val,
                                       callbacks=[early_stopping, checkpoint], verbose=2)
        end_time = time()

        if len(model_history.epoch) < n_epochs:
            n_epochs = len(model_history.epoch)

        train_score = self.model.evaluate(data_train[0], data_train[1], verbose=0)
        val_score = self.model.evaluate(data_val[0], data_val[1], verbose=0)

        self.save_properties(patience, n_epochs, [train_score, val_score],
                             round(end_time-start_time, 2), name + '_properties')
        utils.save_history(model_history, name)

    def save_properties(self, patience, epochs, scores, train_time, file_path):
        vis_utils.plot_model(self.model, file_path + '.png', show_shapes=True)

        with open(file_path + '.txt', 'w+') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
            f.write('\n\n-----------------------------------------------------------------------------------\n\n')
            f.write('Patience: ' + str(patience) + '\n')
            f.write('Epochs: ' + str(epochs) + '\n')
            f.write('Train score: ' + str(round(scores[0], 2)) + '\n')
            f.write('Test score: ' + str(round(scores[1], 2)) + '\n')
            f.write('Execution time: ' + str(train_time) + '\n')

    def test(self, test_x, test_y, gap, data_type):
        predict = self.model.predict(test_x)
        if data_type == "Functions_dataset":
            maximum = [np.max(np.abs(np.append(test_x[i], test_y[i]))) for i in range(len(test_x))]
            predict_values = predict
            real_values = test_y
        elif data_type == "Vectors_dataset" :
            predict_values, real_values, maximum = vect_utils.get_positions(predict, test_y)
        else:  # data_type == "Frames_dataset"
            predict_values, real_values, maximum = frame_utils.get_positions(predict, test_y,
                                                                             (test_x.shape[2], test_x.shape[3]))

        error, relative_error = test_utils.calculate_error(real_values, predict_values, maximum)

        # Calculate stats
        error_stats, rel_error_stats = test_utils.get_errors_statistics(error, relative_error)

        # Draw error percentage
        test_utils.error_histogram(relative_error)

        # Draw the max errors points
        test_utils.draw_max_error_samples(test_x, test_y, predict, gap, error_stats, rel_error_stats, data_type)


class Mlp(Net):

    def __init__(self, **kwargs):
        Net.__init__(self, "MLP", **kwargs)
        if 'model_file' not in kwargs.keys():
            self.create_model()

    def create_model(self):
        print("Creating MLP model")
        self.model.add(Dense(15, input_shape=self.input_shape, activation=self.activation))

        if self.dropout:
            self.model.add(Dropout(self.drop_percentage))

        self.model.add(Dense(self.output_shape))
        self.model.compile(loss=self.loss, optimizer='adam')


class Convolution1D(Net):

    def __init__(self, **kwargs):
        Net.__init__(self, "Conv1D", **kwargs)
        if 'model_file' not in kwargs.keys():
            self.create_model()

    def create_model(self):
        print("Creating 1D convolutional model")
        self.model.add(Conv1D(64, 3, activation=self.activation, input_shape=self.input_shape))
        self.model.add(Conv1D(64, 3, activation=self.activation))

        self.model.add(MaxPooling1D(16))

        if self.dropout:
            self.model.add(Dropout(self.drop_percentage))

        self.model.add(Flatten())

        self.model.add(Dense(self.output_shape, activation="softmax"))
        self.model.compile(loss=self.loss, optimizer='adam')


class Convolution2D(Net):

    def __init__(self, **kwargs):
        Net.__init__(self, "Conv2D", **kwargs)
        if 'model_file' not in kwargs.keys():
            if kwargs['complexity'] == "simple":
                self.create_simple_model()
            else:
                self.create_complex_model()

    def create_simple_model(self):
        print("Creating simple 2D convolutional model")
        self.model.add(Conv2D(32, (3, 3), activation=self.activation, input_shape=self.input_shape))
        self.model.add(Conv2D(32, (3, 3), activation=self.activation))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        if self.dropout:
            self.model.add(Dropout(self.drop_percentage))

        self.model.add(Flatten())
        self.model.add(Dense(self.output_shape, activation="softmax"))
        self.model.compile(loss=self.loss, optimizer='adam')

    def create_complex_model(self):
        print("Creating complex 2D convolutional model")
        self.model.add(Conv2D(32, (5, 5), activation=self.activation, input_shape=self.input_shape))
        self.model.add(Conv2D(40, (3, 3), activation=self.activation))
        self.model.add(Conv2D(64, (3, 3), activation=self.activation))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        if self.dropout:
            self.model.add(Dropout(self.drop_percentage))

        self.model.add(Flatten())
        self.model.add(Dense(self.output_shape, activation="softmax"))
        self.model.compile(loss=self.loss, optimizer='adam')


class Lstm(Net):
    def __init__(self, **kwargs):
        Net.__init__(self, "lstm", **kwargs)
        if 'model_file' not in kwargs.keys():
            self.create_model()

    def create_model(self):
        print("Creating LSTM model")
        self.model.add(LSTM(25, input_shape=self.input_shape))

        if self.dropout:
            self.model.add(Dropout(self.drop_percentage))

        self.model.add(Dense(self.output_shape, activation="softmax"))
        self.model.compile(loss=self.loss, optimizer='adam')


class ConvolutionLstm(Net):
    def __init__(self, **kwargs):
        Net.__init__(self, "lstm", **kwargs)
        if 'model_file' not in kwargs.keys():
            if kwargs['complexity'] == "simple":
                self.create_simple_model()
            elif kwargs['complexity'] == "complex":
                self.create_complex_model()
	    else:
	        self.create_conv_lstm_model()

    def create_simple_model(self):
        print("Creating simple convolution LSTM model")
        self.model.add(TimeDistributed(Conv2D(32, (3, 3), activation=self.activation), input_shape=self.input_shape))
        self.model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
        self.model.add(TimeDistributed(Flatten()))
        self.model.add(LSTM(25))

        if self.dropout:
            self.model.add(Dropout(self.drop_percentage))

        self.model.add(Dense(self.output_shape, activation="softmax"))
        self.model.compile(loss=self.loss, optimizer='adam')    

    def create_complex_model(self):
        print("Creating complex convolution LSTM model")
        self.model.add(TimeDistributed(Conv2D(32, (3, 3), activation=self.activation), input_shape=self.input_shape))
        self.model.add(TimeDistributed(Conv2D(32, (3, 3), activation=self.activation)))
        self.model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
        self.model.add(TimeDistributed(Flatten()))
        self.model.add(LSTM(25, return_sequences=True))
        self.model.add(LSTM(50, return_sequences=True))

        if self.dropout:
            self.model.add(Dropout(self.drop_percentage))

        self.model.add(Dense(self.output_shape, activation="softmax"))
        self.model.compile(loss=self.loss, optimizer='adam')

     def create_conv_lstm_model(self):
        print("Creating convLSTM model")
        self.model.add(TimeDistributed(Conv2D(32, (3, 3), activation=self.activation), input_shape=self.input_shape))
        self.model.add(TimeDistributed(Conv2D(32, (3, 3), activation=self.activation)))
        self.model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
        self.model.add(ConvLSTM2D(filters=5, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
        self.model.add(TimeDistributed(Flatten()))

        if self.dropout:
            self.model.add(Dropout(self.drop_percentage))

        self.model.add(Dense(self.output_shape, activation="softmax"))
        self.model.compile(loss=self.loss, optimizer='adam')

