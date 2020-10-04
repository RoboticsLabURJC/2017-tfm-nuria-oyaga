"""

TFM - Network.py - Description

"""
__author__ = "Nuria Oyaga"
__date__ = "21/05/2018"

from Utils import utils, vect_utils, frame_utils, test_utils

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Flatten, LSTM, ConvLSTM2D, \
    TimeDistributed
from keras.utils import vis_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf

import numpy as np
from time import time


class Net(object):

    def __init__(self, net_type, **kwargs):
        self.net_type = net_type
        self.framework = kwargs['framework']
        print(self.framework)
        if 'model_file' in kwargs.keys():
            self.model_path = kwargs['model_file'][:kwargs['model_file'].rfind("/") + 1]
            if self.framework == "keras":
                self.model = load_model(kwargs['model_file'])
            else:
                self.model = tf.keras.models.load_model(kwargs['model_file'])
        else:
            if self.framework == "keras":
                self.model = Sequential()
            else:
                self.model = tf.keras.Sequential()

            self.dropout = kwargs['dropout']
            if self.dropout:
                self.drop_percentage = kwargs['drop_percentage']
            self.loss = kwargs['loss']
            self.activation = kwargs['activation']
            self.input_shape = kwargs['input_shape']
            self.output_shape = kwargs['output_shape']

    def train(self, n_epochs, batch_size, patience, root, data_train, data_val, batch_data, gauss_pixel, channels):
        utils.check_dirs(root)

        name = root + '/' + str(batch_size) + '_' + str(self.dropout) + '_' + self.activation + '_' + \
            self.loss + '_' + str(patience)

        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=patience)
        checkpoint = ModelCheckpoint(name + '.h5', verbose=1, monitor='val_loss', save_best_only=True, mode='auto')
        print(name)

        print('Training model...')
        start_time = time()
        if batch_data:
            print("Batch data")
            steps_per_epoch = np.ceil(len(data_train) / batch_size)
            validation_steps = np.ceil(len(data_val) / batch_size)
            training_batch_generator = frame_utils.batch_generator(data_train, batch_size, steps_per_epoch,
                                                                   gauss_pixel, channels)
            validation_batch_generator = frame_utils.batch_generator(data_val, batch_size, validation_steps,
                                                                     gauss_pixel, channels)
            model_history = self.model.fit_generator(training_batch_generator,
                                                     epochs=n_epochs, steps_per_epoch=steps_per_epoch,
                                                     validation_data=validation_batch_generator,
                                                     validation_steps=validation_steps,
                                                     callbacks=[early_stopping, checkpoint], verbose=2)

        else:
            print("No batch data")
            model_history = self.model.fit(data_train[0], data_train[1], batch_size=batch_size,
                                           epochs=n_epochs, validation_data=data_val,
                                           callbacks=[early_stopping, checkpoint], verbose=2)
        end_time = time()

        print("End training")

        if len(model_history.epoch) < n_epochs:
            n_epochs = len(model_history.epoch)

        self.save_properties(patience, n_epochs, round(end_time-start_time, 2), name + '_properties')
        utils.save_history(model_history, name)

    def save_properties(self, patience, epochs, train_time, file_path):
        if self.framework == "keras":
            vis_utils.plot_model(self.model, file_path + '.png', show_shapes=True)
        else:
            tf.keras.utils.plot_model(self.model, file_path + '.png', show_shapes=True)

        with open(file_path + '.txt', 'w+') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
            f.write('\n\n-----------------------------------------------------------------------------------\n\n')
            f.write('Patience: ' + str(patience) + '\n')
            f.write('Epochs: ' + str(epochs) + '\n')
            f.write('Execution time: ' + str(train_time) + '\n')

    def test(self, test_x, test_y, gap, data_type, dim):
        predict = self.model.predict(test_x)
        if data_type == "Functions_dataset":
            maximum = [np.max(np.abs(np.append(test_x[i], test_y[i]))) for i in range(len(test_x))]
            predict_values = predict
            real_values = test_y
            v_to_draw = predict_values
        elif data_type == "Vectors_dataset":
            predict_values, real_values, maximum = vect_utils.get_positions(predict, test_y)
            v_to_draw = predict_values
        else:
            raw = True
            if "modeled" in data_type:
                raw = False
            predict_values, real_values, maximum = frame_utils.get_positions(predict, test_y, dim, raw)

            if raw:
                v_to_draw = predict
            else:
                v_to_draw = predict_values

        error, x_error, y_error, relative_error = test_utils.calculate_error(real_values, predict_values, maximum)

        with open(self.model_path + 'error_result.txt', 'w') as file:
            for i in range(error.shape[0]):
                file.write("Processed sample " + str(i) + ": \n")
                file.write("Target position: " + str(real_values[i]) + "\n")
                file.write("Position: " + str(predict_values[i]) + "\n")
                file.write("Error: " + str(np.round(error[i], 2)) + " (" + str(np.round(relative_error[i], 2)) + "%)\n")
                file.write("--------------------------------------------------------------\n")

        # Calculate stats
        test_utils.get_error_stats(test_x, test_y, v_to_draw, gap, data_type, dim,
                                   error, x_error, y_error, relative_error, self.model_path)


class Mlp(Net):

    def __init__(self, **kwargs):
        Net.__init__(self, "MLP", **kwargs)
        if 'model_file' not in kwargs.keys():
            if kwargs['data_type'] == "Function":
                self.create_function_model()
            else:  # kwargs['data_type'] == "Frame"
                if kwargs['complexity'] == "simple":
                    self.create_frame_simple_model()
                else:
                    self.create_frame_complex_model()

    def create_function_model(self):
        print("Creating function MLP model")
        self.model.add(Dense(15, input_shape=self.input_shape, activation=self.activation))

        if self.dropout:
            self.model.add(Dropout(self.drop_percentage))

        self.model.add(Dense(self.output_shape))
        self.model.compile(loss=self.loss, optimizer='adam')

    def create_frame_simple_model(self):
        print("Creating frame simple MLP model")
        self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(10, activation=self.activation),
                                                       input_shape=self.input_shape))

        if self.dropout:
            self.model.add(tf.keras.layers.Dropout(self.drop_percentage))

        self.model.add(tf.keras.layers.Flatten())

        self.model.add(tf.keras.layers.Dense(self.output_shape))
        self.model.compile(loss=self.loss, optimizer='adam')

    def create_frame_complex_model(self):
        print("Creating frame complex MLP model")
        self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(80, activation=self.activation),
                                                       input_shape=self.input_shape))

        self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(20, activation=self.activation)))

        self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(10, activation=self.activation)))

        if self.dropout:
            self.model.add(tf.keras.layers.Dropout(self.drop_percentage))

        self.model.add(tf.keras.layers.Flatten())

        self.model.add(tf.keras.layers.Dense(self.output_shape))
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
            if kwargs['data_type'] == "Vector":
                self.create_vector_model()
            else:  # kwargs['data_type'] == "Frame"
                if kwargs['complexity'] == "simple":
                    self.create_frame_simple_model()
                else:
                    self.create_frame_complex_model()

    def create_vector_model(self):
        print("Creating function LSTM model")
        self.model.add(LSTM(25, input_shape=self.input_shape))

        if self.dropout:
            self.model.add(Dropout(self.drop_percentage))

        self.model.add(Dense(self.output_shape, activation="softmax"))
        self.model.compile(loss=self.loss, optimizer='adam')

    def create_frame_simple_model(self):
        print("Creating frame simple LSTM model")
        self.model.add(LSTM(25, input_shape=self.input_shape))

        if self.dropout:
            self.model.add(Dropout(self.drop_percentage))

        self.model.add(Dense(self.output_shape))
        self.model.compile(loss=self.loss, optimizer='adam')

    def create_frame_complex_model(self):
        print("Creating frame complex LSTM model")

        self.model.add(LSTM(70, return_sequences=True,  input_shape=self.input_shape))
        self.model.add(LSTM(40, return_sequences=True))
        self.model.add(LSTM(25, return_sequences=True))
        self.model.add(LSTM(15, return_sequences=False))

        if self.dropout:
            self.model.add(Dropout(self.drop_percentage))

        self.model.add(Dense(self.output_shape))
        self.model.compile(loss=self.loss, optimizer='adam')


class ConvolutionLstm(Net):
    def __init__(self, **kwargs):
        Net.__init__(self, "lstm", **kwargs)
        if 'model_file' not in kwargs.keys():
            if kwargs['complexity'] == "simple":
                self.create_simple_model()
            elif kwargs['complexity'] == "complex":
                self.create_complex_model()
            elif kwargs['complexity'] == "convLSTM":
                self.create_conv_lstm_model()
            else:
                self.create_complex_conv_lstm_model()

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
        self.model.add(LSTM(50))

        if self.dropout:
            self.model.add(Dropout(self.drop_percentage))

        self.model.add(Dense(self.output_shape, activation="softmax"))
        self.model.compile(loss=self.loss, optimizer='adam')

    def create_conv_lstm_model(self):
        print("Creating convLSTM model")
        self.model.add(TimeDistributed(Conv2D(32, (3, 3), activation=self.activation), input_shape=self.input_shape))
        self.model.add(TimeDistributed(Conv2D(32, (3, 3), activation=self.activation)))
        self.model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
        self.model.add(ConvLSTM2D(filters=5, kernel_size=(3, 3), padding='same'))
        self.model.add(Flatten())

        if self.dropout:
            self.model.add(Dropout(self.drop_percentage))

        self.model.add(Dense(self.output_shape, activation="softmax"))
        self.model.compile(loss=self.loss, optimizer='adam')

    def create_complex_conv_lstm_model(self):
        print("Creating complex convLSTM model")
        self.model.add(TimeDistributed(Conv2D(32, (5, 5), activation=self.activation), input_shape=self.input_shape))
        self.model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
        self.model.add(ConvLSTM2D(filters=20, kernel_size=(5, 5), padding='same', return_sequences=True))
        self.model.add(ConvLSTM2D(filters=15, kernel_size=(7, 7), padding='same', return_sequences=True))
        self.model.add(ConvLSTM2D(filters=10, kernel_size=(7, 7), padding='same', return_sequences=True))
        self.model.add(ConvLSTM2D(filters=5, kernel_size=(9, 9), padding='same', return_sequences=False))
        self.model.add(Flatten())

        if self.dropout:
            self.model.add(Dropout(self.drop_percentage))

        self.model.add(Dense(self.output_shape, activation="softmax"))
        self.model.compile(loss=self.loss, optimizer='adam')
