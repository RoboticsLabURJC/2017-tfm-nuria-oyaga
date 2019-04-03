"""

TFM - main_train.py - Description

"""
__author__ = "Nuria Oyaga"
__date__ = "22/05/2018"

from Utils import utils, func_utils, vect_utils, frame_utils
from Network import Net

import numpy as np


if __name__ == '__main__':
    conf = utils.get_config_file()

    net_type = conf['net_type']
    activation = conf['activation']
    dropout = conf['dropout']['flag']
    drop_percentage = float(conf['dropout']['percentage'])
    n_epochs = conf['n_epochs']
    batch_size = conf['batch_size']
    patience = conf['patience']

    data_dir = conf['data_dir']
    data_type = data_dir.split('/')[5]
    func_type = data_dir.split('/')[6]

    root = conf['root'] + net_type.upper() + '/' + data_type + '/' + func_type
    utils.check_dirs(root)
    version = conf['version']

    print('Puting the data into the right shape...')

    if data_type == 'Functions_dataset':
        print('Training with functions')
        loss = conf['func_loss']
        # Load data
        parameters, train_set = func_utils.read_function_data(data_dir + 'train.txt')
        _, val_set = func_utils.read_function_data(data_dir + 'val.txt')

        if func_type == 'sinusoidal':
            filename = root + '/' + parameters[0][5] + '_' + parameters[0][4] + '_' + parameters[0][6] + '_Predictor'
        else:
            filename = root + '/' + parameters[0][4] + '_' + parameters[0][3] + '_' + parameters[0][5] + '_Predictor'

        # Put the train data into the right shape
        trainX, trainY = func_utils.reshape_function_data(train_set)

        # Put the validation data into the right shape
        valX, valY = func_utils.reshape_function_data(val_set)

        # Model settings
        in_dim = trainX.shape[1:]
        out_dim = 1
        to_train_net = Net.Mlp(activation=activation, loss=loss, dropout=dropout,
                               drop_percentage=drop_percentage, input_shape=trainX[0].shape, output_shape=out_dim)

    elif data_type == 'Vectors_dataset':
        print('Training with vectors')
        loss = conf['vect_loss']
        # Load data
        _, train_set = vect_utils.read_vector_data(data_dir + 'train/samples')
        _, val_set = vect_utils.read_vector_data(data_dir + 'val/samples')
        filename = root

        # Put the train data into the right shape
        trainX, trainY = vect_utils.reshape_vector_data(train_set)

        # Put the validation data into the right shape
        valX, valY = vect_utils.reshape_vector_data(val_set)

        # Model settings
        in_dim = trainX.shape[1:]
        out_dim = np.prod(in_dim[1:])
        if net_type == "NoRec":
            to_train_net = Net.Convolution1D(activation=activation, loss=loss, dropout=dropout,
                                             drop_percentage=drop_percentage, input_shape=in_dim,
                                             output_shape=out_dim)
        else:  # net_type == "Rec"
            to_train_net = Net.Lstm(activation=activation, loss=loss, dropout=dropout,
                                    drop_percentage=drop_percentage, input_shape=in_dim,
                                    output_shape=out_dim)

    else:  # data_type == 'Frames_dataset':
        print('Training with frames')
        loss = conf['vect_loss']
        # Load data
        channels = False
        if net_type == "Rec":
            channels = True

        _, trainX, trainY = frame_utils.read_frame_data(data_dir + 'train/samples', channels)
        _, valX, valY = frame_utils.read_frame_data(data_dir + 'val/samples', channels)
        filename = root

        # Model settings
        in_dim = trainX.shape[1:]
        out_dim = np.prod(in_dim[1:])
        if net_type == "NoRec":
            to_train_net = Net.Convolution2D(activation=activation, loss=loss, dropout=dropout,
                                             drop_percentage=drop_percentage, input_shape=in_dim,
                                             output_shape=out_dim)
        else:
            to_train_net = Net.ConvolutionLstm(activation=activation, loss=loss, dropout=dropout,
                                               drop_percentage=drop_percentage, input_shape=in_dim,
                                               output_shape=out_dim)
    print('Training')

    to_train_net.train(n_epochs, batch_size, patience, filename, [trainX, trainY], [valX, valY])
