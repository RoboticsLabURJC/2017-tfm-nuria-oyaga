"""

TFM - main_train.py - Description

"""
__author__ = "Nuria Oyaga"
__date__ = "22/05/2018"


from utils import read_data, reshape_data, check_dirs
import net

import warnings


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    func_type = 'linear'
    net_type = 'mlp'
    n_neurons = [10]
    activation = 'relu'
    loss = 'mean_squared_error'
    dropout = False
    n_epochs = 300
    batch_size = 15
    patience = 10

    data_dir = '/home/nuria/Documents/MOVA/TFM/functions_dataset/' + func_type + '/' + func_type + '_10_[None]_'

    # Load data
    parameters, train_set = read_data(data_dir + 'train.txt')
    _, val_set = read_data(data_dir + 'val.txt')

    # Put the train data into the right shape
    print('Puting the train data into the right shape...')
    trainX, trainY = reshape_data(train_set)

    # Put the validation data into the right shape
    print('Puting the validation data into the right shape...')
    valX, valY = reshape_data(val_set)

    # Model settings
    root = '/home/nuria/Documents/MOVA/TFM/Models/' + net_type.upper() + '/' + func_type
    check_dirs(root)

    filename = root + '/' + parameters[0][4] + '_' + parameters[0][3] + '_' + parameters[0][5] + '_Predictor'

    to_train_net = net.Mlp(n_neurons=n_neurons, activation=activation, loss=loss, dropout=dropout, input_dim=trainX.shape[1])

    to_train_net.create_model()

    to_train_net.train(n_epochs, batch_size, patience, filename, [trainX, trainY], [valX, valY])
