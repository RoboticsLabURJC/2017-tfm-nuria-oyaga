"""

TFM - main_train.py - Description

"""
__author__ = "Nuria Oyaga"
__date__ = "22/05/2018"


from utils import read_data, reshape_data, check_dirs, get_config_file
import net


if __name__ == '__main__':
    conf = get_config_file()

    net_type = conf['net_type']
    n_neurons = conf['n_neurons']
    activation = conf['activation']
    loss = conf['loss']
    dropout = conf['dropout']['flag']
    drop_percentage = float(conf['dropout']['percentage'])
    n_epochs = conf['n_epochs']
    batch_size = conf['batch_size']
    patience = conf['patience']

    data_dir = conf['data_dir']
    func_type = data_dir.split('/')[7]

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
    root = conf['root'] + net_type.upper() + '/' + func_type
    check_dirs(root)

    filename = root + '/' + parameters[0][4] + '_' + parameters[0][3] + '_' + parameters[0][5] + '_Predictor'

    to_train_net = net.Mlp(n_neurons=n_neurons, activation=activation, loss=loss, dropout=dropout,
                           drop_percentage=drop_percentage, input_dim=trainX.shape[1])

    to_train_net.create_model()

    to_train_net.train(n_epochs, batch_size, patience, filename, [trainX, trainY], [valX, valY])
