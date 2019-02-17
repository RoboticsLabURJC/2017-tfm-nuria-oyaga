"""

TFM - main_train.py - Description

"""
__author__ = "Nuria Oyaga"
__date__ = "22/05/2018"


import net, utils


if __name__ == '__main__':
    conf = utils.get_config_file()

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
    data_type = data_dir.split('/')[5]
    func_type = data_dir.split('/')[6]

    root = conf['root'] + net_type.upper() + '/' + data_type + '/' + func_type
    utils.check_dirs(root)
    version = conf['version']

    print('Puting the data into the right shape...')

    if data_type == 'Functions_dataset':
        print('Training with functions')
        # Load data
        parameters, train_set = utils.read_function_data(data_dir + 'train.txt')
        _, val_set = utils.read_function_data(data_dir + 'val.txt')
        in_dim = (20,)
        out_dim = 1

        if func_type == 'sinusoidal':
            filename = root + '/' + parameters[0][5] + '_' + parameters[0][4] + '_' + parameters[0][6] + '_Predictor'
        else:
            filename = root + '/' + parameters[0][4] + '_' + parameters[0][3] + '_' + parameters[0][5] + '_Predictor'

        # Put the train data into the right shape
        trainX, trainY = utils.reshape_function_data(train_set)

        # Put the validation data into the right shape
        valX, valY = utils.reshape_function_data(val_set)

        # Model settings
        to_train_net = net.Mlp(n_neurons=n_neurons, activation=activation, loss=loss, dropout=dropout,
                               drop_percentage=drop_percentage, input_shape=trainX[0].shape, output_shape=out_dim)

    elif data_type == 'Vectors_dataset':
        print('Training with vectors')
        # Load data
        train_set = utils.read_vector_data(data_dir + 'train/samples')
        val_set = utils.read_vector_data(data_dir + 'val/samples')
        in_dim = (20, 320)
        out_dim = 320
        filename = root

        # Put the train data into the right shape
        trainX, trainY = utils.reshape_vector_data(train_set)


        # Put the validation data into the right shape
        valX, valY = utils.reshape_vector_data(val_set)

        # Model settings
        to_train_net = net.Convolution1D(n_neurons=n_neurons, activation=activation, loss=loss, dropout=dropout,
                                         drop_percentage=drop_percentage, input_shape=in_dim,
                                         output_shape=out_dim)
    print('Training')

    to_train_net.train(n_epochs, batch_size, patience, filename, [trainX, trainY], [valX, valY])
