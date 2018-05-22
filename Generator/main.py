"""

TFM - main.py - Description

"""
__author__ = "Nuria Oyaga"
__date__ = "23/04/2018"

from utils import write_header, check_dirs, get_config_file
import functions


if __name__ == '__main__':

    conf = get_config_file()

    n_samples = int(conf['n_samples'])  # Number of samples to save in the data set
    n_points = int(conf['n_points'])  # Number of points used to make prediction
    gap = int(conf['gap'])  # Separation between last sample and sample to predict
    noise_flag = conf['noise']['flag']  # Introduce noise to the samples
    split_flag = conf['split']['flag']
    if noise_flag:
        mean = float(conf['noise']['mean'])
        stand_deviation = float(conf['noise']['stand_deviation'])
        noise_parameters = [mean, stand_deviation]
    else:
        noise_parameters = [None]

    to_generate = conf['to_generate']  # Type to generate

    if to_generate == 'n':
        func_type = conf['func_type']  # Type of function

        # Create directory
        data_dir = conf['root'] + func_type
        check_dirs(data_dir)

        for i in range(n_samples):
            if i % 100 == 0 or i == n_samples - 1:
                print(i)
            if func_type == 'linear':
                if i == 0:
                    header = '[ a b c gap ftype noise(mean, standard deviation) ]' + \
                             '[ x=0:' + str(n_points - 1) + ' x=' + str(n_points + gap - 1) + ' ]\n'
                a_limit = 100
                b_limit = a_limit
                c_limit = a_limit
                y_limit = 200
                func = functions.Linear(a_limit, b_limit, c_limit, y_limit, noise_parameters, n_points, gap)

            elif func_type == 'quadratic':
                if i == 0:
                    header = '[ a b c gap ftype noise(mean, standard deviation) ]'+ \
                             '[ x=0:' + str(n_points - 1) + ' x=' + str(n_points + gap - 1) + ' ]\n'
                a_limit = 50
                b_limit = 100
                c_limit = 100
                y_limit = 400
                func = functions.Quadratic(a_limit, b_limit, c_limit, y_limit, noise_parameters, n_points, gap)

            elif func_type == 'sinusoidal':
                if i == 0:
                    header = '[ a f theta fs gap ftype noise(mean, standard deviation) ]' +\
                             '[ x=0:' + str(n_points - 1) + ' x=' + str(n_points + gap - 1) + ' ]\n'

                a_limit = 10
                f_limit = 5
                theta_limit = 360
                func = functions.Sinusoidal(a_limit, f_limit, theta_limit, noise_parameters, n_points, gap)

            elif func_type == 'poly3':
                if i == 0:
                    header = '[ a b c d gap ftype noise(mean, standard deviation) ]' + \
                             '[ x=0:' + str(n_points - 1) + ' x=' + str(n_points + gap - 1) + ' ]\n'
                a_limit = 10
                b_limit = 20
                c_limit = 50
                d_limit = 100
                y_limit = 1000
                func = functions.Poly3(a_limit, b_limit, c_limit, d_limit, y_limit, noise_parameters, n_points, gap)

            elif func_type == 'poly4':
                if i == 0:
                    header = '[ a b c d gap ftype noise(mean, standard deviation) ]' + \
                             '[ x=0:' + str(n_points - 1) + ' x=' + str(n_points + gap - 1) + ' ]\n'
                a_limit = 5
                b_limit = 10
                c_limit = 20
                d_limit = 30
                e_limit = 50
                y_limit = 100000
                func = functions.Poly4(a_limit, b_limit, c_limit, d_limit, e_limit, y_limit, noise_parameters,
                                       n_points, gap)

            if split_flag:
                if i == 0:
                    train_filename = data_dir + '/' + func_type + '_' + str(gap) + '_' \
                                     + str(noise_parameters) + '_train.txt'
                    test_filename = data_dir + '/' + func_type + '_' + str(gap) + '_' \
                                    + str(noise_parameters) + '_test.txt'
                    val_filename = data_dir + '/' + func_type + '_' + str(gap) + '_' \
                                   + str(noise_parameters) + '_val.txt'

                    write_header(train_filename, header)
                    write_header(test_filename, header)
                    write_header(val_filename, header)

                    n_test = int(n_samples * float(conf['split']['fraction_test']))
                    n_val = int(n_samples * float(conf['split']['fraction_validation']))
                    n_train = n_samples - n_val - n_test

                if i < n_train:
                    func.write(train_filename)
                elif i < n_train + n_test:
                    func.write(test_filename)
                else:
                    func.write(val_filename)

            else:
                if i == 0:
                    filename = data_dir + '/' + func_type + '_' + str(gap) + '_' + \
                               str(noise_parameters) + '_dataset.txt'
                    write_header(filename, header)

                func.write(filename)
