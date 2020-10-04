"""

TFM - main_gen.py - Description

"""
__author__ = "Nuria Oyaga"
__date__ = "23/04/2018"

import sys
sys.path.insert(0, '/home/docker/2017-tfm-nuria-oyaga')

from Utils.utils import write_header, check_dirs, get_config_file
import Function, Vectors, Frames, Shapes


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
    data_dir = conf['root'] + "_" + str(gap)

    if to_generate == 'n':

        func_type = conf['func_type']  # Type of function

        # Create directory
        data_dir += "/Functions_dataset/" + func_type + '_' + str(n_samples)
        check_dirs(data_dir, True)

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
                func = Function.Linear(a_limit, b_limit, c_limit, y_limit, noise_parameters, n_points, gap)

            elif func_type == 'quadratic':
                if i == 0:
                    header = '[ a b c gap ftype noise(mean, standard deviation) ]'+ \
                             '[ x=0:' + str(n_points - 1) + ' x=' + str(n_points + gap - 1) + ' ]\n'
                a_limit = 50
                b_limit = 100
                c_limit = 100
                y_limit = 400
                func = Function.Quadratic(a_limit, b_limit, c_limit, y_limit, noise_parameters, n_points, gap)

            elif func_type == 'sinusoidal':
                if i == 0:
                    header = '[ a f theta fs gap ftype noise(mean, standard deviation) ]' +\
                             '[ x=0:' + str(n_points - 1) + ' x=' + str(n_points + gap - 1) + ' ]\n'

                a_limit = 10
                f_limit = 5
                theta_limit = 360
                func = Function.Sinusoidal(a_limit, f_limit, theta_limit, noise_parameters, n_points, gap)

            elif func_type == 'poly3':
                if i == 0:
                    header = '[ a b c d gap ftype noise(mean, standard deviation) ]' + \
                             '[ x=0:' + str(n_points - 1) + ' x=' + str(n_points + gap - 1) + ' ]\n'
                a_limit = 10
                b_limit = 20
                c_limit = 50
                d_limit = 100
                y_limit = 1000
                func = Function.Poly3(a_limit, b_limit, c_limit, d_limit, y_limit, noise_parameters, n_points, gap)

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
                func = Function.Poly4(a_limit, b_limit, c_limit, d_limit, e_limit, y_limit, noise_parameters,
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

    elif to_generate == 'v':
        vector_len = conf['vector_len']
        motion_type = conf['motion_type']  # Type of motion

        data_dir += "/Vectors_dataset/" + motion_type + '_' + str(n_samples) + "_" + str(vector_len)
        check_dirs(data_dir, True)

        for i in range(n_samples):
            if i % 100 == 0 or i == n_samples - 1:
                print(i)

            if motion_type == 'URM':
                if i == 0:
                    header = 'x0 u_x n_points gap motion noise(mean,std)\n'
                sample = Vectors.URM(noise_parameters, n_points, gap, vector_len)

            if split_flag:
                if i == 0:
                    train_path = data_dir + '/' + motion_type + '_' + str(gap) + '_' \
                                     + str(noise_parameters) + '_train'
                    check_dirs(train_path + '/samples')

                    test_path = data_dir + '/' + motion_type + '_' + str(gap) + '_' \
                                    + str(noise_parameters) + '_test'
                    check_dirs(test_path + '/samples')

                    val_path = data_dir + '/' + motion_type + '_' + str(gap) + '_' \
                                   + str(noise_parameters) + '_val'
                    check_dirs(val_path + '/samples')

                    write_header(train_path + '/parameters.txt', header)
                    write_header(test_path + '/parameters.txt', header)
                    write_header(val_path + '/parameters.txt', header)

                    n_test = int(n_samples * float(conf['split']['fraction_test']))
                    n_val = int(n_samples * float(conf['split']['fraction_validation']))
                    n_train = n_samples - n_val - n_test

                if i < n_train:
                    sample.save(train_path + '/samples/sample' + str(i) + '.png', train_path + '/parameters.txt')
                elif i < n_train + n_test:
                    sample.save(test_path + '/samples/sample' + str(i) + '.png', test_path + '/parameters.txt')
                else:
                    sample.save(val_path + '/samples/sample' + str(i) + '.png', val_path + '/parameters.txt')

            else:
                if i == 0:
                    data_path = data_dir + '/' + motion_type + '_' + str(gap) + '_' \
                                 + str(noise_parameters)
                    check_dirs(data_path + '/samples')

                    write_header(data_path + '/parameters.txt', header)

                sample.save(data_path + '/samples/sample' + str(i) + '.png', data_path + '/parameters.txt')
    else:  # to_generate == "f"
        h = conf['height']
        w = conf['width']
        obj_shape = conf['object']
        obj_color = conf['obj_color']
        motion_type = conf['motion_type']
        dof = conf['dof']
        data_dir += "/Frames_dataset/" + motion_type + '_' + obj_shape + '_' + str(obj_color) + '_' + dof \
                    + '_' + str(n_samples)
        if obj_shape == 'point':
            shape = Shapes.Point(obj_color)
        else:
            circle_parameters = conf['circle_parameters']
            shape = Shapes.Circle(obj_color, circle_parameters['radius'])
            data_dir = data_dir + '_' + str(circle_parameters['radius'])

        data_dir = data_dir + "_" + str(h) + "_" + str(w)
        check_dirs(data_dir, True)

        for i in range(n_samples):
            if i % 100 == 0 or i == n_samples - 1:
                print(i)

            if motion_type == 'URM':
                if i == 0:
                    header = 'x0 u_x y0 n_points gap motion noise(mean, standard deviation)\n'
                sample = Frames.URM(noise_parameters, n_points, gap, h, w, shape, dof)
            elif motion_type == 'linear':
                if i == 0:
                    header = 'x0 u_x y0 m n_points gap motion noise(mean, standard deviation)\n'
                sample = Frames.Linear(noise_parameters, n_points, gap, h, w, shape, dof)

            elif motion_type == 'parabolic':
                if i == 0:
                    header = 'x0 u_x y0 a b n_points gap motion noise(mean, standard deviation)\n'
                sample = Frames.Parabolic(noise_parameters, n_points, gap, h, w, shape, dof)

            else:  # motion_type == 'sinusoidal'
                if i == 0:
                    header = 'x0 u_x y0 a b c f n_points gap motion noise(mean, standard deviation)\n'
                sample = Frames.Sinusoidal(noise_parameters, n_points, gap, h, w, shape, dof)

            if split_flag:
                if i == 0:
                    train_path = data_dir + '/' + motion_type + '_' + str(gap) + '_' \
                                     + str(noise_parameters) + '_train'
                    check_dirs(train_path + '/raw_samples')
                    check_dirs(train_path + '/modeled_samples')

                    test_path = data_dir + '/' + motion_type + '_' + str(gap) + '_' \
                                    + str(noise_parameters) + '_test'
                    check_dirs(test_path + '/raw_samples')
                    check_dirs(test_path + '/modeled_samples')

                    val_path = data_dir + '/' + motion_type + '_' + str(gap) + '_' \
                                   + str(noise_parameters) + '_val'
                    check_dirs(val_path + '/raw_samples')
                    check_dirs(val_path + '/modeled_samples')

                    write_header(train_path + '/parameters.txt', header)
                    write_header(test_path + '/parameters.txt', header)
                    write_header(val_path + '/parameters.txt', header)

                    n_test = int(n_samples * float(conf['split']['fraction_test']))
                    n_val = int(n_samples * float(conf['split']['fraction_validation']))
                    n_train = n_samples - n_val - n_test

                if i < n_train:
                    sample.save(train_path + '/raw_samples/sample' + str(i), train_path + '/parameters.txt',
                                train_path + '/modeled_samples/sample' + str(i) + '.txt')
                elif i < n_train + n_test:
                    sample.save(test_path + '/raw_samples/sample' + str(i), test_path + '/parameters.txt',
                                test_path + '/modeled_samples/sample' + str(i) + '.txt')
                else:
                    sample.save(val_path + '/raw_samples/sample' + str(i), val_path + '/parameters.txt',
                                val_path + '/modeled_samples/sample' + str(i) + '.txt')

            else:
                if i == 0:
                    data_path = data_dir + '/' + motion_type + '_' + str(gap) + '_' \
                                 + str(noise_parameters)
                    check_dirs(data_path + '/samples')

                    write_header(data_path + '/parameters.txt', header)

                sample.save(data_path + '/samples/sample' + str(i), data_path + '/parameters.txt')
