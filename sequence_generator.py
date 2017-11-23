"""
sequence_generator.py: Numbers or frames sequence generator

"""
__author__ = "Nuria Oyaga"
__date__ = "2017/10/06"

import cv2
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import data_utils
import random
import yaml


toGenerate = 'n'  # Choose type to generate: frame('f') or number('n')


# Load the configuration file
conf = yaml.load(open('sequence_generator_config.yml', 'r'))

n_samples = int(conf['n_samples']) # Number of samples to save in dataset (aproximated)
n_points = int(conf['n_points'])  # Number of points used to make prediction
gap = int(conf['gap'])  # Separation between last sample and sample to predict
noise_flag = conf['noise']['flag'] # Introduce noise to the samples


def get_position(x0, t, u_x):
    x = x0 + (t * u_x)  # URM
    return x


def create_frame(x, size):
    frame = np.zeros(size)
    cv2.circle(frame, (int(x), 128), 10, (0, 255, 0), -1)
    cv2.rectangle(frame, (100, 0), (170, size[0]), (255, 0, 0), -1)
    return frame


def get_numbers(func):
    # Get first samples
    numbers = [func(x) for x in range(n_points)]
    # Create number to predict
    numbers.append(func(n_points+gap-1))
    return numbers


def to_file(seq, n_param, file):
    for i,element in enumerate(seq):
        if i == 0:
            file.write('[ ')

        if i == 4:
            file.write(element + ' ')

        elif i == n_param and n_param == 5:
            file.write(str(element) + ' ')
            file.write('][ ')

        elif i == n_param:
            file.write("{0:.4f}".format(element) + ' ')
            file.write('][ ')

        else:
            file.write("{0:.4f}".format(element) + ' ')

        if i == len(seq) - 1:
            file.write('] \n')


if __name__ == '__main__':
    if toGenerate == 'f':  # Generate a directory with 4+1 frames for each speed
        # Create directory
        if os.path.exists('frames'):
            shutil.rmtree('frames')
            os.mkdir('frames')
        else:
            os.mkdir('frames')

        # Set frame size
        imSize = (256, 256, 3)  # (h,w,chanels)

        for u_x in range(5,23):  # Speed in px/t
            # Create 4 initial frames
            for t in range(4):
                cv2.imwrite('frames/frame_'+str(u_x)+'_'+str(t)+'.png',
                            create_frame(get_position(10, t, u_x),imSize))

            # Create frame 10
            cv2.imwrite('frames/frame_' + str(u_x) + '_' + str(10) + '.png',
                        create_frame(get_position(10, 10, u_x),imSize))

    elif toGenerate == 'n':  # Generate a list with 4+1 arrays for each speed
        sequences = []

        func_type = conf['func_type']  # Choose type of function: linear, quadratic or sinusoidal

        # Create samples
        for i in range(n_samples):
            a = random.uniform(-100, 100)
            b = random.uniform(-100, 100)
            c = random.uniform(-100, 100)

            if func_type == 'linear':
                # Set function: ax + by + c = 0
                f = lambda x: (a * x + c) / -b

            elif func_type == 'quadratic':
                while a == 0:
                    a = random.uniform(-100, 100)

                f = lambda x: a * (x ** 2) + b * x + c

            else:
                print('Choose a correct function to generate (linear,quadratic or sinusoidal) ')
                break

            parameters = [a, b, c, gap, func_type]

            num = get_numbers(f)

            if not noise_flag:
                noise_par = [None]
            else:
                mean = int(conf['noise']['mean'])
                stand_deviation = int(conf['noise']['stand_deviation'])

                noise_par= [mean, stand_deviation]

                noise = np.random.normal(mean, stand_deviation, len(num))  # Different noise for each sample

                num = list(num + noise)

            for param in noise_par:
                parameters.append(param)

            for p in parameters[::-1]:
                num.insert(0, p)

            sequences.append(num)
            print(i)


        fraction_test = 0.1
        fraction_validation = 0.1

        # Separate train, test and validation
        test_set_size = int(len(sequences) * fraction_test)
        validation_set_size = int(len(sequences) * fraction_validation)

        train_set = sequences[:-(test_set_size + validation_set_size)]
        test_set = sequences[len(sequences) - (test_set_size + validation_set_size):-validation_set_size]
        validation_set = sequences[-validation_set_size:]

        print(len(train_set) + len(test_set) + len(validation_set) == len(sequences))  # Check the separation

        # Init files
        file_train = open('functions_dataset/' + func_type + '_' + str(gap) + '_' + str(noise_par) + '_train.txt', 'w')
        file_train.write(
            '[ a b c gap ftype noise(mean, standard deviation) ][ x=0:' + str(n_points - 1) + ' x=' +
            str(n_points + gap - 1) + ' ]\n')
        file_test = open('functions_dataset/' + func_type + '_' + str(gap) + '_' + str(noise_par) + '_test.txt', 'w')
        file_test.write(
            '[ a b c gap ftype noise(mean, standard deviation) ][ x=0:' + str(n_points - 1) + ' x=' +
            str(n_points + gap - 1) + ' ]\n')
        file_val = open('functions_dataset/' + func_type + '_' + str(gap) + '_' + str(noise_par) + '_val.txt', 'w')
        file_val.write('[ a b c gap ftype noise(mean, standard deviation) ][ x=0:' + str(n_points - 1) + ' x=' +
                       str(n_points + gap - 1) + ' ]\n')

        # Write files
        # Train
        for element in train_set:
            to_file(element, len(parameters)-1, file_train)

        # Test
        for element in test_set:
            to_file(element, len(parameters)-1, file_test)

        # Validation
        for element in validation_set:
            to_file(element, len(parameters)-1, file_val)


        # Draw an example
        index = random.randrange(0, len(sequences))

        x = sequences[index][6:n_points + 6]
        y = sequences[index][-1:]
        data_utils.draw_data(plt, [x, y], gap)
        plt.title(str(index))

        plt.show()

    else:
        print('Choose a correct type to generate (frame or number) ')