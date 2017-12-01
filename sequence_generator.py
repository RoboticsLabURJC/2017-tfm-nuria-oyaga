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
    """
    Function to get the position of an object in a frame

    :param x0: Initial position
    :param t: Time instant
    :param u_x: Speed
    :return: Final position
    """

    x = x0 + (t * u_x)  # URM
    return x


def create_frame(x, size):
    """
    Draw a frame with a ball and a occlusion element

    :param x: Ball position
    :param size: Frame size
    :return: Frame created
    """

    frame = np.zeros(size)
    cv2.circle(frame, (int(x), 128), 10, (0, 255, 0), -1)
    cv2.rectangle(frame, (100, 0), (170, size[0]), (255, 0, 0), -1)
    return frame


def get_numbers(func):
    """
    Function to get a number sequence following a function

    :param func: Function
    :return: The sequence
    """

    # Get first samples
    numbers = [func(x) for x in range(n_points)]
    # Create number to predict
    numbers.append(func(n_points+gap-1))
    return numbers


def to_file(seq, n_param, file):
    """
    Save the sequence created in a dataset

    :param seq: Sequence to save
    :param n_param: Number of parameters to save
    :param file: File where save the sequence
    """

    for i, element in enumerate(seq):
        if i == 0:
            file.write('[ ')

        if i == 4:
            file.write(element + ' ')

        elif i == 3:
            file.write(str(element) + ' ')

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

    elif toGenerate == 'n':  # Generate a sequence of numbers
        sequences = []

        func_type = conf['func_type']  # Type of function: linear, quadratic, sinusoidal

        # Create samples of the function
        limit = n_samples/2
        for i in range(n_samples):
            a = random.uniform(-limit, limit)
            b = random.uniform(-limit, limit)
            c = random.uniform(-limit, limit)

            if func_type == 'linear':
                # Set function: ax + by + c = 0
                f = lambda x: (a * x + c) / -b

            elif func_type == 'quadratic':
                while a == 0:
                    a = random.uniform(-limit, )
                # Set function: ax² + by + c = 0 with a != 0
                f = lambda x: a * (x ** 2) + b * x + c

            else:
                print('Choose a correct function to generate (linear,quadratic or sinusoidal) ')
                break

            parameters = [a, b, c, gap, func_type]  # Parameters used to create the sample

            sample = get_numbers(f)  # Get the sample

            if not noise_flag:
                noise_parameters = [None]
            else:
                mean = float(conf['noise']['mean'])
                stand_deviation = float(conf['noise']['stand_deviation'])
                noise_parameters = [mean, stand_deviation]

                noise = np.random.normal(mean, stand_deviation, len(sample))  # Different noise for each sample

                sample = list(sample + noise)

            for param in noise_parameters:  # Add noise parameters to all parameters
                parameters.append(param)

            for p in parameters[::-1]:  # Add parameters to save in file
                sample.insert(0, p)

            sequences.append(sample)  # Add the sample to the list of sequences created
            print(i)

        # Separate train, test and validation
        if conf['split']['flag']:
            fraction_test = conf['split']['fraction_test']
            fraction_validation = conf['split']['fraction_validation']

            test_set_size = int(len(sequences) * fraction_test)
            validation_set_size = int(len(sequences) * fraction_validation)

            train_set = sequences[:-(test_set_size + validation_set_size)]
            test_set = sequences[len(sequences) - (test_set_size + validation_set_size):-validation_set_size]
            validation_set = sequences[-validation_set_size:]

            print(len(train_set) + len(test_set) + len(validation_set) == len(sequences))  # Check the separation

            # Init files
            file_train = open('functions_dataset/' + func_type + '_' + str(gap) + '_' + str(noise_parameters) + '_train.txt', 'w')
            file_train.write(
                '[ a b c gap ftype noise(mean, standard deviation) ][ x=0:' + str(n_points - 1) + ' x=' +
                str(n_points + gap - 1) + ' ]\n')
            file_test = open('functions_dataset/' + func_type + '_' + str(gap) + '_' + str(noise_parameters) + '_test.txt', 'w')
            file_test.write(
                '[ a b c gap ftype noise(mean, standard deviation) ][ x=0:' + str(n_points - 1) + ' x=' +
                str(n_points + gap - 1) + ' ]\n')
            file_val = open('functions_dataset/' + func_type + '_' + str(gap) + '_' + str(noise_parameters) + '_val.txt', 'w')
            file_val.write(
                '[ a b c gap ftype noise(mean, standard deviation) ][ x=0:' + str(n_points - 1) + ' x=' +
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

        else:
            # Init file
            file = open(
                'functions_dataset/' + func_type + '_' + str(gap) + '_' + str(noise_parameters) + '_dataset.txt', 'w')
            file.write(
                '[ a b c gap ftype noise(mean, standard deviation) ][ x=0:' + str(n_points - 1) + ' x=' +
                str(n_points + gap - 1) + ' ]\n')

            # Write file
            for element in sequences:
                to_file(element, len(parameters) - 1, file)

        # Draw an example
        index = random.randrange(0, len(sequences))

        x = sequences[index][len(parameters):n_points + len(parameters)]
        y = sequences[index][-1:]
        data_utils.draw_data(plt, [x, y], gap)
        plt.title(str(index))

        plt.show()

    else:
        print('Choose a correct type to generate (frame or number) ')