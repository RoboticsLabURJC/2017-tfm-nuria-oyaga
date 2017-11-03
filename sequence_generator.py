
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
import math

toGenerate = 'n'  # Choose type to generate: frame('f') or number('n')


def get_position(x0, t, u_x):
    x = x0 + (t * u_x)  # URM
    return x


def create_frame(x, size):
    frame = np.zeros(size)
    cv2.circle(frame, (int(x), 128), 10, (0, 255, 0), -1)
    cv2.rectangle(frame, (100, 0), (170, size[0]), (255, 0, 0), -1)
    return frame


def get_numbers(func):
    numbers = [func(x) for x in range(4)]
    # Create number 10
    numbers.append(func(10))
    return numbers


def to_file(ftype, data):
    f = open('numbers.txt', 'w')
    if ftype == 'l':
        f.write('[m, n]:[x=0, x=1, x=2, x=3, x=10] \n')
    elif ftype == 'q':
        f.write('[a, b, c]:[x=0, x=1, x=2, x=3, x=10] \n')
    for seq_key in data.keys():
        f.write(seq_key + ':' + str(data[seq_key]) + '\n')


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
        sequences = {}

        func_type = 'l'  # Choose type of function: linear(l), quadratic(q) or sinusoidal(s)

        if func_type == 'l':
            rect = lambda x: m*x + n

            for m in np.arange(-5.0, 5.0):
                for n in np.arange(-5.0, 5.0):
                    num = get_numbers(rect)
                    sequences[str([m,n])] = num

        elif func_type == 'q':
            parab = lambda x: a*(x**2) + b*x + c

            for a in np.arange(-5.0, 5.0):
                if a != 0:
                    for b in np.arange(-5.0,5.0):
                        for c in np.arange(-5.0,5.0):
                            num = get_numbers(parab)
                            sequences[str([a,b,c])] = num

        elif func_type == 's':
            sen = lambda x: A * np.sin((2 * np.pi * frec * x) + math.radians(theta))
            x = (np.linspace(-10, 10, 200))
            frec = 0.1
            theta = 0
            A = 1
            plt.plot(x,sen(x))
            plt.show()

        else:
            print('Choose a correct function to generate (linear,quadratic or sinusoidal) ')

        to_file(func_type, sequences)

    else:
        print('Choose a correct type to generate (frame or number) ')
