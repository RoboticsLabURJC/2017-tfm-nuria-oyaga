
"""
sequence_generator.py: Generador de secuencias de frames o numeros

"""
__author__ = "Nuria Oyaga"
__date__ = "2017/10/06"

import cv2
import numpy as np
import os
import shutil

toGenerate = 'n'  # Choose type to generate: frame('f') or number('n')


def get_position(x0, t, u_x):
    x = x0 + (t * u_x)  # URM
    return x


def create_frame(x, size):
    frame = np.zeros(size)
    cv2.circle(frame, (int(x), 128), 10, (0, 255, 0), -1)
    cv2.rectangle(frame, (100, 0), (170, size[0]), (255, 0, 0), -1)
    return frame


def get_numbers(x0, u_x):
    numbers = ['%.1f' % get_position(x0, t, u_x) for t in range(4)]
    # Create number 10
    numbers.append('%.1f' % get_position(x0, 10, u_x))
    return numbers


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
        # numbers =[getNumbers(0,u_x) for u_x in np.arange(0.1, 5.0, 0.1)]
        f = open('numbers.txt','w')
        for u_x in np.arange(0.1, 5.0, 0.1):
            f.write(str(get_numbers(0,u_x))+ '\n')

    else:
        print('Choose a correct type to generate (frame or number) ')
