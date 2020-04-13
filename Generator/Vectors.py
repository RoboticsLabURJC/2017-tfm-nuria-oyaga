"""

TFM - Vectors.py - Description

"""
__author__ = "Nuria Oyaga"
__date__ = "31/05/2018"


import numpy as np
import random
import cv2


class Vector(object):

    def __init__(self, m_type, noise_parameters, n_points, gap, v_len):
        self.v_len = v_len
        self.type = m_type
        self.noise_parameters = noise_parameters
        self.n_points = n_points
        self.gap = gap
        self.f = None
        self.parameters = []
        self.v_size = (self.n_points + 1, v_len)
        self.sample = np.zeros(self.v_size, np.uint8)

    def get_sample(self):
        positions = self.get_positions()
        for i in range(self.v_size[0]):
            self.sample[i][positions[i]] = 255

    def get_positions(self):
        # Get first positions
        numbers = [self.f(x) for x in range(self.n_points)]
        # Create position to predict
        numbers.append(self.f(self.n_points + self.gap - 1))
        return numbers

    def save(self, image_path, filename):
        cv2.imwrite(image_path, self.sample)
        with open(filename, 'a+') as file:
            for p in self.parameters:
                file.write(str(p) + ' ')
            file.write(str(self.n_points) + ' ')
            file.write(str(self.gap) + ' ')
            file.write(self.type + ' ')
            file.write(str(self.noise_parameters) + '\n')


class URM(Vector):

    def __init__(self, noise_parameters, n_points, gap, v_len):
        Vector.__init__(self, "URM", noise_parameters, n_points, gap, v_len)

        u_x = random.randint(1, int(self.v_size[1] / (n_points + gap)))
        x0 = 0
        self.parameters = [x0, u_x]
        self.f = lambda t: x0 + u_x * t
        self.get_sample()
