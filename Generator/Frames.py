from Utils import utils

import numpy as np
import pandas as pd
import random
import cv2
import math


class Frames(object):

    def __init__(self, m_type, noise_parameters, n_points, gap, h, w, shape, dof_type):
        self.h = h
        self.w = w
        self.shape = shape
        self.type = m_type
        self.noise_parameters = noise_parameters
        self.n_points = n_points
        self.gap = gap
        # x = x0 + ux*t
        self.f = lambda t, x0, u_x: x0 + u_x * t
        self.g = None
        self.dof_type = dof_type
        self.parameters = []
        self.raw_sample = []
        self.modeled_sample = []

    def get_sample(self):
        x0 = 0
        limit = int(self.w / (self.n_points + self.gap))
        if self.shape.type == "Circle":
            x0 = self.shape.r
            limit = int((self.w - self.shape.r) / (self.n_points + self.gap))

        u_x = random.randint(1, limit)

        if self.dof_type == 'fix':
            y0 = int(self.h / 2)
        else:
            y0 = None
        self.parameters = [x0, u_x, y0]

        positions_x, positions_y = self.get_positions(x0, u_x, y0)

        for i in range(self.n_points + 1):
            self.raw_sample.append(self.get_image(positions_x[i], positions_y[i]))
            self.modeled_sample.append([positions_x[i], positions_y[i]])

    def get_positions(self, x0, u_x, y0):
        init_y0 = y0
        while True:
            if y0 is None:
                y0 = random.randint(1, self.h - 1)
                self.parameters[-1] = y0

            if self.type == 'URM':
                numbers_x = [self.f(x, x0, u_x) for x in range(self.n_points)]
                numbers_x.append(self.f(self.n_points + self.gap - 1, x0, u_x))
                numbers_y = [self.g(n_x, y0) for n_x in numbers_x]

            elif self.type == 'Linear':
                numbers_x = [self.f(x, x0, u_x) for x in range(self.n_points)]
                numbers_x.append(self.f(self.n_points + self.gap - 1, x0, u_x))
                m = np.round(random.uniform(-self.h/10, self.h/10), 2)
                self.parameters.append(m)
                numbers_y = [int(self.g(n_x, y0, m)) for n_x in numbers_x]

            elif self.type == 'Parabolic':
                numbers_x = [self.f(x, x0, u_x) for x in range(self.n_points)]
                numbers_x.append(self.f(self.n_points + self.gap - 1, x0, u_x))
                a = round(random.uniform(-0.75, 0.75), 3)
                if self.dof_type in ['fix', 'var']:
                    b = 1.5
                else:
                    b = round(random.uniform(-2, 2), 3)
                self.parameters += [a, b]
                numbers_y = [int(self.g(n_x, a, b, y0)) for n_x in numbers_x]

            else:  # self.type == 'Sinusoidal'
                numbers_x = [self.f(x, x0, u_x) for x in range(self.n_points)]
                numbers_x.append(self.f(self.n_points + self.gap - 1, x0, u_x))
                if self.dof_type in ['fix', 'var']:
                    a = 25
                else:
                    a = random.randint(15, 40)
                if self.dof_type == "var_2":
                    b = round(random.uniform(0, 2 * math.pi), 3)
                else:
                    b = 0
                f = round(random.uniform(0.5, 5), 2)
                self.parameters += [a, b]
                numbers_y = [int(self.g(n_x, a, b, y0, f)) for n_x in numbers_x]

            if self.is_valid(numbers_x, numbers_y):
                break
            else:
                self.parameters = self.parameters[0:2]
                y0 = init_y0

        return numbers_x, numbers_y

    def is_valid(self, values_x, values_y):
        max_val_x = np.max(values_x)
        min_val_x = np.min(values_x)
        max_val_y = np.max(values_y)
        min_val_y = np.min(values_y)

        return (max_val_x < self.w and min_val_x >= 0) and (max_val_y < self.h and min_val_y >= 0)

    def get_image(self, posx, posy):
        if isinstance(self.shape.color, int):
            image = np.zeros((self.h, self.w),  np.uint8)

        else:
            image = np.zeros((self.h, self.w, 3),  np.uint8)

        image = self.shape.draw(image, (posx, posy))

        return image

    def save(self, image_path, filename, sample_file_path):
        sample_df = pd.DataFrame(columns=['x', 'y'])

        for i, image in enumerate(self.raw_sample):
            if i == 0:
                utils.check_dirs(image_path, True)
            cv2.imwrite(image_path + "/" + str(i) + '.png', image)
            sample_df.loc[i] = self.modeled_sample[i]

        sample_df.to_csv(sample_file_path, index=False)

        with open(filename, 'a+') as file:
            for p in self.parameters:
                file.write(str(p) + ' ')
            file.write(str(self.n_points) + ' ')
            file.write(str(self.gap) + ' ')
            file.write(self.type + ' ')
            file.write(str(self.noise_parameters) + '\n')


class URM(Frames):

    def __init__(self, noise_parameters, n_points, gap, h, w, shape, dof_type):
        Frames.__init__(self, "URM", noise_parameters, n_points, gap, h, w, shape, dof_type)

        # y = y0
        self.g = lambda x, y0: y0
        self.get_sample()


class Linear(Frames):
    def __init__(self, noise_parameters, n_points, gap, h, w, shape, dof_type):
        Frames.__init__(self, "Linear", noise_parameters, n_points, gap, h, w, shape, dof_type)
        # y = m*x + y0
        self.g = lambda x, y0, m: (m * x) + y0
        self.get_sample()


class Parabolic(Frames):
    def __init__(self, noise_parameters, n_points, gap, h, w, shape, dof_type):
        Frames.__init__(self, "Parabolic", noise_parameters, n_points, gap, h, w, shape, dof_type)

        # y = a*t^2 + b*t + c
        self.g = lambda x, a, b, c: a * (x ** 2) + b * x + c

        self.get_sample()


class Sinusoidal(Frames):
    def __init__(self, noise_parameters, n_points, gap, h, w, shape, dof_type):
        Frames.__init__(self, "Sinusoidal", noise_parameters, n_points, gap, h, w, shape, dof_type)

        # y = a*sin(fx + b) + c
        self.g = lambda x, a, b, c, f: a * math.sin(2 * math.pi * f * math.radians(x) + b) + c

        self.get_sample()


