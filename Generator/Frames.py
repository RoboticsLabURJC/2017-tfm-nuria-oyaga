from Utils import utils

import numpy as np
import pandas as pd
import random
import cv2


class Frames(object):

    def __init__(self, m_type, noise_parameters, n_points, gap, h, w, shape):
        self.h = h
        self.w = w
        self.shape = shape
        self.type = m_type
        self.noise_parameters = noise_parameters
        self.n_points = n_points
        self.gap = gap
        self.f = None
        self.g = None
        self.parameters = []
        self.raw_sample = []
        self.modeled_sample = []

    def get_sample(self):
        positions_x, positions_y = self.get_positions()
        for i in range(self.n_points + 1):
            self.raw_sample.append(self.get_image(positions_x[i], positions_y[i]))
            self.modeled_sample.append([positions_x[i], positions_y[i]])

    def get_positions(self):
        while True:
            if self.type == 'URM':
                numbers_x = [self.f(x) for x in range(self.n_points)]
                numbers_x.append(self.f(self.n_points + self.gap - 1))
                if self.parameters[2] is None:
                    y0 = random.randint(1, self.h - 1)
                    self.parameters[2] = y0
                else:
                    y0 = self.parameters[2]

                numbers_y = [self.g(n_x, y0) for n_x in numbers_x]
                numbers_y.append(self.g(numbers_x[-1], y0))
            elif self.type == 'Linear':
                numbers_x = [self.f(x) for x in range(self.n_points)]
                numbers_x.append(self.f(self.n_points + self.gap - 1))
                if self.parameters[2] is None:
                    y0 = random.randint(1, self.h - 1)
                else:
                    y0 = self.parameters[2]

                m = np.round(random.uniform(-self.h/10, self.h/10), 2)

                self.parameters[-2:] = [y0, m]
                numbers_y = [int(self.g(n_x, y0, m)) for n_x in numbers_x]
                numbers_y.append(int(self.g(numbers_x[-1], y0, m)))
            else:  # self.type == 'Parabolic'
                angle = np.radians(89)
                v0 = np.round(random.uniform(10, 100), 2)
                print(v0)
                self.parameters[-2:] = [angle, v0]
                numbers_x = [int(self.f(t, v0, angle)) for t in range(self.n_points)]
                numbers_x.append(self.f(self.n_points + self.gap - 1, v0, angle))
                numbers_y = [int(self.g(t, v0, angle)) for t in range(self.n_points)]
                numbers_y.append(self.g(self.n_points + self.gap - 1, v0, angle))

            if self.is_valid(numbers_x, numbers_y):
                break

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

    def get_complex_image(self, object_pos):
        pass

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

    def __init__(self, noise_parameters, n_points, gap, h, w, shape, y0_type):
        Frames.__init__(self, "URM", noise_parameters, n_points, gap, h, w, shape)

        x0 = 0
        if y0_type == 'fix':
            y0 = int(self.h / 2)
        else:
            y0 = None

        limit = int(self.w / (n_points + gap))
        if self.shape.type == "Circle":
            x0 = self.shape.r
            limit = int((self.w - self.shape.r) / (n_points + gap))

        u_x = random.randint(1, limit)
        self.parameters = [x0, u_x, y0]
        self.f = lambda x: x0 + u_x * x
        self.g = lambda y, y0: y0
        self.get_sample()


class Linear(Frames):
    def __init__(self, noise_parameters, n_points, gap, h, w, shape, y0_type):
        Frames.__init__(self, "Linear", noise_parameters, n_points, gap, h, w, shape)

        x0 = 0
        if y0_type == 'fix':
            y0 = int(self.h / 2)
        else:
            y0 = None

        limit = int(self.w / (n_points + gap))
        if self.shape.type == "Circle":
            x0 = self.shape.r
            limit = int((self.w - self.shape.r) / (n_points + gap))
        u_x = random.randint(1, limit)

        self.parameters = [x0, u_x, y0, None]
        self.f = lambda x: x0 + u_x * x
        self.g = lambda y, y0, m: (m * y) + y0
        self.get_sample()


class Parabolic(Frames):
    def __init__(self, noise_parameters, n_points, gap, h, w, shape):
        Frames.__init__(self, "Parabolic", noise_parameters, n_points, gap, h, w, shape)

        x0 = 0
        if self.shape.type == "Circle":
            x0 = self.shape.r
        y0 = 0
        gravity = 9.8

        # x=v0·cosθ·t + x0
        # y=v0·senθ·t-gt2/2 + y0
        self.parameters = [x0, y0, gravity, None, None]
        self.f = lambda t, v0, angle: (v0 * np.cos(angle) * t) + x0
        self.g = lambda t, v0, angle: (v0 * np.sin(angle) * t) - (0.5 * gravity * (t**2)) + y0
        self.get_sample()


