"""

TFM - Function.py - Description

"""
__author__ = "Nuria Oyaga"
__date__ = "23/04/2018"


import numpy as np
import random


class Function(object):

    def __init__(self, f_type, noise_parameters, n_points, gap):
        self.type = f_type
        self.know_values = np.array([])
        self.to_predict_value = 0
        self.func_parameters = np.array([])
        self.noise_parameters = noise_parameters
        self.n_points = n_points
        self.gap = gap
        self.f = None

    def get_values(self):
        # Get first samples
        self.know_values = np.array([self.f(x) for x in range(self.n_points)])
        # Create number to predict
        self.to_predict_value = self.f(self.n_points + self.gap - 1)
        if len(self.noise_parameters) > 1:
            self.add_noise()

    def add_noise(self):
        noise = np.random.normal(self.noise_parameters[0], self.noise_parameters[1], len(self.know_values) + 1)
        self.know_values = self.know_values + noise[:-1]
        self.to_predict_value = self.to_predict_value + noise[-1]

    def write(self, filename):
        with open(filename, 'a+') as file:
            # Write parameters
            for i, f_param in enumerate(self.func_parameters):
                if i == 0:
                    file.write('[ ')

                file.write("{0:.4f}".format(f_param) + ' ')

            file.write(str(self.gap) + ' ')
            file.write(self.type + ' ')

            for i, noise_param in enumerate(self.noise_parameters):
                file.write(str(noise_param) + ' ')
                if i == len(self.noise_parameters) - 1:
                    file.write('][ ')

            # Write sample
            sample = self.get_sample()
            for i, element in enumerate(sample):
                file.write("{0:.4f}".format(element) + ' ')
                if i == len(sample) - 1:
                    file.write('] \n')

    def draw(self, fig):
        # For know elements
        last_know = len(self.know_values)
        x = list(range(last_know))
        fig.scatter(x, self.know_values, 5, color='green')

        # For to predict element
        fig.scatter(last_know - 1 + self.gap, self.to_predict_value, 5, color='green')

    def get_sample(self):
        return np.append(self.know_values, self.to_predict_value)


class Linear(Function):

    def __init__(self, a_limit, b_limit, c_limit, y_limit, noise_parameters, n_points, gap):
        Function.__init__(self, "LINEAR", noise_parameters, n_points, gap)
        self.f = lambda x: (a * x + c) / -b
        while 1:
            a = random.uniform(-a_limit, a_limit)
            b = random.uniform(-b_limit, b_limit)
            c = random.uniform(-c_limit, c_limit)
            self.get_values()
            if self.is_valid(y_limit):
                self.func_parameters = np.array([a, b, c])  # Parameters used to create the sample
                break

    def is_valid(self, y_limit):
        sample = self.get_sample()
        return abs(sample[0]) < y_limit and abs(sample[-1]) < y_limit


class Quadratic(Function):

    def __init__(self, a_limit, b_limit, c_limit, y_limit, noise_parameters, n_points, gap):
        Function.__init__(self, "QUADRATIC", noise_parameters, n_points, gap)
        self.f = lambda x: a * (x ** 2) + b * x + c
        while 1:
            a = random.uniform(-a_limit, a_limit)
            b = random.uniform(-b_limit, b_limit)
            c = random.uniform(-c_limit, c_limit)
            self.get_values()
            if self.is_valid(a, y_limit):
                self.func_parameters = np.array([a, b, c])  # Parameters used to create the sample
                break

    def is_valid(self, a, y_limit):
        sample = self.get_sample()
        return abs(a) > 1 and abs(max(sample)) < y_limit and abs(min(sample)) < y_limit


class Sinusoidal(Function):

    def __init__(self, a_limit, f_limit, theta_limit, noise_parameters, n_points, gap):
        Function.__init__(self, "SINUSOIDAL", noise_parameters, n_points, gap)
        fs = n_points
        self.f = lambda x: a * np.sin(2 * np.pi * freq * (x / fs) + np.deg2rad(theta))
        while 1:
            a = random.uniform(-a_limit, a_limit)
            if abs(a) > 0.1:
                break
        freq = random.uniform(1, f_limit)  # freq
        theta = random.uniform(0, theta_limit)  # theta
        self.get_values()
        self.func_parameters = np.array([a, freq, theta, fs])  # Parameters used to create the sample


class Poly3(Function):

    def __init__(self, a_limit, b_limit, c_limit, d_limit, y_limit, noise_parameters, n_points, gap):
        Function.__init__(self, "POLY3", noise_parameters, n_points, gap)
        self.f = lambda x:  a * (x ** 3) + b * (x ** 2) + c * x + d
        while 1:
            a = random.uniform(-a_limit, a_limit)
            b = random.uniform(-b_limit, b_limit)
            c = random.uniform(-c_limit, c_limit)
            d = random.uniform(-d_limit, d_limit)
            self.get_values()
            if self.is_valid(y_limit):
                self.func_parameters = np.array([a, b, c, d])  # Parameters used to create the sample
                break

    def is_valid(self, y_limit):
        sample = self.get_sample()
        return abs(max(sample)) < y_limit and abs(min(sample)) < y_limit


class Poly4(Function):

    def __init__(self, a_limit, b_limit, c_limit, d_limit, e_limit, y_limit, noise_parameters, n_points, gap):
        Function.__init__(self, "POLY4", noise_parameters, n_points, gap)
        self.f = lambda x: a * (x ** 4) + b * (x ** 3) + c * (x ** 2) + d * x + e
        while 1:
            a = random.uniform(-a_limit, a_limit)
            b = random.uniform(-b_limit, b_limit)
            c = random.uniform(-c_limit, c_limit)
            d = random.uniform(-d_limit, d_limit)
            e = random.uniform(-e_limit, e_limit)
            self.get_values()
            if self.is_valid(y_limit):
                self.func_parameters = np.array([a, b, c])  # Parameters used to create the sample
                break

    def is_valid(self, y_limit):
        sample = self.get_sample()
        return abs(max(sample)) < y_limit and abs(min(sample)) < y_limit
