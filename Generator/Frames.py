import numpy as np
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
        self.parameters = []
        self.sample = []

    def get_sample(self):
        positions = self.get_positions()
        for i in range(self.n_points + 1):
            self.sample.append(self.get_image(positions[i]))

    def get_positions(self):
        # Get first positions
        numbers = [self.f(x) for x in range(self.n_points)]
        # Create position to predict
        numbers.append(self.f(self.n_points + self.gap - 1))
        return numbers

    def get_image(self, pos):
        if isinstance(self.shape.color, int):
            image = np.zeros((self.h, self.w),  np.uint8)

        else:
            image = np.zeros((self.h, self.w, 3),  np.uint8)

        image = self.shape.draw(image, (pos, int(self.h/2)))

        return image

    def get_complex_image(self, object_pos):
        pass

    def save(self, image_path, filename):
        for i, image in enumerate(self.sample):
            cv2.imwrite(image_path + "_" + str(i) + '.png', image)
        with open(filename, 'a+') as file:
            for p in self.parameters:
                file.write(str(p) + ' ')
            file.write(str(self.n_points) + ' ')
            file.write(str(self.gap) + ' ')
            file.write(self.type + ' ')
            file.write(str(self.noise_parameters) + '\n')


class URM(Frames):

    def __init__(self, noise_parameters, n_points, gap, h, w, shape):
        Frames.__init__(self, "URM", noise_parameters, n_points, gap, h, w, shape)

        x0 = 0
        limit = int(self.w / (n_points + gap))
        if self.shape.type == "Circle":
            x0 = self.shape.r
            limit = int((self.w - self.shape.r) / (n_points + gap))

        u_x = random.randint(1, limit)
        self.parameters = [x0, u_x]
        self.f = lambda x: x0 + u_x * x
        self.get_sample()
