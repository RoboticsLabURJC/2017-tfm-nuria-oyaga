import numpy as np
import cv2


class Shape(object):
    def __init__(self, shape_type, color):
        self.type = shape_type
        self.color = color


class Point(Shape):
    def __init__(self, color):
        Shape.__init__(self, "Point", color)

    def draw(self, image, pos):
        if len(image.shape) == 2:
            image[pos[1]][pos[0]] = self.color
        else:
            for i in range(image.shape[-1]):
                image[pos[1]][pos[0]][i] = self.color[i]
        return image


class Circle(Shape):

    def __init__(self, color, r):
        Shape.__init__(self, "Circle", color)
        self.r = r

    def draw(self, image, pos):
        image = cv2.circle(image, pos, self.r, self.color, -1)

        return image
