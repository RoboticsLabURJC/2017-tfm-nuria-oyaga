"""
analyze_dataset.py: A script to analyze a functions dataset with 3 parameters

"""
__author__ = "Nuria Oyaga"
__date__ = "2018/01/16"

import utils
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def get_function_parameters(data_parameters):
    a = [float(param[0]) for param in data_parameters]
    b = [float(param[1]) for param in data_parameters]
    c = [float(param[2]) for param in data_parameters]

    return a, b, c


def get_statistics(values):
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values)

    return mean, std


if __name__ == '__main__':
    # Load data
    print('Loading data...')
    func_type = 'sinusoidal'
    dir = 'functions_dataset/' + func_type + '/' + func_type + '_10_[None]_'
    train_parameters, train_set = utils.read_data(dir + 'train.txt')
    test_parameters, test_set = utils.read_data(dir + 'test.txt')
    val_parameters, val_set = utils.read_data(dir + 'val.txt')

    # Get function parameters
    print('Getting train functions parameters...')
    train_a, train_b, train_c = get_function_parameters(train_parameters)
    print('Getting test functions parameters...')
    test_a, test_b, test_c = get_function_parameters(test_parameters)
    print('Getting validation functions parameters...')
    val_a, val_b, val_c = get_function_parameters(val_parameters)

    # Histogram
    a = train_a + test_a + val_a
    b = train_b + test_b + val_b
    c = train_c + test_c + val_c

    x = np.array([0, 29])

    if train_parameters[0][4] == 'linear':
        f = lambda x: (a * x + c)/-b

        labela = 'a'
        labelb = 'b'
        labelc = 'c'

        a_prim = [a[i] / -b[i] for i in range(len(a))]
        c_prim = [c[i] / -b[i] for i in range(len(c))]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        hist, xedges, yedges = np.histogram2d(a_prim, c_prim)

        xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
        xpos = xpos.flatten('F')
        ypos = ypos.flatten('F')
        zpos = np.zeros_like(xpos)

        dx = 0.5 * np.ones_like(zpos)
        dy = dx.copy()
        dz = hist.flatten()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b')
        ax.set_xlabel("a'")
        ax.set_ylabel("c'")
        plt.savefig(dir + 'histogram3d.png')

        plt.figure()
        ax = plt.subplot(211)
        plt.hist(a_prim, 50)
        plt.title("Histogram a'", fontsize=10)
        plt.subplot(212, sharex=ax)
        plt.hist(c_prim, 50)
        plt.title("Histogram c'", fontsize=10)
        plt.subplots_adjust(hspace=0.5)
        plt.savefig(dir + 'histogram.png')

    elif train_parameters[0][4] == 'quadratic':
        f = lambda x: a * (x ** 2) + b * x + c

        labela = 'a'
        labelb = 'b'
        labelc = 'c'

    else:
        fs = 100
        f = lambda x: a * np.sin(2 * np.pi * b * (x / fs) + np.deg2rad(c))

        labela = 'A'
        labelb = 'f'
        labelc = 'theta'

    # Plot
    print('Ploting...')
    '''plt.figure()
    for i, seq in enumerate(train_set):
        a = train_a[i]
        b = train_b[i]
        c = train_c[i]
        print(i)
        y = f(x)
        plt.plot(x, y)
    for i, seq in enumerate(test_set):
        a = test_a[i]
        b = test_b[i]
        c = test_c[i]
        print(i)
        y = f(x)
        plt.plot(x, y)
    for i, seq in enumerate(val_set):
        a = val_a[i]
        b = val_b[i]
        c = val_c[i]
        print(i)
        y = f(x)
        plt.plot(x, y)
    plt.axis('equal')
    plt.title(train_parameters[0][4].upper() + ' dataset', fontsize=18)
    plt.savefig(dir + train_parameters[0][4] + '_dataset.png')'''

    plt.figure()
    ax = plt.subplot(311)
    plt.hist(a)
    plt.title('Histogram ' + labela, fontsize=10)
    plt.subplot(312, sharex=ax)
    plt.hist(b)
    plt.title('Histogram ' + labelb, fontsize=10)
    plt.subplot(313, sharex=ax)
    plt.hist(c)
    plt.title('Histogram ' + labelc, fontsize=10)
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(dir + 'histogram.png')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(train_a, train_b, train_c, c='r', marker='o')
    ax.set_xlabel(labela)
    ax.set_ylabel(labelb)
    ax.set_zlabel(labelc)
    ax.set_title('Train dataset')
    plt.savefig(dir + 'train_distribution.png')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(test_a, test_b, test_c, c='b', marker='^')
    ax.set_xlabel(labela)
    ax.set_ylabel(labelb)
    ax.set_zlabel(labelc)
    ax.set_title('Test dataset')
    plt.savefig(dir + 'test_distribution.png')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(val_a, val_b, val_c, c='y', marker='v')
    ax.set_xlabel(labela)
    ax.set_ylabel(labelb)
    ax.set_zlabel(labelc)
    ax.set_title('Validation dataset')
    plt.savefig(dir + 'val_distribution.png')
