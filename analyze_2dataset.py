"""
analyze_dataset.py: A script to analyze a functions dataset with 2 parameters

"""
__author__ = "Nuria Oyaga"
__date__ = "2018/01/16"

import utils
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def get_function_parameters(data_parameters):
    a = [float(param[0]) for param in data_parameters]
    c = [float(param[1]) for param in data_parameters]

    return a, c


def get_statistics(values):
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values)

    return mean, std


if __name__ == '__main__':
    # Load data
    print('Loading data...')
    dir = 'functions_dataset/linear/'
    dataset_feat = 'linear_10_[None]_'
    train_parameters, train_set = utils.read_data(dir + dataset_feat + 'train2.txt')
    test_parameters, test_set = utils.read_data(dir + dataset_feat + 'test2.txt')
    val_parameters, val_set = utils.read_data(dir + dataset_feat + 'val2.txt')

    # Get function parameters
    print('Getting train functions parameters...')
    train_a, train_c = get_function_parameters(train_parameters)
    print('Getting test functions parameters...')
    test_a, test_c = get_function_parameters(test_parameters)
    print('Getting validation functions parameters...')
    val_a, val_c = get_function_parameters(val_parameters)

    # Get statistics
    print('Getting functions parameters statistics...')
    # Train
    mean_a_train, std_a_train = get_statistics(train_a)
    mean_c_train, std_c_train = get_statistics(train_c)
    # Test
    mean_a_test, std_a_test = get_statistics(test_a)
    mean_c_test, std_c_test = get_statistics(test_c)
    # Val
    mean_a_val, std_a_val = get_statistics(val_a)
    mean_c_val, std_c_val = get_statistics(val_c)

    # Histogram
    a = train_a + test_a + val_a
    c = train_c + test_c + val_c

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.random.rand(2, 100) * 4
    hist, xedges, yedges = np.histogram2d(a, c)

    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')
    zpos = np.zeros_like(xpos)

    dx = 0.5 * np.ones_like(zpos)
    dy = dx.copy()
    dz = hist.flatten()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b')

    plt.figure()
    plt.subplot(311)
    plt.hist(a)
    plt.title('Histogram a')

    plt.subplot(313)
    plt.hist(c)
    plt.title('Histogram c')
    plt.savefig(dir + dataset_feat + 'histogram2-2.png')

    f = lambda x: (a * x + c)
    x = np.array([0, 29])

    # Plot
    print('Ploting...')
    plt.figure()
    for i, seq in enumerate(train_set):
        a = train_a[i]
        c = train_c[i]
        print(i)
        y = f(x)
        plt.plot(x, y)
    for i, seq in enumerate(test_set):
        a = test_a[i]
        c = test_c[i]
        print(i)
        y = f(x)
        plt.plot(x, y)
    for i, seq in enumerate(val_set):
        a = val_a[i]
        c = val_c[i]
        print(i)
        y = f(x)
        plt.plot(x, y)
    plt.axis('equal')
    plt.title('Linear dataset', fontsize=18)
    plt.savefig(dir + dataset_feat + 'linear_dataset2.png')

    fig = plt.figure()
    plt.scatter(train_a, train_c, c='r', marker='o')
    plt.xlabel('a')
    plt.ylabel('c')
    plt.title('Train dataset')
    plt.savefig(dir + dataset_feat + 'train_distribution2.png')

    fig = plt.figure()

    plt.scatter(test_a, test_c, c='b', marker='^')
    plt.xlabel('a')
    plt.ylabel('c')

    plt.title('Test dataset')
    plt.savefig(dir + dataset_feat + 'test_distribution2.png')

    fig = plt.figure()
    plt.scatter(val_a, val_c, c='y', marker='v')
    plt.xlabel('a')
    plt.ylabel('c')
    plt.title('Validation dataset')
    plt.savefig(dir + dataset_feat + 'val_distribution2.png')

    plt.show()
