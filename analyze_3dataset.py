"""
analyze_dataset.py: A script to analyzea dataset

"""
__author__ = "Nuria Oyaga"
__date__ = "2018/01/16"

import data_utils
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
    dir = 'functions_dataset/quadratic/'
    dataset_feat = 'quadratic_10_[None]_'
    train_parameters, train_set = data_utils.read_data(dir + dataset_feat + 'train.txt')
    test_parameters, test_set = data_utils.read_data(dir + dataset_feat + 'test.txt')
    val_parameters, val_set = data_utils.read_data(dir + dataset_feat + 'val.txt')

    # Get function parameters
    print('Getting train functions parameters...')
    train_a, train_b, train_c = get_function_parameters(train_parameters)
    print('Getting test functions parameters...')
    test_a, test_b, test_c = get_function_parameters(test_parameters)
    print('Getting validation functions parameters...')
    val_a, val_b, val_c = get_function_parameters(val_parameters)

    # Get statistics
    print('Getting functions parameters statistics...')
    # Train
    mean_a_train, std_a_train = get_statistics(train_a)
    mean_b_train, std_b_train = get_statistics(train_b)
    mean_c_train, std_c_train = get_statistics(train_c)
    # Test
    mean_a_test, std_a_test = get_statistics(test_a)
    mean_b_test, std_b_test = get_statistics(test_b)
    mean_c_test, std_c_test = get_statistics(test_c)
    # Val
    mean_a_val, std_a_val = get_statistics(val_a)
    mean_b_val, std_b_val = get_statistics(val_b)
    mean_c_val, std_c_val = get_statistics(val_c)

    # Plot
    print('Ploting...')
    plt.figure()
    x = range(5)
    for i, seq in enumerate(train_set):
        print(i)
        y = seq[:5]
        plt.plot(x, y)
    for i, seq in enumerate(test_set):
        print(i)
        y = seq[:5]
        plt.plot(x, y)
    for i, seq in enumerate(val_set):
        print(i)
        y = seq[:5]
        plt.plot(x, y)

    plt.title('Linear dataset', fontsize=18)
    plt.savefig(dir + dataset_feat + 'quadratic_dataset2.png')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(train_a, train_b, train_c, c='r', marker='o')
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('c')
    ax.set_title('Train dataset')
    plt.savefig(dir + dataset_feat + 'train_distribution.png')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(test_a, test_b, test_c, c='b', marker='^')
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('c')
    ax.set_title('Test dataset')
    plt.savefig(dir + dataset_feat + 'test_distribution.png')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(val_a, val_b, val_c, c='y', marker='v')
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('c')
    ax.set_title('Validation dataset')
    plt.savefig(dir + dataset_feat + 'val_distribution.png')

    f, ax = plt.subplots(1, 3, sharey=True)
    plt.suptitle('Mean value and standard deviation')
    x = range(1, 4)
    means_train = [mean_a_train, mean_b_train, mean_c_train]
    std_train = [std_a_train, std_b_train, std_c_train]
    ax[0].errorbar(x, means_train, std_train, linestyle='None', marker='o', capsize=5)
    for i, m in enumerate(means_train):
        str_mean = "%2.1f" % m
        ax[0].text(i + 1.4, m, str_mean, horizontalalignment='center', fontsize=6)
    ax[0].plot([0, 4], [0, 0], 'k--', linewidth=0.2)
    ax[0].set_title('Train dataset')

    means_test = [mean_a_test, mean_b_test, mean_c_test]
    std_test = [std_a_test, std_b_test, std_c_test]
    ax[1].errorbar(x, means_test, std_test, linestyle='None', marker='o', capsize=5)
    for i, m in enumerate(means_test):
        str_mean = "%2.1f" % m
        ax[1].text(i + 1.4, m, str_mean, horizontalalignment='center', fontsize=6)
    ax[1].plot([0, 4], [0, 0], 'k--', linewidth=0.2)
    ax[1].set_title('Test dataset')

    means_val = [mean_a_val, mean_b_val, mean_c_val]
    std_val = [std_a_val, std_b_val, std_c_val]
    ax[2].errorbar(x, means_val, std_val, linestyle='None', marker='o', capsize=5)
    for i, m in enumerate(means_val):
        str_mean = "%2.1f" % m
        ax[2].text(i + 1.4, m, str_mean, horizontalalignment='center', fontsize=6)
    ax[2].plot([0, 4], [0, 0], 'k--', linewidth=0.2)
    ax[2].set_title('Validation dataset')

    plt.setp(ax, xticks=x, xticklabels=['a', 'b', 'c'])
    plt.savefig(dir + dataset_feat + 'dataset_statistics.png')

    plt.show()
