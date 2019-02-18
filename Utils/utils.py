"""
utils.py: A script with some utils to process data

"""
__author__ = "Nuria Oyaga"
__date__ = "2017/11/02"


import numpy as np
import os
import argparse
import yaml
import cv2
import shutil
from matplotlib import pyplot as plt
import pandas as pd


def get_config_file():
    # Load the configuration file
    ap = argparse.ArgumentParser()
    ap.add_argument("-cf", "--config", required=True, help="Path to where the config file resides")
    args = vars(ap.parse_args())

    config_path = args['config']
    conf = yaml.load(open(config_path, 'r'))

    return conf


def check_dirs(path, replace=False):
    if os.path.exists(path) and replace==True:
        shutil.rmtree(path)
        os.makedirs(path)
    elif not os.path.exists(path):
        os.makedirs(path)


def write_header(file_path, header):
    with open(file_path, 'w+') as file:
        file.write(header)


def read_function_data(file):
    with open(file) as f:
        data = f.readlines()
    data = data[1:]

    parameters = [data[i].split('][')[0].split(' ')[1:-1] for i in range(len(data))]
    samples = [np.array(data[i].split('][')[1].split(' ')[1:-2], dtype=np.float32) for i in range(len(data))]

    return parameters, samples


def read_vector_data(path):
    parameters_path = path.replace('samples', 'parameters.txt')
    images_paths = read_images(path)

    parameters = pd.read_csv(parameters_path, sep=' ')
    images = [cv2.imread(img_path, 0) for img_path in images_paths]

    return parameters, images


def read_images(path):
    for root, dirs, files in os.walk(path):
        img_paths = [os.path.join(root, file) for file in files if file.endswith(".png")]

    return img_paths


def reshape_function_data(data):
    for i, sample in enumerate(data):
        if i % 5000 == 0:
            print(i)

        if i == 0:
            dataX = sample[0:-1]  # Get known elements
            dataY = sample[sample.size - 1]  # Get to predict element

        else:
            dataX = np.vstack([dataX, sample[0:-1]])  # Stack all the known elements (each sample)
            dataY = np.append(dataY, sample[sample.size - 1])  # Add all to predict element (each sample)

    return dataX, dataY


def reshape_vector_data(data):
    dataX = []
    dataY = []
    for i, sample in enumerate(data):
        if i % 5000 == 0:
            print(i)

        dataX.append(sample[:][0:-1])
        dataY.append(sample[:][sample.shape[0] - 1])

    dataX = np.array(dataX, dtype="float") / 255
    dataY = np.array(dataY, dtype="float") / 255

    dataY = np.expand_dims(dataY, axis=1)

    return dataX, dataY


def save_history(model_history, path):
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    n_epochs = len(model_history.epoch)
    plt.figure()
    plt.plot(np.arange(0, n_epochs), model_history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, n_epochs), model_history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(path + '_history.png')
