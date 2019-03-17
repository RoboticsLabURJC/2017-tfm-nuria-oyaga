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
import matplotlib
matplotlib.use('Agg')
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


def get_dirs(path):
    dirs = [d[0] for d in os.walk(path)][1:]

    return dirs


def read_images(path):
    img_paths = []
    for root, dirs, files in os.walk(path):
        img_paths = [os.path.join(root, file) for file in files if file.endswith(".png")]

    return img_paths


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
