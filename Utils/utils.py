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


def check_dirs(dir_path, replace=False):
    if os.path.exists(dir_path) and replace==True:
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    elif not os.path.exists(dir_path):
        os.makedirs(dir_path)


def write_header(file_path, header):
    with open(file_path, 'w+') as file:
        file.write(header)


def get_dirs(dir_path):
    dirs = [d[0] for d in os.walk(dir_path)][1:]

    return dirs


def get_images(dir_path):
    img_paths = []
    for root, dirs, files in os.walk(dir_path):
        img_paths = [os.path.join(root, file) for file in files if file.endswith(".png")]

    return img_paths


def get_files(dir_path):
    file_paths = []
    for root, dirs, files in os.walk(dir_path):
        file_paths = [os.path.join(root, file) for file in files if file.endswith(".txt")]

    return file_paths


def save_history(model_history, dir_path):
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    n_epochs = len(model_history.epoch)
    plt.figure()
    plt.plot(np.arange(0, n_epochs), model_history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, n_epochs), model_history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(dir_path + '_history.png')


def calculate_max(values):
    max_pos = np.argmax(values)
    return max_pos, round(float(values[np.argmax(values)]), 3)


def calculate_mean(values):
    return round(np.sum(values) / values.size, 3)


def combine_figures(figures_dir, figures):
    figures_images = [cv2.imread(figures_dir + f_path + ".png") for f_path in figures]
    if len(figures) % 2 == 0:
        horizontal = [np.concatenate((figures_images[i], figures_images[i+1]), axis=1)
                      for i in range(len(figures)) if i % 2 == 0]
    else:
        horizontal = [np.concatenate((figures_images[i], figures_images[i + 1]), axis=1)
                      for i in range(len(figures) - 1) if i % 2 == 0]
        border_size = int(figures_images[-1].shape[1] / 2)
        last = cv2.copyMakeBorder(figures_images[-1], 0, 0, border_size, border_size,
                                  cv2.BORDER_CONSTANT, None, [255, 255, 255])
        horizontal.append(last)

    vertical = np.concatenate(horizontal, axis=0)

    cv2.imwrite(figures_dir + "error_stats.png", vertical)
