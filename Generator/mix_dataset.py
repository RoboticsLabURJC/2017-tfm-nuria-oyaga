import os
import numpy as np
import random
from shutil import copyfile, copytree
import sys

sys.path.insert(0, '/home/docker/2017-tfm-nuria-oyaga')

from Utils import utils

if __name__ == '__main__':
    # General parameters
    base_path = "/home/docker/data_far_50/Frames_dataset"
    to_mix = ["linear_point_255_var_100000_80_120",
              "parabolic_point_255_var_1_100000_80_120",
              "sinusoidal_point_255_var_2_100000_80_120"]

    samples_type_proportion = np.array([0.33, 0.33, 0.34])
    n_samples = 100000
    save_dir = os.path.join(base_path, "mix_lin_par_sin_255_var_100000_80_120")
    print("Creating dir")
    utils.check_dirs(save_dir, True)
    
    # Train parameters
    train_proportion = 0.8
    train_interval = [0, int(n_samples*train_proportion)]
    train_samples = samples_type_proportion * n_samples * train_proportion
    train_dir = os.path.join(save_dir, "mix_train")
    print("Creating train dir")
    utils.check_dirs(train_dir)
    utils.check_dirs(os.path.join(train_dir, "raw_samples"))
    utils.check_dirs(os.path.join(train_dir, "modeled_samples"))
    
    # Test parameters
    test_proportion = 0.1
    test_interval = [train_interval[1], int(train_interval[1] + n_samples * test_proportion)]
    test_samples = samples_type_proportion * n_samples * test_proportion
    test_dir = os.path.join(save_dir, "mix_test")
    print("Creating test dir")
    utils.check_dirs(test_dir)
    utils.check_dirs(os.path.join(test_dir, "raw_samples"))
    utils.check_dirs(os.path.join(test_dir, "modeled_samples"))
    
    # Val parameters
    val_proportion = 0.1
    val_interval = [test_interval[1], int(test_interval[1] + n_samples * val_proportion)]
    val_samples = samples_type_proportion * n_samples * val_proportion
    val_dir = os.path.join(save_dir, "mix_val")
    print("Creating val dir")
    utils.check_dirs(val_dir)
    utils.check_dirs(os.path.join(val_dir, "raw_samples"))
    utils.check_dirs(os.path.join(val_dir, "modeled_samples"))
    num = 0
    num_train = 0
    num_test = 0
    num_val = 0
    for i, mix_elem in enumerate(to_mix):
        for action in ["train", "test", "val"]:
            sub_elem = [k for k in os.listdir(os.path.join(base_path, mix_elem)) if action in k][0]
            if "train" in sub_elem:
                n_sub_samples = int(train_samples[i])
                interval = train_interval
                sub_save_dir = train_dir
            elif "test" in sub_elem:
                n_sub_samples = int(test_samples[i])
                interval = test_interval
                sub_save_dir = test_dir
            else:  # "val" in sub_elem:
                n_sub_samples = int(val_samples[i])
                interval = val_interval
                sub_save_dir = val_dir

            # Get random samples
            samples = random.sample(range(interval[0], interval[1]), n_sub_samples)

            # Original samples paths
            original_modeled_dir = os.path.join(base_path, mix_elem, sub_elem, "modeled_samples")
            original_raw_dir = os.path.join(base_path, mix_elem, sub_elem, "raw_samples")

            # Mix samples paths
            save_modeled_dir = os.path.join(sub_save_dir, "modeled_samples")
            save_raw_dir = os.path.join(sub_save_dir, "raw_samples")

            print(n_sub_samples)
                  
            for n, sample in enumerate(samples):
                num += 1
                if action == "train":
                    save_num = num_train
                    num_train += 1
                elif action == "test":
                    save_num = test_interval[0] + num_test
                    num_test += 1
                else:
                    save_num = val_interval[0] + num_val
                    num_val += 1

                if n % 500 == 0 or n == len(samples)-1:
                    print(action, sub_elem, n, num, num_train, num_test, num_val, save_num)

                original_copy_file = "sample{}".format(sample)
                save_copy_file = "sample{}".format(save_num)
                copyfile(os.path.join(original_modeled_dir, original_copy_file + ".txt"),
                         os.path.join(save_modeled_dir, save_copy_file + ".txt"))
                copytree(os.path.join(original_raw_dir, original_copy_file),
                         os.path.join(save_raw_dir, save_copy_file))
