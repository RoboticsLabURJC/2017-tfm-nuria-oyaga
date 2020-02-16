from Utils import func_utils
from matplotlib import pyplot as plt
import random

if __name__ == '__main__':
    parameters, samples = func_utils.read_function_data("/home/nuria/Documents/TFM/Functions_dataset/linear/linear_10_[None]_train.txt")
    to_draw = random.randint(0, len(samples)-1)
    sample_to_draw = samples[to_draw]
    gap = int(parameters[0][3])
    f = plt.figure()
    ax = plt.axes()
    func_utils.draw_function(ax, [sample_to_draw[:-1], sample_to_draw[-1]], None, gap)
    plt.show()
