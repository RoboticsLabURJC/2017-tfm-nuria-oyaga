import numpy as np


def read_function_data(file):
    with open(file) as f:
        data = f.readlines()
    data = data[1:]

    parameters = [data[i].split('][')[0].split(' ')[1:-1] for i in range(len(data))]
    samples = [np.array(data[i].split('][')[1].split(' ')[1:-2], dtype=np.float32) for i in range(len(data))]

    return parameters, samples


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


def draw_function(fig, real_data, pred_data, gap):
    # For know elements
    last_know = len(real_data[0])
    x = list(range(last_know))

    fig.scatter(x, real_data[0], 5, color='green')
    # For to predict element
    fig.scatter(last_know - 1 + gap, real_data[1], 5, color='green')
    if pred_data is not None:
        fig.scatter(last_know - 1 + gap, pred_data, 5, color='red')



