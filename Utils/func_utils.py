from matplotlib import pyplot as plt


def draw_function(fig, real_data, pred_data, gap):
    # For know elements
    last_know = len(real_data[0])
    x = list(range(last_know))

    fig.scatter(x, real_data[0], 5, color='green')
    # For to predict element
    fig.scatter(last_know - 1 + gap, real_data[1], 5, color='green')
    fig.scatter(last_know - 1 + gap, pred_data, 5, color='red')



