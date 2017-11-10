"""
function_predictor_test.py: A script to test a model of predictor

"""
__author__ = "Nuria Oyaga"
__date__ = "2017/11/02"

# Keras
from keras.models import load_model
# Others
import numpy as np
from matplotlib import pyplot as plt
import data_utils


if __name__ == '__main__':
    # Load data
    test_set = data_utils.read_data('functions_dataset/quadratic_test.txt')

    # Put the test data into the right shape
    print('Puting the test data into the right shape...')
    testX, testY = data_utils.reshape_data(test_set)

    # Load model
    model = load_model('Models/QuadraticPredictorMultilayerPerceptron200.h5')

    # Generate predictions
    prediction = model.predict(testX)

    # To make prediction of a sample
    # sample = np.array(testX[12]).reshape(1,20)
    # pred = model.predict(sample)
    # print(pred)

    #Calculate error
    error = np.array([abs(testY[i]-prediction[i]) for i in range(testY.size-1)])

    # Calculate error percentage
    pct = lambda x: (x / len(error)) * 100

    pct_error = [pct(np.sum((error > 0) & (error < 0.05))),
                pct(np.sum((error > 0.05) & (error < 0.1))),
                pct(np.sum((error > 0.1) & (error < 0.2))),
                pct(np.sum((error > 0.2) & (error < 0.3))),
                pct(np.sum((error > 0.3) & (error < 0.4))),
                pct(np.sum((error > 0.4) & (error < 0.5))),
                pct(np.sum((error > 0.5)))]

    labels = ["[0, 0.05)", "[0.05, 0.1)", "[0.1, 0.2)",
              "[0.2, 0.3)", "[0.3, 0.4)", "[0.4, 0.5)", ">0.5"]

    # Draw error percentage
    f = plt.figure()

    x = range(len(pct_error))
    y = pct_error

    plt.bar(x, y, color='red')

    plt.ylim((0, 100))
    plt.xticks(x, labels)
    plt.title("Error percentage")

    plt.show()

    # Calculate error mean
    error_mean = np.sum(error) / error.size
    print('Error mean = ', error_mean)

    # Draw max error
    max_error_index = np.argmax(error)
    limit = [testY[np.argmax(error)]-(np.max(error)*2),testY[np.argmax(error)]+(np.max(error)*2)]
    max_error = round(float(error[max_error_index]),3)

    f, ((s1, s2)) = plt.subplots(1, 2)
    x = list(range(20))

    s1.scatter(x, testX[max_error_index], 5, color='green')
    s1.scatter(29, testY[max_error_index], 5, color='green')
    s1.scatter(29, prediction[max_error_index], 5, color='red')
    s1.set_title('Sample ' + str(np.argmax(error)))

    s2.scatter(29, testY[np.argmax(error)], color='green', label='Real')
    s2.scatter(29, prediction[np.argmax(error)], color='red', label='Prediction')
    s2.plot([29,29],
                   [testY[np.argmax(error)],prediction[np.argmax(error)]],
                   '--',
                   color='black',
                   label='Distance = ' + str(max_error))

    s2.set_title('Distance between prediction and real')
    s2.set_ylim(limit[0], limit[1])
    s2.set_xlim(28,30)

    plt.legend()

    plt.show()