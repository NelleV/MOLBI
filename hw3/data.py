import numpy as np


def load_data():
    filename = "BRCA1.csv"
    data_file = open(filename, 'r')
    Y = []
    X = []
    for i, line in enumerate(data_file):
        if i == 0:
            Y = line.split(',')
        else:
            X.append([float(i) for i in line.split(',')])

    return np.array(Y), np.array(X)
