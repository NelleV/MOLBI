import csv

import numpy as np


def load_data(filename='expression.csv'):
    """
    """
    csv_reader = csv.reader(open(filename, 'r'))
    data = []
    for row in csv_reader:
        data.append([float(element) for element in row])

    return np.array(data)

if __name__ == "__main__":
    data = load_data()
