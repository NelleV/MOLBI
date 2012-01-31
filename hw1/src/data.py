from scipy import stats
import numpy as np


D = (1000, 100, 100)
P = [(1. / 3, 9. / 20, 2. / 3), (1. / 3, 11. / 20, 1. / 3)]


def generate_data(n_samples):
    """
    """
    samples = np.array([])
    for sample in range(n_samples):
        if sample < n_samples / 2:
            Y = 0
        else:
            Y = 1
        dist = np.concatenate(
                    (stats.bernoulli.rvs(P[Y][0], size=D[0]),
                     stats.bernoulli.rvs(P[Y][1], size=D[1]),
                     stats.bernoulli.rvs(P[Y][2], size=D[2]),
                     np.array([Y])))
        samples = np.concatenate((samples, dist))
    return samples.reshape((n_samples, len(samples) / n_samples))

if __name__ == "__main__":
    el = generate_data(10)
