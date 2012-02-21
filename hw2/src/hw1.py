import numpy as np

from data import load_data

data = load_data()

# Let's discretize the expression levels
thres = data.flatten()
thres.sort()
a = thres[thres.size / 3]
b = thres[thres.size * 2. / 3]

data[data <= a] = 0
data[(data > a) & (data < b)] = 1
data[data >= b] = 2

###############################################################################
# Calculate the mutual information
n_samples = data.shape[1]
n0 = (data == 0).sum(axis=1).astype(float) / n_samples
n1 = (data == 1).sum(axis=1).astype(float) / n_samples
n2 = (data == 2).sum(axis=1).astype(float) / n_samples

# Calculate each term of the sum, and make sure none of them are NaNs (not a
# number). Replace NaNs with 0

lp0 = - n0 * np.log(n0)
lp0[np.isnan(lp0)] = 0

lp1 = - n1 * np.log(n1)
lp1[np.isnan(lp1)] = 0

lp2 = - n2 * np.log(n2)
lp2[np.isnan(lp2)] = 0

# Now copy these values 153 and reshape them, in order to obtain a 153 * 153
# matrix
lp0.shape = (1, 153)
lp0 = lp0.repeat(153, axis=0)

lp1.shape = (1, 153)
lp1 = lp1.repeat(153, axis=0)

lp2.shape = (1, 153)
lp2 = lp2.repeat(153, axis=0)

lp = np.zeros((9, 153, 153))

for i in range(153):
    m0 = (data[i] == 0).sum()
    m1 = (data[i] == 1).sum()
    m2 = (data[i] == 2).sum()

    lp[0, i] = - n0 * (data[:, data[i] == 0] == 0).sum(axis=1).astype(float) / m0
    lp[1, i] = - n1 * (data[:, data[i] == 0] == 1).sum(axis=1).astype(float) / m0
    lp[2, i] = - n2 * (data[:, data[i] == 0] == 2).sum(axis=1).astype(float) / m0

    lp[3, i] = - n0 * (data[:, data[i] == 1] == 0).sum(axis=1).astype(float) / m1
    lp[4, i] = - n1 * (data[:, data[i] == 1] == 1).sum(axis=1).astype(float) / m1
    lp[5, i] = - n2 * (data[:, data[i] == 1] == 2).sum(axis=1).astype(float) / m1

    lp[6, i] = - n0 * (data[:, data[i] == 2] == 0).sum(axis=1).astype(float) / m2
    lp[7, i] = - n1 * (data[:, data[i] == 2] == 1).sum(axis=1).astype(float) / m2
    lp[8, i] = - n2 * (data[:, data[i] == 2] == 2).sum(axis=1).astype(float) / m2

lp = - lp * np.log(lp)
lp[np.isnan(lp)] = 0

I = lp0 + lp1 - lp.sum(axis=0)
