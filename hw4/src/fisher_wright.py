import numpy as np
from scipy import misc
from matplotlib import pyplot as plt

d = 100
N0 = 1e5
Nt = N0
gen = 1500
statemat = np.zeros((d + 1, gen))
statemat[0, 0] = N0

u = 1e-7
s = 0.01
t = 2
alpha = 1.7 * 1e-4

for iteration in range(gen - 1):
    if iteration % 100 == 0:
        print iteration
    state = statemat[:, iteration]
    Nr = ((1 + s) ** np.arange(d + 1)) * state
    first_term = Nr / Nr.sum()
    theta = np.zeros((d + 1))
    for k in range(d + 1):
        el = misc.factorial(d - np.arange(k + 1)) / \
             (misc.factorial(k - np.arange(k + 1)) * misc.factorial(d - k))
        el *= u ** (k - np.arange(k + 1)) * (1 - u) ** (d - k)
        el *= first_term[:k + 1]
        theta[k] = el.sum()
    theta /= theta.sum()
    Nt = np.ceil((1 + alpha) * Nt)
    statemat[:, iteration + 1] = np.random.multinomial(Nt, theta)

fig = plt.figure(0)
for i in range(d):
    plt.plot(statemat[i])
