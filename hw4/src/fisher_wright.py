import numpy as np

d = 100
N0 = 1e5
gen = 2000
statemat = np.zeros((d + 1, gen))
statemat[0, 0] = N0

u = 1e-7
s = 0.01
t = 2

for iteration in range(gen):
    state = statemat[:, iteration]
    Nr = ((1 + s) ) * state.T
    first_term = Nr / Nr.sum()
    second_term = u * first_term
