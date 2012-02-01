import numpy as np
from matplotlib import pyplot as plt

from fisher import pvalue

from sklearn.externals.joblib import Memory

from data import generate_data
from fet import compute_values


def mutual_information(X, Y):
    n_samples = len(X)

    px = X.sum(axis=0) / n_samples
    pxy0 = X[:n_samples / 2].sum(axis=0) / (n_samples / 2)
    pxy1 = X[n_samples / 2:].sum(axis=0) / (n_samples / 2)
    # We now have to calculate each term of the sum, and make sure none of the
    # terms are NaN
    lp0 = - px * np.log(px)
    lp0[np.isnan(lp0)] = 0

    lp1 = - (1 - px) * np.log(1 - px)
    lp1[np.isnan(lp1)] = 0

    lp00 = - pxy0 * np.log(pxy0)
    lp00[np.isnan(lp00)] = 0

    lp10 = - (1 - pxy0) * np.log(1 - pxy0)
    lp10[np.isnan(lp10)] = 0

    lp01 = - pxy1 * np.log(pxy1)
    lp01[np.isnan(lp01)] = 0

    lp11 = - (1 - pxy1) * np.log(1 - pxy1)
    lp11[np.isnan(lp11)] = 0

    I = lp0 + lp1 - 1. / 2 * (lp00 + lp01 + lp10 + lp11)
    return I


def n_mutual_information(n, n_samples):
    nI = np.zeros((n, 1200))
    for i in range(n):
        data = generate_data(n_samples=n_samples)
        X, Y = data[:, :-1], data[:, -1]
        nI[i, :] = mutual_information(X, Y)

    return nI


def roc_curves(n_samples, p_value=False):
    """
    """
    data = generate_data(n_samples=n_samples)
    X, Y = data[:, :-1], data[:, -1]
    if p_value:
        I = np.abs(0.5 - compute_values(X))
    else:
        I = mutual_information(X, Y)
    tau_min, tau_max = I.min(), I.max()
    if not p_value and tau_min < 0:
        tau_min = 0
        print "warning - tau_min < 0"

    step = 500
    erange = [tau_min + i * (tau_max - tau_min) * 1. / step
              for i
              in range(step)]

    roc_x = []
    roc_y = []
    for tau in erange:
        pos = I > tau
        fp = pos[:1000].sum()
        fn = (1 - pos[1000:]).sum()
        tp = pos[1000:].sum()
        tn = (1 - pos[:1000]).sum()
        roc_x.append(float(tp) / (200))
        roc_y.append(1 - float(tn) / (1000))
    return np.array(roc_x), np.array(roc_y), np.array(erange)


def compute_ratios(tau, p_value=False):
    data = generate_data(n_samples=1000)
    X, Y = data[:, :-1], data[:, -1]
    if p_value:
        I = np.abs(0.5 - compute_values(X))
    else:
        I = mutual_information(X, Y)
    pos = I > tau
    fp = pos[:1000].sum()
    fn = (1 - pos[1000:]).sum()
    tp = pos[1000:].sum()
    tn = (1 - pos[:1000]).sum()

    return fp, fn, tp, tn


if __name__ == "__main__":
    fig = plt.figure(1)
    mem = Memory(cachedir='.')
    for i, n_samples in enumerate([10, 100]):

        nI = mem.cache(n_mutual_information)(100, n_samples)
        mean, var = nI.mean(axis=0), nI.var(axis=0)
        ax = fig.add_subplot(3, 2, i * 2 + 1)
        ax.bar(range(1200), mean)
        ax = fig.add_subplot(3, 2, i * 2 + 2)
        ax.bar(range(1200), var)

    roc_x, roc_y, erange = mem.cache(roc_curves)(10)
    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    plot1 = ax.plot(roc_x, roc_y)
    dis = np.sqrt((roc_x - roc_y) ** 2)
    tau_opt = erange[dis.argmax()]
    fp, fn, tp, tn = compute_ratios(tau_opt, p_value=True)
    print "n=10, with the IM, found TP %d, FP %d, TN %d, FN %d" % \
            (tp, fp, tn, fn)
    print "tau %f" % tau_opt


    roc_x, roc_y, erange = mem.cache(roc_curves)(100)
    plot2 = ax.plot(roc_x, roc_y)
    dis = np.sqrt((roc_x - roc_y) ** 2)
    tau_opt = erange[dis.argmax()]
    fp, fn, tp, tn = compute_ratios(tau_opt, p_value=True)
    print "n=100, with the IM, found TP %d, FP %d, TN %d, FN %d" % \
            (tp, fp, tn, fn)
    print "tau %f" % tau_opt

    roc_x1, roc_y1, erange1 = mem.cache(roc_curves)(1000)
    plot3 = ax.plot(roc_x1, roc_y1)
    fig.legend((plot1[0], plot2[0], plot3[0]),
                ('n = 10', 'n = 100', 'n = 1000'))
    dis = np.sqrt((roc_x1 - roc_y1) ** 2)
    tau_opt1 = erange1[dis.argmax()]
    fp, fn, tp, tn = compute_ratios(tau_opt1)
    print "with the mutual information, found TP %d, FP %d, TN %d, FN %d" % \
        (tp, fp, tn, fn)
    print "tau %f" % tau_opt1


    fig = plt.figure(3)
    ax = fig.add_subplot(111)
    roc_x, roc_y, erange = mem.cache(roc_curves)(10, p_value=True)
    plot1 = ax.plot(roc_x, roc_y)
    dis = np.sqrt((roc_x - roc_y) ** 2)
    tau_opt = erange[dis.argmax()]
    fp, fn, tp, tn = compute_ratios(tau_opt, p_value=True)
    print "n=10, with the FET, found TP %d, FP %d, TN %d, FN %d" % \
            (tp, fp, tn, fn)
    print "tau %f" % tau_opt

    roc_x, roc_y, erange = mem.cache(roc_curves)(100, p_value=True)
    plot2 = ax.plot(roc_x, roc_y)
    dis = np.sqrt((roc_x - roc_y)**2)
    tau_opt = erange[dis.argmax()]
    fp, fn, tp, tn = compute_ratios(tau_opt, p_value=True)
    print "n=100, with the FET, found TP %d, FP %d, TN %d, FN %d" % \
            (tp, fp, tn, fn)
    print "tau %f" % tau_opt


    roc_x, roc_y, erange = mem.cache(roc_curves)(1000, p_value=True)
    plot3 = ax.plot(roc_x, roc_y)
    fig.legend((plot1[0], plot2[0], plot3[0]),
                ('n = 10', 'n = 100', 'n = 1000'))

    dis = np.sqrt((roc_x - roc_y) ** 2)
    tau_opt = erange[dis.argmax()]
    fp, fn, tp, tn = compute_ratios(tau_opt, p_value=True)
    print "n=1000, with the FET, found TP %d, FP %d, TN %d, FN %d" % \
            (tp, fp, tn, fn)
    print "tau %f" % tau_opt

