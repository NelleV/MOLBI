from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from data import load_data


# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

X, Y = load_data()
h = 0.2
kf = KFold(len(Y), 10)

errors = []
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = {
 'C': [0.1, 1, 5, 10, 50, 100],
 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
}

for i, kernel in enumerate(kernels):
    print kernel
    error = 0
    clf = GridSearchCV(SVC(kernel=kernel),
                        param_grid,
                        fit_params={'class_weight': 'auto'})
    clf.fit(X, Y)
    best_clf = clf.best_estimator_
    plot_x, plot_y, plot_z = [], [], []
    for el, data, _ in clf.grid_scores_:
        plot_x.append(el['C'])
        plot_y.append(el['gamma'])
        plot_z.append(data)

    fig = plt.figure(i)
    fig.clf()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(plot_x, plot_y, plot_z)
    # FIXME

    for train, test in kf:
        best_clf.fit(X[train], Y[train])
        Z = best_clf.predict(X[test])
        error += float((Z != Y[test]).sum()) / len(Z)

        errors.append(error / 10)
