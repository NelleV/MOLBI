from sklearn import neighbors
from sklearn.cross_validation import KFold
from sklearn.externals.joblib import Memory
from sklearn.decomposition import pca
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from data import load_data
import numpy as np

mem = Memory('./joblib')
# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

n_neighbors = 4
X, Y = mem.cache(load_data)()
h = 0.02
tmp_errors = []
k = 10
n_iter = 20

def compute_knn(X, Y, weights):
    errors = []
    for n_neighbors in range(1, 30):
        # Let's shuffle the data again, and do that several time
        tmp_error = 0
        print n_neighbors
        for j in range(n_iter):
            indxs = np.arange(len(Y))
            np.random.shuffle(indxs)
            X = X[indxs]
            Y = Y[indxs]
            kf = KFold(len(Y), k)
            error = 0
            for train, test in kf:
                clf = neighbors.KNeighborsClassifier(n_neighbors,
                                                     weights=weights)
                clf.fit(X[train], Y[train])
                Z = clf.predict(X[test])
                error += float((Z != Y[test]).sum()) / len(Z)
            tmp_error += error / k
        errors.append(tmp_error / n_iter)
    return errors


for i, weights in enumerate(['distance', 'uniform']):
    errors = mem.cache(compute_knn)(X, Y, weights)
    tmp_errors.append(errors)

plt.figure(1)
plt.plot(tmp_errors[0], tmp_errors[1])
plt.legend(['uniform', 'distance'])


# Plot also the training points
plt.title("KNN - error rates")
plt.axis('tight')
plt.show()


k = np.argmin(tmp_errors[0])
print "minimal error", np.min(tmp_errors[0]), "for", k
k = np.argmin(tmp_errors[1])
print "minimal error", np.min(tmp_errors[1]), "for", k


k = 12
for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(k, weights=weights)
    clf.fit(X[:, :2], Y)

    # Plot the decision boundary. For that, we will asign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold)
    plt.title("3-Class classification (k = %i, weights = '%s')"
             % (k, weights))
    plt.axis('tight')

plt.show()
