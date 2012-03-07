from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
#from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from data import load_data


# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

X, Y = load_data()
h = 0.2
kf = KFold(len(Y), 10)

errors = []
for n_estimators in range(1, 50, 5):
    print n_estimators
    error = 0
    for train, test in kf:
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(X[train], Y[train])
        Z = clf.predict(X[test])
        error += float((Z != Y[test]).sum()) / len(Z)

    errors.append(error / 10)
