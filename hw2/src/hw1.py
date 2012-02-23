import sys
import numpy as np
import pydot
from matplotlib import pyplot as plt

from sklearn.externals.joblib import Memory

from data import load_data
from mutual_information import mutual_information

mem = Memory(cachedir='./')

data = load_data()
n_samples = data.shape[1]

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

I = mem.cache(mutual_information)(data)


###############################################################################
# P - Values
#
# Compute p_values by permuting the data matrix, and recomputing mutual
# information.
def compute_pvalues(data, num_perm=500):
    logfile = sys.stdout
    p_value = np.zeros((153, 153))
    data_shuffled = data.copy()
    perc = 0
    for n in range(num_perm):
        if n * 100 / num_perm >= perc:
            logfile.write('\r%d %%' % perc)
            logfile.flush()
            perc += 1
        for i in range(len(data)):
            np.random.shuffle(data_shuffled[i])

        I_shuffled = mutual_information(data_shuffled)
        p_value += (I_shuffled > I).astype(float)

    p_value /= num_perm
    return p_value


def clean_up_p_value(p_value):
    p_value += np.identity(153)
    # Add some noise, to  deal with p_value's equal to 0
    # p_value += 1e-16 * np.random.random(p_value.shape)
    return p_value

p_value = mem.cache(compute_pvalues)(data, num_perm=50)
p_value = mem.cache(clean_up_p_value)(p_value)

def clean_up_graph(p_value, nodes, graph):
    print "number of nodes: %d" % graph.sum()
    for i in range(len(graph)):
        graph[i, i] = False
        for j in range(len(graph)):
            if not graph[i, j]:
                continue
            for k in range(len(graph)):
                if not graph[j, k]:
                    continue
                elif graph[k, i]:
                    ij = p_value[i, j]
                    ik = p_value[i, k]
                    jk = p_value[k, j]

                    if ij > ik and ij > jk:
                        graph[i, j] = False
                        graph[j, i] = False
                    elif ik > ij and ik > jk:
                        graph[i, k] = False
                        graph[k, i] = False
                    else:
                        graph[j, k] = False
                        graph[k, j] = False
    print "number of remaining nodes: %d" % graph.sum()
    return graph


def compute_graph(p_value, alpha=0.75, true_graph=False):
    if true_graph:
        graph = p_value == 1
    else:
        graph = (p_value <= alpha)

    nodes = []
    return graph, nodes


def export_graph(graph, filename='example_graph.dot'):
    dot_graph = pydot.Dot(graph_type='graph')
    logfile = sys.stdout
    for i in range(len(graph)):

        logfile.write('.')
        logfile.flush()
        for j in range(len(graph)):
            if j > i or not graph[j, i]:
                continue
            edge = pydot.Edge("%d" % i, "%d" % j)
            dot_graph.add_edge(edge)
    print "writing file"
    dot_graph.write_raw(filename)


###############################################################################
def compute_precision_recall(graph, true_graph):
    tp = graph[true_graph].sum()
    if graph.sum() != 0:
        precision = float(tp) / graph.sum()
    else:
        precision = 0

    recall = float(tp) / true_graph.sum()
    print tp, precision, recall

    return precision, recall

true_graph = load_data(filename='interactions.csv')
true_graph, true_nodes = compute_graph(true_graph, true_graph=True)
true_graph = clean_up_graph(true_graph, true_nodes, true_graph)

precisions = []
recalls = []
for alpha in range(0, 1000, 20):
    alpha = float(alpha) / 1000
    graph, nodes = mem.cache(compute_graph)(p_value, alpha=alpha)
    graph = mem.cache(clean_up_graph)(p_value, nodes, graph)
    precision, recall = mem.cache(compute_precision_recall)(graph, true_graph)
    precisions.append(precision)
    recalls.append(recall)

# Recompute for alpha very small
alpha = 0.002
graph, nodes = mem.cache(compute_graph)(p_value, alpha=alpha)
graph = mem.cache(clean_up_graph)(p_value, nodes, graph)
export_graph(graph)

fig = plt.figure(0)
ax = fig.add_subplot(111)
ax.plot(recalls, precisions, 'o-')

###############################################################################
# False discovery rate
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.plot(recalls, 1 - np.array(precisions), 'o-')
