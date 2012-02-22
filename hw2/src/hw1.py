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

p_value = mem.cache(compute_pvalues)(data, num_perm=1500)


###############################################################################
# Compute graph
def compute_graph(p_value, alpha=0.75):
    graph = (p_value > alpha)

    nodes = []
    for i in range(graph.shape[0]):
        el = []
        for j in range(graph.shape[0]):
            if graph[i, j]:
                el.append(j)
        nodes.append(el)
    return graph, nodes

graph, nodes = compute_graph(p_value)

###############################################################################
# Let's visualise the data with pydot
print "Generating graph"


def export_graph(nodes, filename='example_graph.dot'):
    dot_graph = pydot.Dot(graph_type='graph')
    logfile = sys.stdout
    for i in range(len(nodes)):
        logfile.write('.')
        logfile.flush()
        for k in nodes[i]:
            if k > i:
                continue
            edge = pydot.Edge("%d" % i, "%d" % k)
            dot_graph.add_edge(edge)
    print "writing file"
    dot_graph.write_raw(filename)

###############################################################################
# Now, let's destroy all loops

print "cleaning up graph"
sets = []
for i in range(data.shape[0]):
    for j in nodes[i]:
        for k in nodes[j]:
            if k != i:
                sets.append((i, j, k))

for i, j, k in sets:
    ij = p_value[i, j]
    ik = p_value[i, k]
    jk = p_value[k, j]

    if ij <= ik and ij <= jk:
        p_value[i, j] = 0
        p_value[j, i] = 0
    elif ik <= ij and ik <= jk:
        p_value[i, k] = 0
        p_value[k, i] = 0
    else:
        p_value[j, k] = 0
        p_value[k, j] = 0

graph, nodes = compute_graph(p_value)
export_graph(nodes, filename='final_graph.dot')

###############################################################################
# Compare to results
true_graph = load_data(filename='interactions.csv')
true_graph, true_nodes = compute_graph(true_graph)


def compute_precision_recall(nodes, true_nodes):
    precision = 0.
    total_precision = 0.
    recall = 0.
    total_recall = 0.
    for i in range(len(nodes)):
        for k in nodes[i]:
            total_precision += 1
            if k in true_nodes[i]:
                precision += 1
        for k in true_nodes[i]:
            total_recall += 1
            if k in nodes[i]:
                recall += 1
    if total_precision != 0:
        precision /= total_precision
    recall /= total_recall
    return precision, recall

precision, recall = compute_precision_recall(nodes, true_nodes)

###############################################################################
# Compute precision-recall curve with alpha varying

precisions = []
recalls = []
for alpha in range(1, 99):
    alpha = float(alpha) / 100
    graph, nodes = compute_graph(p_value, alpha=alpha)
    precision, recall = compute_precision_recall(nodes, true_nodes)
    precisions.append(precision)
    recalls.append(recall)

for alpha, precision, recall in zip(range(1, 99), precisions, recalls):
    if precision == 1 and recall == 1:
        print "best alpha %d" % alpha
        break

fig = plt.figure(0)
ax = fig.add_subplot(111)
ax.plot(recalls, precisions, 'o-')

graph, nodes = compute_graph(p_value, alpha=float(alpha) / 100)
