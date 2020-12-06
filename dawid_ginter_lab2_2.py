from operator import mod
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as shc
from yellowbrick.cluster import SilhouetteVisualizer


input_file = "Sales_Transactions_Dataset_Weekly.csv"

# Ilość klastrów
no_of_clusters = 5
# Podobieństwo (odległość)
affinity_type = "euclidean"

# def plot_dendrogram(model, **kwargs):

#     children = model.children_

#     distance = np.arange(children.shape[0])

#     no_of_observations = np.arange(2, children.shape[0]+2)

#     linkage_matrix = np.column_stack(
#         [children, distance, no_of_observations]).astype(float)

#     dendrogram(linkage_matrix, **kwargs)


data_source = pd.read_csv(input_file, sep=",", header=0, index_col=0)

scikit_input_data = data_source.to_numpy()

# model = AgglomerativeClustering(n_clusters=no_of_clusters, affinity=affinity_type)

db = DBSCAN(eps=0.3, min_samples=10).fit(scikit_input_data)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = scikit_input_data[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = scikit_input_data[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

# model = model.fit(scikit_input_data)
# plot_dendrogram(model, labels=model.labels_)
# plt.show()
