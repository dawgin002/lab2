import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

input_file = "Sales_Transactions_Dataset_Weekly.csv"

# Możliwe rodzaje wskaźników
all_poss = ['.', 'o', 'v', '^', '>', '<', 's',
            'p', '*', 'h', 'H', 'D', 'd', '1', '', '']


# Zadanie 1 - ładowanie danych z pliku

data_source = pd.read_csv(input_file, sep=",", header=0, index_col=0)
# column_headers = data_source.columns.values

scikit_input_data = data_source.to_numpy()

# KMeans

kmeans = KMeans(n_clusters=5, init="random")

label = kmeans.fit_predict(scikit_input_data)

# KMeans++

kmeans_pp = KMeans(n_clusters=5, init="k-means++")

labels = np.unique(label)

f, (ax1, ax2) = plt.subplots(1, 2, sharex=False, sharey=False)

centroids = kmeans.cluster_centers_
for i in labels:
    ax1.scatter(scikit_input_data[label == i, 0],
                scikit_input_data[label == i, 1], label=i, marker=all_poss[i])
    ax1.scatter(centroids[:, 0], centroids[:, 1], marker="*", c='blue', s=50)

label = kmeans_pp.fit_predict(scikit_input_data)
centroids_pp = kmeans_pp.cluster_centers_

labels = np.unique(label)

for i in labels:
    ax2.scatter(scikit_input_data[label == i, 0],
                scikit_input_data[label == i, 1], label=i, marker=all_poss[i])
    ax2.scatter(centroids[:, 0], centroids[:, 1], marker="*", c='blue', s=50)

# blue_star = (color = 'blue', marker='*',label='Blue stars')
# plt.plot()
# patch = mpatches.Patch(color='blue', label='Centroidy')
# handles1, labels1 = ax1.get_legend_handles_labels()
# handles2, labels2 = ax2.get_legend_handles_labels()
# handles1.append(patch)
# handles2.append(patch)
ax1.legend()
ax1.set_title("KMeans")
ax2.legend()
ax2.set_title("KMeans++")

plt.show()
# Wyświetlania klastrów
# https://www.askpython.com/python/examples/plot-k-means-clusters-python
