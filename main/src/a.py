import numpy as np
from scipy.stats import zscore
from tools.utils import load_temporal_series
import matplotlib.pyplot as plt
def initial(S: np.array, windows: np.array):

    lmap = {} # {'length' : 'label'} ... {'1': 'A', '2': 'B'}
    clusters = {} # {'cluster': [indices]} ... {'A': [0, 1], 'B': [2, 3]}
    clabel = 65    

    # Creamos las etiquetas para cada longitud de ventana
    for i in range(len(windows)):
        length = windows[i][1] - windows[i][0]
        if length not in lmap:
            lmap[length] = chr(clabel)
            clusters[lmap[length]] = []
            clabel += 1
        clusters[lmap[length]].append(i)



windows = np.array([[12, 37], [62, 87], [145, 234], [325, 414], [459, 460], [494, 559], [629, 694], [742, 767], [792, 817], [875, 964], [1009, 1010], [1044, 1109]])
S = load_temporal_series('main/charts/Seno-Different-Factors.csv')

label_map, clusters = initial(S, windows)
print("Label Map:", label_map)
print("Clusters:", clusters)



# plt.plot(S)

# for label, indices in clusters.items():
#     for idx in indices:
#         start, end = windows[idx]
#         plt.axvspan(start, end, alpha=0.3, label=label, color=plt.cm.tab10(ord(label) % 10))

# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys())
# plt.show()