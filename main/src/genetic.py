import numpy as np
import random
from copy import deepcopy
from scipy.stats import zscore

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
    return clusters

class Specimen:
    def __init__(self, cluster: dict, windows: np.ndarray):
        self._cluster = cluster  # e.g., {'A': [0, 1], 'B': [2, 3]}
        self._windows = windows  # np.array of shape (n_windows, 2)

    def cluster(self) -> dict:
        return self._cluster

    def labels(self) -> list[str]:
        return list(self._cluster.keys())

    def windows(self) -> np.ndarray:
        return self._windows

# Helper functions

def extract_series(S: np.ndarray, window: list[int]) -> np.ndarray:
    return S[window[0]:window[1] + 1]

def correlate(w1: list[int], w2: list[int], S: np.ndarray) -> float:
    seg1 = extract_series(S, w1)
    seg2 = extract_series(S, w2)
    len_min = min(len(seg1), len(seg2))
    if len_min < 2:
        return 0.0
    return np.corrcoef(zscore(seg1[:len_min]), zscore(seg2[:len_min]))[0, 1]

def compute_unlabeled_length(specimen: Specimen, series_length: int) -> int:
    intervals = [tuple(win) for win in specimen.windows()]
    if not intervals:
        return series_length
    intervals.sort(key=lambda x: x[0])
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        last = merged[-1]
        if start <= last[1] + 1:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    total_covered = sum(end - start + 1 for start, end in merged)
    return series_length - total_covered

def crossover(parent: Specimen, S: np.ndarray) -> Specimen:
    child = deepcopy(parent)
    for label in child.labels():
        indices = child.cluster()[label]
        if len(indices) >= 2:
            i1, i2 = random.sample(indices, 2)
            w1, w2 = child._windows[i1].copy(), child._windows[i2].copy()
            corr = correlate(w1, w2, S)
            if corr > 0.5:  # only allow crossover if similar
                child._windows[i1], child._windows[i2] = w2, w1
    return child

def mutate(specimen: Specimen, S: np.ndarray, p: float = 0.2) -> Specimen:
    child = deepcopy(specimen)
    windows = child._windows
    for i in range(len(windows)):
        if random.random() < p:
            dl = random.randint(-10, 10)
            dr = random.randint(-10, 10)
            start, end = windows[i]
            new_start = max(0, start + dl)
            new_end = min(len(S)-1, end + dr)
            original = extract_series(S, windows[i])
            new = extract_series(S, [new_start, new_end])
            len_min = min(len(original), len(new))
            if len_min >= 2:
                corr = np.corrcoef(zscore(original[:len_min]), zscore(new[:len_min]))[0, 1]
                if corr < 0.5:
                    continue  # skip mutation if too different
            for j, (os, oe) in enumerate(windows):
                if i != j and not (new_end < os or new_start > oe):
                    corr = correlate([new_start, new_end], [os, oe], S)
                    if corr > 0.8:
                        label_i = next(lbl for lbl, idxs in child._cluster.items() if i in idxs)
                        label_j = next(lbl for lbl, idxs in child._cluster.items() if j in idxs)
                        if label_i != label_j:
                            child._cluster[label_i].remove(i)
                            child._cluster[label_j].append(i)
            child._windows[i] = [new_start, new_end]
    return child

def evaluate(specimen: Specimen, S: np.ndarray, alpha: float = 1.0, beta: float = 0.5) -> float:
    intra_corr = 0.0
    inter_corr = 0.0
    count_intra = 0
    count_inter = 0
    cluster = specimen.cluster()
    windows = specimen.windows()

    for label_i in specimen.labels():
        idxs_i = cluster[label_i]
        for i in range(len(idxs_i)):
            for j in range(i + 1, len(idxs_i)):
                intra_corr += correlate(windows[idxs_i[i]], windows[idxs_i[j]], S)
                count_intra += 1

            for label_j in specimen.labels():
                if label_j <= label_i:
                    continue
                idxs_j = cluster[label_j]
                for idx_i in idxs_i:
                    for idx_j in idxs_j:
                        inter_corr += correlate(windows[idx_i], windows[idx_j], S)
                        count_inter += 1

    avg_intra = intra_corr / count_intra if count_intra else 0
    avg_inter = inter_corr / count_inter if count_inter else 0
    unlabeled = compute_unlabeled_length(specimen, len(S))

    return -(avg_intra - beta * avg_inter) + alpha * unlabeled

def genetic(S: np.ndarray,
            original: Specimen,
            generations: int = 50,
            pmutate: float = 0.2,
            pop_size: int = 20,
            alpha: float = 1.0,
            beta: float = 0.5) -> Specimen:
    population = [deepcopy(original) for _ in range(pop_size)]
    for i in range(generations):
        print(f"Gen : {i}")
        scores = [evaluate(ind, S, alpha, beta) for ind in population]
        elite = population[int(np.argmin(scores))]
        new_pop = [deepcopy(elite)]
        while len(new_pop) < pop_size:
            parent = random.choice(population)
            child = crossover(parent, S)
            child = mutate(child, S, pmutate)
            new_pop.append(child)
        population = new_pop
    best = min(population, key=lambda ind: evaluate(ind, S, alpha, beta))
    return best


from tools.utils import load_temporal_series

S = load_temporal_series("main/charts/Seno-Different-Factors.csv")
windows = [[0, 11], [37, 62], [87, 144], [234, 325], [414, 493], [561, 628], [696, 741], [767, 792], [817, 874], [964, 1043], [1111, 1144]]
clusters = initial(S, windows)
spc = Specimen(clusters, windows)

opt = genetic(S, spc, 50, pop_size=20, alpha=1, pmutate=0.3)

import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_clusters(S, specimen):
    win = specimen.windows()
    clust = specimen.cluster()
    labels = specimen.labels()

    # Usamos un mapa de colores con tantos colores como etiquetas
    cmap = cm.get_cmap('tab10', len(labels))  # Puedes usar 'Set3', 'tab20', etc.
    label_to_color = {label: cmap(i) for i, label in enumerate(labels)}

    # Graficamos la serie temporal
    plt.figure(figsize=(14, 6))
    plt.plot(S, label="Serie Temporal", color="gray", alpha=0.6)

    # Graficamos las ventanas por clase
    for label, indices in clust.items():
        color = label_to_color[label]
        for idx in indices:
            start, end = win[idx]
            plt.axvspan(start, end, alpha=0.4, color=color, label=f"Cluster {label}")

    # Eliminar duplicados en la leyenda
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper right")
    plt.title("Serie Temporal con Ventanas Etiquetadas por Clúster")
    plt.xlabel("Tiempo")
    plt.ylabel("Valor")
    plt.tight_layout()
    plt.show()

plot_clusters(S, opt)