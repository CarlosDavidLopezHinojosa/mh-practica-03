import numpy as np
import heapq as heap
import matplotlib.pyplot as plt

from scipy.stats import zscore
from collections import deque, Counter
from tools.utils import load_temporal_series
from algorithm.pso import reduce

from warnings import filterwarnings
filterwarnings("ignore", category=RuntimeWarning)

"""
Objetivo buscar patrones dada una serie temporal y etiquetar dichos patrones.

Representación de individuos:
    - Vectores de enteros que representan ventanas de las serie temporal, pueden tener distintas longitudes.
    Ej: [1, 5, 9, 12, 14], [2, 9, 14]
Fitness
    - Correspondencias entre las medias de las diferentes ventanas
    Se correlacionan las medias de las diferentes ventanas para determinar si son de la misma clase.

    Es decir si μ1 - μ2 ~= 0 Pertenecen a la misma clase, varias ventanas pueden tener distintas clases, 
    hay que tener cuidado ya que puede haber el mismo patrón en diferentes alturas de las serie temporal.

    Hay penalización por demasidados huecos de la gráfica sin cubrir/ no etiquetados

Algoritmo: AG

    - Selección:
        Torneo Binario
    - Cruce: 
        Cruce de uno o varios puntos
    - Mutación:
        Cambio de indices de algunas ventanas 
        Ej: [1, 3, 4, 8] -> [1, 2, 5, 8]
    - Reemplazo: 
        Completo o solo los mejores en el primer caso con elitismo de 2 o 3

Etiquetado:
    1. Se escoge una ventana de pivote y se compara su media con la demás ventanas si estás pasan el test estadistico pertenecen a la misma clase que la ventana 
    pivote.

    2. Se continua con la siguiente ventana si esta ya tiene una clase elegida no se compara con ninguna otra se pasa a la siguiente iteración.
"""

# Gráficas

def plot_windows(S: np.array, windows: np.array):
    plt.plot(S)
    plt.scatter(windows, S[windows.astype(int)], color="red")
    plt.show()


# Creación de Individuos

"""
    Lo individuos serán numpy arrays que representan ventanas del tipo [[1,2], [5,6]...], pero aplanados, es decir, 
    la codificación será -> [1, 2, 5, 6].

    Inicialización:
    - Inicialización Aleatoria de Índices ordenados, podemos utilizar un radio en formato aleatorio, es decir, 
    la siguiente ventana tendrá una distancia respecto a la siguiente.

    Parametros:
    - Mínimo tamaño de ventana: (int)
    - Máximo tamaño de ventana: (int)
    - Radio (Distancia entre ventanas): **Esto lo vamos a hacer con la distribución de pareto**
"""
import numpy as np
from scipy.stats import zscore
from collections import deque

def windows(S: np.ndarray, w: int, threshold: float):
    """
    Desliza una ventana de tamaño w por S y devuelve los segmentos cuya correlación
    con la ventana pivote aleatoria supera `threshold`.
    """
    size = len(S)
    # Escoger pivote
    pivot_start = np.random.randint(0, size - w + 1)
    pivot = S[pivot_start:pivot_start + w]
    pivot_z = zscore(pivot, nan_policy='omit')

    matches = deque()
    matches.append((pivot_start, pivot_start + w - 1))

    for i in range(size - w + 1):
        if i == pivot_start:
            continue
        seg = S[i:i + w]
        seg_z = zscore(seg, nan_policy='omit')
        coef = np.corrcoef(pivot_z, seg_z)[0, 1]
        if coef >= threshold:
            matches.append((i, i + w - 1))

    unique = sorted(set(matches), key=lambda x: x[0])
    flat = np.array([idx for (s, e) in unique for idx in (s, e)], dtype=np.int64)
    return flat


def population(S: np.array, pop_size: int, threshold: float):
    """
    Genera una población de ventanas con tamaños proporcionales a la serie.
    """
    size = len(S)
    lmax = int(size ** 0.6)
    lmin = max(2, int(np.log(size)))
    # Géométricamente repartir entre lmin y lmax
    ws = np.unique(np.round(np.geomspace(lmax, len(S) - lmax, num=pop_size))).astype(int)
    pops = []
    # Si hay menos tamaños que individuos, repetir cíclicamente
    for i in range(pop_size):
        w = ws[i % len(ws)]
        pops.append(windows(S, w, threshold))
    return pops

# Creación de la función de fitness

"""
    La función de fitness deberá cumplir varios objetivos:

    1. Cubrir el máximo de la gráfica, las ventas etiquetadas deberan ocupar la máxima area de la serie.
    2. Máximizar el número de ocurrencias de clases (intra clase), podemos probar teniendo en cuenta la distancia interclase.
    3. Debe haber penalización por complejidad, aquellas ventanas con muchas clases tendran penalización

    Para la identificación de patrones y clasificación haremos los siguiente:
    a). Seleccionamos un pivote
    b). Comparamos con las demás ventanas en caso que estadísticamente esten correlacionadas serán de la misma clase (Cuidado con diferentes tamaños de ventana)
        - Hacemos split de la ventana de mayor tamaño:
            Ej: w1: [1,5] (longitud 4) , w2: [6, 11] (longitud 5), hacemos correlación w1 en w2[:4] y w2[-4:].
        - Podemos hacer DTW, aunque creo que no asegura encontrar diferencias entre patrones en distinta escala y altura.
    c). Siguiente pivote
    d). En caso de que una ventana este ya etiquetada podemos hacer 2 cosas buscamos como en el paso b), permitimos que un patrón tenga más de una clase 
        la ventaja de esto es que al tener redundancia busca mejor clases con diferencias, habrá que hacer filtrado despues:
        Ej: Tenemos las siguientes ventanas [[1, 3], [5, 9], [10, 12], [17, 20]], con sus etiquetas.
            [['A'], ['B'], ['A', 'B'], ['C']]
"""

def score(a: np.array):
    return np.nan_to_num(zscore(a, nan_policy='omit'))

import numpy as np
from scipy.stats import gmean

def mark(S: np.array, Ascores: np.array, sectionB: np.array, threshold: float):
    """
    Compara dos ventanas (ya convertidas a z-scores) y decide si pertenecen
    al mismo patrón según el umbral. Maneja ventanas de distinta longitud.
    """
    Bscores = score(sectionB)
    lA = len(Ascores)
    lB = len(Bscores)

    # --- Nuevo guard: si alguna ventana está vacía, no hay comparación ---
    if lA == 0 or lB == 0:
        return (False, 0.0)

    # Caso de igual longitud: correlación directa
    if lA == lB:
        coef = np.corrcoef(Ascores, Bscores)[0, 1]
        return (bool(coef >= threshold), float(coef))

    # Determinar cuál es la más larga y cuál la más corta
    if lA > lB:
        longer, shorter = Ascores, Bscores
    else:
        longer, shorter = Bscores, Ascores

    len_long = len(longer)
    len_short = len(shorter)

    # Calcular número de divisiones necesarias
    num_splits = len_long // len_short
    if num_splits >= 3 or len_long % len_short != 0:
        return (False, 0.0)

    coefs = []
    for i in range(num_splits):
        segment = longer[i * len_short:(i + 1) * len_short]
        coef = np.corrcoef(segment, shorter)[0, 1]
        if coef > 0:
            return (False, 0.0)
        coefs.append(abs(coef))  # usamos valor absoluto para evitar signos opuestos

    coef_mean = gmean(coefs)
    return (bool(coef_mean >= threshold), float(coef_mean))


def label(S: np.array, windows: np.array, threshold: float):
    size = len(windows) // 2
    labeled = [False] * size
    clusters = {i: {'cluster': '?', 'dominance': 0.0} for i in range(size)}
    dominances = {}

    cluster = 65

    for label in range(size):

        if labeled[label]:  # Si ya esta etiquetado no es pivote
            continue

        clusters[label]['cluster'] = chr(cluster)
        dominances[chr(cluster)] = 0.0

        startA = windows[2 * label]
        endA = windows[2 * label + 1]
        sectionA = S[startA: endA + 1]
        Ascores = score(sectionA)

        for next in range(label + 1, size):

            startB = windows[2 * next]
            endB = windows[2 * next + 1]
            sectionB = S[startB: endB + 1]

            dominated, dominance = mark(S, Ascores, sectionB, threshold)
            if dominated:
                labeled[next] = True
                dom = dominance if dominance > clusters[next]['dominance'] else clusters[next]['dominance']
                clt = chr(cluster) if dominance > clusters[next]['dominance'] else clusters[next]['cluster']
                clusters[next]['cluster'] = clt
                clusters[next]['dominance'] = dom

        cluster += 1
    return clusters, cluster - 65


def fitness(S: np.array, windows: np.array, threshold: float, simplicity: float = 0.1):
    clusters, num_cluster = label(S, windows, threshold)
    counts = Counter([clusters[i]['cluster'] for i in clusters])
    
    fit = 0.0

    for cluster, count in counts.items():
        # if count == 1: 
        #     continue

        area = sum(
            windows[2 * i + 1] - windows[2 * i] + 1
            for i in clusters
            if clusters[i]['cluster'] == cluster
        )
        dominance = sum(
            clusters[i]['dominance']
            for i in clusters
            if clusters[i]['cluster'] == cluster
        )
        fit += dominance * area * count

    # Penalización por área no cubierta por ventanas
    covered_area = sum(
        windows[2 * i + 1] - windows[2 * i] + 1
        for i in range(len(windows) // 2)
    )
    uncovered_area = len(S) - covered_area
    fit -= uncovered_area   # Ajustar el factor de penalización según sea necesario

    fit -= simplicity  * len(windows) // 2  # Penalización por complejidad
    return fit


# Operador de selección

def tournament(pop: list[np.array], S: np.array, threshold: float, simplicity: float = 0.1):
    k = 3
    selected = np.random.choice(range(len(pop)), k).astype(np.int64)
    selected_individuals = [pop[i] for i in selected]
    return max(selected_individuals, key=lambda x: fitness(S, x, threshold, simplicity))

# Operador de cruce

import numpy as np

def uniform_crossover_adapted(p1, p2):
    """
    Cruce uniforme adaptado para vectores de distinta longitud.
    Devuelve dos hijos combinando aleatoriamente genes de los padres.
    """
    max_len = max(len(p1), len(p2))
    
    # Usar -1 como relleno temporal
    pad_val = -1
    pad_p1 = np.pad(p1, (0, max_len - len(p1)), constant_values=-1)
    pad_p2 = np.pad(p2, (0, max_len - len(p2)), constant_values=-1)

    
    # Generar máscara aleatoria
    mask = np.random.randint(0, 2, size=max_len, dtype=bool)
    
    # Cruce
    h1 = np.where(mask, pad_p1, pad_p2)
    h2 = np.where(mask, pad_p2, pad_p1)
    
    # Filtrar los valores -1 (inexistentes)
    h1 = h1[h1 != -1]
    h2 = h2[h2 != -1]
    # print(h1, h2)


    return h1.astype(int), h2.astype(int)

# Operador de mutación

"""
    El operador de mutación es el más importane del algoritmo, es el que va a permitir 
    que se varien las longitudes de la ventanas y crear nuevos patrones más complejos
    o más simples, realizara las siguientes operaciones.

    1). Expansión Lateral Derecha, Un valor concreto de las ventanas abarcará más, hay que tener varios casos.
        - Los indices consecutivos sean menores que el índice mutado (Overlapping), deberán incrementar tambien con este.
    2). Expansión Lateral Izquierda, Un valor concreto de las ventanas abarcará menos, hay que tener varios casos.
        - Los indices consecutivos sean mayores que el índice mutado (Underlapping), deberán reducir tambien con este.
    3). Adopción, Se improtará un valor dividiendo una ventana.
    4). Secuestro, Se eliminara un valor ampliando una ventana.

    Los dos primeros puntos deberan usarse en fase últimas del algorimo, para refinar.
    Los dos últimos se usarán para exploración, en fases tempranas
"""


def mold(ind: np.ndarray, pmutate: float, stage: float):
    pass
    l = len(ind)
    mask = np.random.random(l) > pmutate
    # Tomamos índices, no valores
    sel_indices = np.where(mask)[0]
    # Orden descendente para manejar eliminaciones sin desajustes
    sel_indices = np.sort(sel_indices)[::-1]

    exploration = stage
    exploit     = 1 - stage

    def up(idx):
        nonlocal ind
        value = np.random.randint(5, 20)
        if idx + 1 < len(ind) and ind[idx] + value < ind[idx + 1]:
            ind[idx] += value

    def down(idx):
        nonlocal ind
        value = np.random.randint(5, 20)
        if idx - 1 >= 0 and ind[idx] - value > ind[idx - 1]:
            ind[idx] -= value

    def segment(idx):
        nonlocal ind
        if idx == 0:
            return
        prev = ind[idx - 1]
        curr = ind[idx]
        if curr - prev >= 2:
            new = np.random.randint(prev + 1, curr)
            # Insertamos y actualizamos in-place
            ind = np.insert(ind, idx, new)

    def consolide(idx):
        nonlocal ind
        if 0 <= idx < len(ind) and len(ind) > 1:
            ind = np.delete(ind, idx)

    # mutators = [up, down, segment, consolide]
    mutators = [segment, consolide, up, down]

    # weights = np.random.random(2) * np.array([exploration, exploration, exploit, exploit])
    weights = np.random.random(4)

    probs   = weights / weights.sum()

    for idx in sel_indices:
        if idx < len(ind):
            op = np.random.choice(4, p=probs)
            mutators[op](idx)



# Operador de reemplazo

def total(oldpop, newpop):
    oldpop[::] = newpop[::]

def genetic(S, generations, popsize, threshold, simplicity, elitesize):
    pmutate = 1
    pop = population(S, popsize, threshold)
    elite = []

    for gen in range(generations):
        print(f"Generación: {gen} de {generations}")
        newpop = []

        # Generar nuevos hijos
        for _ in range(popsize // 2):
            ind1 = tournament(pop, S, threshold, simplicity)
            ind2 = tournament(pop, S, threshold, simplicity)

            # Desempaquetar correctamente los hijos
            child1, child2 = uniform_crossover_adapted(ind1, ind2)

            stage = gen / generations
            mold(child1, pmutate, stage)
            mold(child2, pmutate, stage)

            newpop.extend([child1, child2])


        # Seleccionar élites correctamente
        combined = pop + elite
        elite = heap.nlargest(elitesize, combined, key=lambda x: fitness(S, x, threshold, simplicity))

        # Reemplazar población
        pop = newpop

    # Devolver el mejor individuo final
    return max(pop + elite, key=lambda x: fitness(S, x, threshold, simplicity))


path = 'main/charts/ecg-sintetico.csv'
S = load_temporal_series(path, numpy=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

def plot_classified_windows(S: np.ndarray, windows: np.ndarray, clusters_output):
    """
    Plotea la serie temporal S, resalta las ventanas y muestra sus clases.
    
    Args:
        S (np.ndarray): Serie temporal (1D) a plotear.
        windows (np.ndarray): Array aplanado de índices de ventanas [s0, e0, s1, e1, ...].
        clusters_output (dict or tuple): Salida de la función label, puede ser directamente el dict
            de clusters o la tupla (clusters, num_clusters).
    """
    # Manejar si se pasó la tupla completa de label()
    if isinstance(clusters_output, tuple):
        clusters, _ = clusters_output
    else:
        clusters = clusters_output

    # Preparar plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(S, label='Serie Temporal')

    # Extraer clases únicas y asignar colores
    all_classes = sorted({clusters[i]['cluster'] for i in clusters})
    cmap = plt.get_cmap('tab10')
    color_map = {cl: cmap(idx % 10) for idx, cl in enumerate(all_classes)}

    # Recolectar leyenda
    legend_patches = [Patch(facecolor=color_map[cl], alpha=0.3, label=f'Clase {cl}')
                      for cl in all_classes]

    # Dibujar ventanas
    n = len(windows) // 2
    for i in range(n):
        start = windows[2*i]
        end = windows[2*i + 1]
        cl = clusters[i]['cluster']
        color = color_map.get(cl, 'gray')
        # Sombrear región
        ax.axvspan(start, end, color=color, alpha=0.3)
        # Marcar puntos extremos
        ax.scatter([start, end], [S[start], S[end]], color=color, edgecolor='k')
        # Etiquetar en el centro de la ventana
        mid = int((start + end) / 2)
        ax.text(mid, S[mid], cl, ha='center', va='bottom', fontweight='bold')

    # Configurar plot
    ax.set_title('Serie Temporal con Ventanas y Clases')
    ax.set_xlabel('Índice')
    ax.set_ylabel('Valor')
    ax.legend(handles=legend_patches, loc='upper right')
    plt.tight_layout()
    plt.show()

# Ejemplo de uso:

from tools.utils import load_temporal_series
from algorithm.pso import reduce
path = 'main/charts/Synthetic-three-patterns-with-noise.csv'
S = load_temporal_series(path, numpy=True)
red = reduce(S)
plot_windows(S, red)

# Mapear las ventanas de la serie reducida a la serie original
def map_to_original(reduced_windows, reduction_indices):
    """
    Mapea las ventanas de la serie reducida a la serie original.

    Args:
        reduced_windows (np.ndarray): Ventanas en la serie reducida.
        reduction_indices (np.ndarray): Índices de la serie original que corresponden a la reducida.

    Returns:
        np.ndarray: Ventanas mapeadas a la serie original.
    """
    original_windows = []
    for i in range(len(reduced_windows) // 2):
        start = reduction_indices[reduced_windows[2 * i]]
        end = reduction_indices[reduced_windows[2 * i + 1]]
        original_windows.extend([start, end])
    return np.array(original_windows).flatten().astype(np.int64)

s = S[red]
best = genetic(s, 200, 100, 0.9, 0.1, 3)
labels = label(s, best, 0.7)

# Mapear las ventanas de la serie reducida a la serie original
mapped_windows = map_to_original(best, red)
plot_classified_windows(S, mapped_windows, labels)
