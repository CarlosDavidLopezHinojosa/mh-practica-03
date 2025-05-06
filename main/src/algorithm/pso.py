import numpy as np
from scipy.stats import zscore
from collections import deque

import concurrent.futures as ft
from os import cpu_count


class particle:
    """
    Clase párticula para reconocimiento de patrones en series temporales
    utilizando el algoritmo evolutivo PSO (Particle Swarm Optimization).
    """
    def __init__(self, lmin: int, lmax: int):
        """
        Inicializa la partícula con un patrón aleatorio y una longitud
        aleatoria dentro de los límites especificados.

        Parameters:
            lmin (int): Longitud mínima del patrón.
            lmax (int): Longitud máxima del patrón.
        """
        self._min = lmin
        self._max = lmax
        self._pattern = np.random.uniform(-1, 1, size=lmax+1)
        self._pattern[0] = np.random.randint(lmin, lmax+1)

        self._velocity = np.zeros(lmax+1)
        self._pbest = self._pattern.copy() # mejor patrón personal
        self._pfitness = -np.inf # mejor fitness personal

    def length(self):
        """
        Devuelve la longitud del patrón.

        Returns:
            int: Longitud del patrón.
        """
        return np.clip(int(self._pattern[0]), self._min, self._max)
    
    def pattern(self):
        """
        Devuelve el patrón de la partícula.

        Returns:
            array: Patrón de la partícula.
        """
        return self._pattern[1:self.length()+1]
    
    def copy(self):
        """
        Devuelve una copia del patrón de la partícula.
        Returns:
            array: Copia del patrón de la partícula.
        """
        return self._pattern.copy()
    
    def move(self, omega: float, c1: float, c2: float, global_best: np.array):
        """
        Actualiza la posición y velocidad de la partícula.

        Parameters:
            omega (float): Coeficiente de inercia.
            c1 (float): Coeficiente cognitivo.
            c2 (float): Coeficiente social.
            global_best (array): Mejor patrón global encontrado.
        """
        r1 = np.random.rand()
        r2 = np.random.rand()
        self._velocity = (omega * self._velocity +
                         c1 * r1 * (self._pbest - self._pattern) +
                         c2 * r2 * (global_best - self._pattern))
        self._pattern += self._velocity
        self._pattern[0] = np.clip(self._pattern[0], self._min, self._max)

    def decode(self):
        """
        Decodifica la partícula para obtener la longitud y los coeficientes
        del patrón.

        Returns:
            tuple: Longitud del patrón y coeficientes del patrón.
        """
        
        return self.length(), self.pattern()


def find_occurrences(S, pattern, threshold):
    """
    Encuentra las ocurrencias de un patrón en una serie temporal.

    Parameters:
        S (array): Serie temporal.
        pattern (array): Patrón a buscar.
        threshold (float): Umbral de correlación.

    Returns:
        list: Lista de índices donde se encuentra el patrón.
    """
    L = len(pattern)
    patt_z = np.nan_to_num(zscore(pattern))
    occ = deque()
    for i in range(len(S) - L + 1):
        window = S[i:i + L]
        win_z = np.nan_to_num(zscore(window))
        if np.std(patt_z) == 0 or np.std(win_z) == 0:
            continue
        corr = np.corrcoef(patt_z, win_z)[0, 1]
        if corr >= threshold:
            occ.append(i)
    return occ


def fitness(S: np.array, ptc: particle, threshold: float):
    """
    Calcula la función de aptitud de una partícula.

    Parameters:
        S (array): Serie temporal.
        ptc (array): Patrón a evaluar.
        threshold (float): Umbral de correlación.

    Returns:
        float: Valor de la función de aptitud.
    """

    def count(S: np.array, pattern: np.array, threshold: float, start: int, end: int):
        L = len(pattern)
        patt_z = np.nan_to_num(zscore(pattern))
        occs = 0
        
        for i in range(start, end - L + 1):
            window = S[i:i+L]
            win_z = np.nan_to_num(zscore(window))
            if np.std(patt_z) == 0 or np.std(win_z) == 0:
                continue
            corr = np.corrcoef(patt_z, win_z)[0, 1]
            if corr >= threshold:
                occs += 1
        return occs
    
    L, pattern = ptc.decode()
    num_threads = cpu_count()
    start = lambda i : i * len(S) // num_threads
    end = lambda i : (i + 1) * len(S) // num_threads
    occs = 0
    with ft.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [ executor.submit(count, S, pattern, threshold, start(i), end(i))
                   for i in range(num_threads) ]
        for future in ft.as_completed(futures):
            occs += future.result()
            
    return occs

def upgrade(ptc: particle, fvalue: float, gvalue: float) -> bool:
    
    """
    Actualiza el mejor patrón personal y el mejor patrón global.

    Parameters:
        ptc (particle): Partícula a evaluar.
        fvalue (float): Valor de la función de aptitud de la partícula.
        gvalue (float): Valor de la función de aptitud del mejor patrón global.

    Returns:
        bool: True si se actualizó el mejor patrón, False en caso contrario.
    """

    if fvalue > ptc._pfitness:
        ptc._pfitness = fvalue
        ptc._pbest = ptc._pattern.copy()
    
    return fvalue > gvalue

def pso(S: np.array, 
        lmax: int, lmin: int, threshold: float, swarm_size: int, 
        iterations: int, omega: float, c1: float, c2: float):
    """
    Algoritmo PSO para encontrar patrones en series temporales.

    Parameters:
        S (array): Serie temporal a analizar.
        lmax (int): Longitud máxima del patrón.
        lmin (int): Longitud mínima del patrón.
        threshold (float): Umbral de correlación.
        swarm_size (int): Tamaño del enjambre de partículas.
        iterations (int): Número de iteraciones del algoritmo.
        omega (float): Coeficiente de inercia.
        c1 (float): Coeficiente cognitivo.
        c2 (float): Coeficiente social.

    Returns:
    """
    gbest = None
    gvalue = -np.inf
    particles = [particle(lmin, lmax) for _ in range(swarm_size)]
    
    for _ in range(iterations):
        for ptc in particles:
            fvalue = fitness(S, ptc, threshold)

            if upgrade(ptc, fvalue, gvalue):
                gbest = ptc.copy()
                gvalue = fvalue
            
        for p in particles:
            p.move(omega, c1, c2, gbest)
    
    return gbest
    
def filter_and_merge_occurrences(raw_occ, L, merge_thresh):
    """
    Filtra solapamientos y luego hace merge de ocurrencias cercanas menores que merge_thresh
    """
    # Filtrado sin solapamientos
    filtered = deque()
    last_end = -1
    for i in sorted(raw_occ):
        if i > last_end:
            filtered.append(i)
            last_end = i + L - 1
    # Merge de cercanos
    merged = deque()
    for start in filtered:
        end = start + L - 1
        if not merged:
            merged.append([start, end])
        else:
            prev_start, prev_end = merged[-1]
            # si la brecha entre prev_end y nuevo start <= merge_thresh
            if start - prev_end <= merge_thresh:
                # extender el prev_end
                merged[-1][1] = end
            else:
                merged.append([start, end])
    return merged


S = np.sin(np.linspace(0, 10, 2000))

a = pso(S, 5, 1, 0.8, 10, 10, 4, 1.5, 1.5)

print(a)

occ = find_occurrences(S, a, 0.8)
print(occ)

i = filter_and_merge_occurrences(occ, a[0], 2)
print(i)