import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import polars as pl

def load_temporal_series(path: str):
    df = pl.read_csv(path)
    index = df.columns[0] if len(df.columns) == 1 else df.columns[1]
    return df[index].to_numpy()

class particle:
    """
    Clase párticula para reconocimiento de patrones en series temporales
    utilizando el algoritmo evolutivo PSO (Particle Swarm Optimization).
    """
    def __init__(self, min_length, max_length):
        """
        Inicializa la partícula con un patrón aleatorio y una longitud
        aleatoria dentro de los límites especificados.

        Parameters:
            min_length (int): Longitud mínima del patrón.
            max_length (int): Longitud máxima del patrón.
        """
        self.pattern = np.random.uniform(-1, 1, size=max_length+1)
        self.pattern[0] = np.random.randint(min_length, max_length+1)

        self.velocity = np.zeros(max_length+1)
        self.best_pattern = self.pattern.copy()
        self.best_fitness = -np.inf
    
    def move(self, omega, c1, c2, global_best):
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
        self.velocity = (omega * self.velocity +
                         c1 * r1 * (self.best_pattern - self.pattern) +
                         c2 * r2 * (global_best - self.pattern))
        self.pattern += self.velocity
        self.pattern[0] = np.clip(self.pattern[0], 1, len(self.pattern)-1)

    def decode(self):
        """
        Decodifica la partícula para obtener la longitud y los coeficientes
        del patrón.

        Returns:
            tuple: Longitud del patrón y coeficientes del patrón.
        """
        length = int(round(self.pattern[0]))
        length = np.clip(length, 1, len(self.pattern)-1)
        coefficients = self.pattern[1:length+1]
        return length, coefficients

def occurrences(S, pattern, threshold):
    L = len(pattern)
    patt_z = np.nan_to_num(zscore(pattern, nan_policy='omit'))
    occs = 0
    for i in range(len(S) - L + 1):
        window = S[i:i + L]
        win_z = np.nan_to_num(zscore(window, nan_policy='omit'))
        # Evitar calcular la correlación si la desviación estándar es cero
        # (esto puede ocurrir si todos los valores son iguales)
        if np.std(patt_z) == 0 or np.std(win_z) == 0:
            continue
        corr = np.corrcoef(patt_z, win_z)[0, 1]
        if corr >= threshold:
            occs += 1
    return occs

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
    patt_z = np.nan_to_num(zscore(pattern, nan_policy='omit'))
    occ = []
    for i in range(len(S) - L + 1):
        window = S[i:i + L]
        win_z = np.nan_to_num(zscore(window, nan_policy='omit'))
        if np.std(patt_z) == 0 or np.std(win_z) == 0:
            continue
        corr = np.corrcoef(patt_z, win_z)[0, 1]
        if corr >= threshold:
            occ.append(i)
    return occ

def fitness(S, pattern, threshold):
    """
    Calcula la función de aptitud de una partícula.

    Parameters:
        S (array): Serie temporal.
        pattern (array): Patrón a evaluar.
        threshold (float): Umbral de correlación.

    Returns:
        float: Valor de la función de aptitud.
    """
    L = int(pattern[0])
    coeffs = pattern[1:L+1]
    occ = occurrences(S, coeffs, threshold)
    return occ

def pso(temporal_series, max_lenght, min_lenght, threshold, swarm_size, iterations, omega, c1, c2):
    """
    Algoritmo PSO para encontrar patrones en series temporales.

    Parameters:
        temporal_series (array): Serie temporal a analizar.
        max_lenght (int): Longitud máxima del patrón.
        min_lenght (int): Longitud mínima del patrón.
        threshold (float): Umbral de correlación.
        swarm_size (int): Tamaño de la población de partículas.
        iterations (int): Número de iteraciones del algoritmo.
        omega (float): Coeficiente de inercia.
        c1 (float): Coeficiente cognitivo.
        c2 (float): Coeficiente social.

    Returns:
    """
    best_pattern = None
    best_fitness = -np.inf
    particles = [particle(min_lenght, max_lenght) for _ in range(swarm_size)]

    to_stop_iterations = 7
    it = 0
    upgrade = False
    
    for _ in range(iterations):
        for p in particles:
            fitness_value = fitness(temporal_series, p.pattern, threshold)
            if fitness_value > p.best_fitness:
                p.best_fitness = fitness_value
                p.best_pattern = p.pattern.copy()
            if fitness_value > best_fitness:
                best_fitness = fitness_value
                best_pattern = p.pattern.copy()
                upgrade = True

        if not upgrade:
            it += 1

        if it >= to_stop_iterations:
            return best_pattern

        for p in particles:
            p.move(omega, c1, c2, best_pattern)
        
        upgrade = False
    
    return best_pattern
    
def filter_and_merge_occurrences(raw_occ, L, merge_thresh):
    """
    Filtra solapamientos y luego hace merge de ocurrencias cercanas menores que merge_thresh
    """
    # Filtrado sin solapamientos
    filtered = []
    last_end = -1
    for i in sorted(raw_occ):
        if i > last_end:
            filtered.append(i)
            last_end = i + L - 1
    # Merge de cercanos
    merged = []
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


# if __name__ == "__main__":
#     # Ejemplo de uso
#     # Generar una serie temporal con 3 patrones
#     t = np.linspace(0, 10, 2000)
#     temporal_series = (np.sin(2 * np.pi * 3 * t) +  # Patrón 1: Senoidal
#                        np.sin(2 * np.pi * 7 * t) * (t > 3) * (t < 6) +  # Patrón 2: Senoidal en ventana
#                        np.sin(2 * np.pi * 15 * t) * (t > 7))  # Patrón 3: Senoidal en ventana
#     max_lenght = 10
#     min_lenght = 2
#     threshold = 0.9
#     swarm_size = 10
#     iterations = 10
#     omega = 0.7
#     c1 = 1.5
#     c2 = 1.5
#     df_path = "/Users/more/Desktop/code/college/metaheuristicas/mh-practica-03/main/charts/Synthetic-three-patterns-with-noise.csv"
#     temporal_series = load_temporal_series(df_path)
#     # temporal_series = [0, 0, 1, 2, 3, 0, 1, 2, 3, 0]

#     best_pattern = pso(temporal_series, max_lenght, min_lenght, threshold, swarm_size, iterations, omega, c1, c2)
#     L = int(best_pattern[0])
#     coeffs = best_pattern[1:L+1]
#     raw_occ = find_occurrences(temporal_series, coeffs, threshold)
#     merge_thresh = 4
#     best_pattern = filter_and_merge_occurrences(raw_occ, L, merge_thresh)
#     print("Mejor patrón encontrado:", best_pattern)
#     plt.figure(figsize=(10,4))
#     plt.plot(temporal_series, label='Serie original')
#     for start, end in best_pattern:
#         plt.axvspan(start, end, color='orange', alpha=0.3)
#     plt.title('Detección de patrones con merge de cercanos')
#     plt.xlabel('Índice')
#     plt.ylabel('Valor')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()