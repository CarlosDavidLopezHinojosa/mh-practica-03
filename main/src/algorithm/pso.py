import numpy as np
from scipy.stats import zscore
from collections import deque
from typing import Deque, List, Tuple, Optional

from fastevaluate import fastfitness


class particle:
    """
    Representa una partícula para reconocimiento de patrones en series temporales
    mediante el algoritmo Particle Swarm Optimization (PSO).
    """

    def __init__(self, lmin: int, lmax: int) -> None:
        """
        Inicializa una partícula con un patrón aleatorio y longitud aleatoria.

        Args:
            lmin (int): Longitud mínima del patrón.
            lmax (int): Longitud máxima del patrón.
        """
        self._min: int = lmin
        self._max: int = lmax
        # Patrón: el primer valor indica la longitud, el resto los coeficientes
        self._pattern: np.ndarray = np.random.uniform(-1, 1, size=lmax + 1).astype(np.float64)
        self._pattern[0] = np.random.randint(lmin, lmax + 1)

        self._velocity: np.ndarray = np.zeros(lmax + 1, dtype=np.float64)
        self._pbest: np.ndarray = self._pattern.copy()  # mejor patrón personal
        self._pfitness: float = -np.inf  # mejor fitness personal

    def length(self) -> int:
        """
        Obtiene la longitud actual del patrón de la partícula.

        Returns:
            int: Valor de la longitud, acotado entre lmin y lmax.
        """
        raw_len = int(self._pattern[0])
        return int(np.clip(raw_len, self._min, self._max))

    def pattern(self) -> np.ndarray:
        """
        Extrae los coeficientes del patrón de la partícula.

        Returns:
            np.ndarray: Vector de coeficientes de longitud `length()`.
        """
        L = self.length()
        return self._pattern[1 : L + 1]

    def copy(self) -> np.ndarray:
        """
        Crea una copia completa de la representación interna de la partícula.

        Returns:
            np.ndarray: Copia del arreglo interno _pattern.
        """
        return self._pattern.copy()

    def move(self, omega: float, c1: float, c2: float, gbest: np.ndarray) -> None:
        """
        Actualiza velocidad y posición según la fórmula PSO.

        Args:
            omega (float): Coeficiente de inercia.
            c1 (float): Coeficiente cognitivo (atracción al pbest).
            c2 (float): Coeficiente social (atracción al gbest).
            gbest (np.ndarray): Mejor patrón global encontrado.
        """
        r1, r2 = np.random.rand(), np.random.rand()
        cognitive = c1 * r1 * (self._pbest - self._pattern)
        social = c2 * r2 * (gbest - self._pattern)
        self._velocity = omega * self._velocity + cognitive + social
        self._pattern += self._velocity
        # Garantizar que la longitud esté dentro de los límites
        self._pattern[0] = np.clip(self._pattern[0], self._min, self._max)

    def decode(self) -> Tuple[int, np.ndarray]:
        """
        Decodifica la partícula a su forma (longitud, coeficientes).

        Returns:
            Tuple[int, np.ndarray]: Longitud del patrón y vector de coeficientes.
        """
        return self.length(), self.pattern()


def occurrences(series: np.ndarray, pattern: np.ndarray, threshold: float) -> int:
    """
    Cuenta cuántas veces aparece un patrón en una serie temporal superando 
    un umbral de correlación.

    Args:
        series (np.ndarray): Serie temporal de datos.
        pattern (np.ndarray): Vector de coeficientes a buscar.
        threshold (float): Umbral mínimo de correlación ([-1, 1]).

    Returns:
        int: Número de ocurrencias detectadas.
    """
    L: int = len(pattern)
    patt_z = np.nan_to_num(zscore(pattern))
    count: int = 0

    for i in range(len(series) - L + 1):
        window = series[i : i + L]
        win_z = np.nan_to_num(zscore(window))
        if np.std(patt_z) == 0 or np.std(win_z) == 0:
            continue
        corr = np.corrcoef(patt_z, win_z)[0, 1]
        if corr >= threshold:
            count += 1
    return count


def find_occurrences(series: np.ndarray, pattern: np.ndarray, threshold: float) -> Deque[int]:
    """
    Encuentra índices de inicio de cada ocurrencia de un patrón en la serie 
    usando correlación z-score.

    Args:
        series (np.ndarray): Serie temporal.
        pattern (np.ndarray): Patrón de coeficientes.
        threshold (float): Umbral de correlación para admitir coincidencia.

    Returns:
        Deque[int]: Cola con índices de inicio donde apareció el patrón.
    """
    L: int = len(pattern)
    patt_z = np.nan_to_num(zscore(pattern))
    occ: Deque[int] = deque()

    for i in range(len(series) - L + 1):
        window = series[i : i + L]
        win_z = np.nan_to_num(zscore(window))
        if np.std(patt_z) == 0 or np.std(win_z) == 0:
            continue
        corr = np.corrcoef(patt_z, win_z)[0, 1]
        if corr >= threshold:
            occ.append(i)
    return occ


def fitness(series: np.ndarray, pattern: np.ndarray, threshold: float) -> float:
    """
    Evalúa la aptitud de un patrón combinando longitud y ocurrencias.

    Args:
        series (np.ndarray): Serie temporal.
        pattern (np.ndarray): Patrón codificado (posición 0: longitud).
        threshold (float): Umbral de correlación.

    Returns:
        float: Valor de aptitud = #ocurrencias + longitud.
    """
    L: int = int(pattern[0])
    coeffs: np.ndarray = pattern[1 : L + 1]
    occ_count = occurrences(series, coeffs, threshold)
    return float(occ_count + L)


def upgrade(
    ptc: particle, fvalue: float, gfitness: float) -> bool:
    """
    Actualiza la mejor solución personal (pbest) y determina si hay mejora global.

    Args:
        ptc (particle): Partícula evaluada.
        fvalue (float): Aptitud de la partícula.
        gfitness (float): Mejor aptitud global actual.

    Returns:
        bool: True si fvalue supera gfitness; False en caso contrario.
    """
    if fvalue > ptc._pfitness:
        ptc._pfitness = fvalue
        ptc._pbest = ptc._pattern.copy()

    return fvalue > gfitness


def pso(series: np.ndarray, max_length: int, min_length: int, 
        threshold: float, swarm_size: int, iterations: int, 
        omega: float, c1: float, c2: float
        ) -> Optional[np.ndarray]:
    """
    Ejecuta PSO para detectar el patrón óptimo en la serie temporal.

    Args:
        series (np.ndarray): Serie temporal de entrada.
        max_length (int): Longitud máxima de patrón.
        min_length (int): Longitud mínima de patrón.
        threshold (float): Umbral de correlación.
        swarm_size (int): Nº de partículas en la población.
        iterations (int): Nº de iteraciones del algoritmo.
        omega (float): Coeficiente de inercia.
        c1 (float): Coeficiente cognitivo.
        c2 (float): Coeficiente social.

    Returns:
        Optional[np.ndarray]: Patrón global (vector codificado) si se encuentra.
    """
    best_pattern: Optional[np.ndarray] = None
    best_fitness: float = -np.inf
    particles: List[particle] = [particle(min_length, max_length) for _ in range(swarm_size)]

    for _ in range(iterations):
        for p in particles:
            fval = fastfitness(series, p._pattern, threshold)
            if upgrade(p, fval, best_fitness):
                best_fitness = fval
                best_pattern = p._pattern.copy()

        # Movimiento de todas las partículas
        for p in particles:
            p.move(omega, c1, c2, best_pattern)

    return best_pattern


def merge(raw_indices: List[int], length: int, merge_thresh: int) -> Deque[List[int]]:
    """
    Filtra solapamientos y fusiona ocurrencias próximas.

    Args:
        raw_indices (List[int]): Índices iniciales sin procesar.
        length (int): Longitud del segmento patrón.
        merge_thresh (int): Distancia máxima para fusionar intervalos.

    Returns:
        Deque[List[int]]: Lista de pares [start, end] fundidos.
    """
    # Filtrado de solapamientos
    filtered: Deque[int] = deque()
    last_end = -1
    for idx in sorted(raw_indices):
        if idx > last_end:
            filtered.append(idx)
            last_end = idx + length - 1

    # Fusión de ocurrencias cercanas
    merged: Deque[List[int]] = deque()
    for start in filtered:
        end = start + length - 1
        if not merged:
            merged.append([start, end])
        else:
            prev_start, prev_end = merged[-1]
            if start - prev_end <= merge_thresh:
                merged[-1][1] = end
            else:
                merged.append([start, end])

    return merged
