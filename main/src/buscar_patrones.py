from tools.utils import load_temporal_series

def buscar_patrones_valores(signal: list, min_win: int = 2, max_win: int = 10, tolerance: float = 0.1) -> dict:
    """
    Busca patrones en forma de ventanas de valores de una señal. Al usar los valores, tiene en cuenta la posición y de los patrones.

    Args:
        signal (list): señal en la que buscar patrones.
        min_win (int): tamaño mínimo de ventana.
        max_win (int): tamaño máximo de ventana.
        tolerance (float): tolerancia a la hora de comparar si una ventana coincide con un patrón.
    
    Returns:
        out (dictionary): diccionario que contiene pares patrón(tupla) - número de veces que se ha encontrado, ordenado de mayor a menor número de veces encontrado.
    """

    out = {}

    for size in range(min_win, max_win, 1):
        for x in range(len(signal)-size):
            window = []
            for i in range(size):
                window.append(signal[x+i])
            found = False
            for pattern in out:
                if len(pattern) == len(window):
                    match = True
                    for j in range(len(pattern)):
                        if window[i] > pattern[i] + (pattern[i] * tolerance) or window[i] < pattern[i] - (pattern[i] * tolerance):
                           match = False 
                           break
                    if match == True:
                        out[pattern] += 1
                        found = True
                        break
            if found == False:
                out[tuple(window)] = 1
    
    sorted_out = {}

    for key in sorted(out, key=out.get, reverse=True):
        sorted_out[key] = out[key]

    return sorted_out


def buscar_patrones_pendientes(signal: list, min_win: int = 2, max_win: int = 10, tolerance: float = 0.1) -> dict:
    """
    Busca patrones en forma de ventanas de pendientes de una señal. Al usar las pendientes, los patrones son independientes de la posición y del patrón.

    Args:
        signal (list): señal en la que buscar patrones.
        min_win (int): tamaño mínimo de ventana.
        max_win (int): tamaño máximo de ventana.
        tolerance (float): tolerancia a la hora de comparar si una ventana coincide con un patrón.
    
    Returns:
        out (dictionary): diccionario que contiene pares patrón(tupla) - número de veces que se ha encontrado, ordenado de mayor a menor número de veces encontrado.
    """

    out = {}

    for size in range(min_win, max_win, 1):
        for x in range(len(signal)-size):
            window = []
            for i in range(size-1):
                window.append(signal[x+i+1]-signal[x+i])
            found = False
            for pattern in out:
                if len(pattern) == len(window):
                    match = True
                    for j in range(len(pattern)):
                        if window[i] > pattern[i] + (pattern[i] * tolerance) or window[i] < pattern[i] - (pattern[i] * tolerance):
                           match = False 
                           break
                    if match == True:
                        out[pattern] += 1
                        found = True
                        break
            if found == False:
                out[tuple(window)] = 1
    
    sorted_out = {}

    for key in sorted(out, key=out.get, reverse=True):
        sorted_out[key] = out[key]

    return sorted_out

if __name__ == "__main__":
    signal = load_temporal_series("main/charts/ecg-sintetico.csv")
    out_valores: dict = buscar_patrones_valores(signal)
    out_pendientes: dict = buscar_patrones_pendientes(signal)
    pass 