import polars as pl
import json

import time
import tracemalloc as tm
from os import path
from functools import wraps

from typing import Union


def instant() -> float:
    """
    Devuelve el tiempo actual en segundos desde la época (epoch).
    
    Returns:
        float: Tiempo actual en segundos.
    """
    return time.perf_counter()

def memory() -> float:
    """
    Devuelve el uso máximo de memoria en bytes.
    
    Returns:
        float: Uso máximo de memoria en bytes.
    """
    return tm.get_traced_memory()[0]

def memstart():
    """
    Inicia el rastreo de memoria.
    """
    tm.start()

def memstop():
    """
    Detiene el rastreo de memoria.
    """
    tm.stop()

def measure(func):
    """
    Decorador para medir el tiempo de ejecución y el uso de memoria de una función.
    
    Args:
        func (function): Función a medir.
    
    Returns:
        function: Función decorada que devuelve un diccionario con:
                  - 'solution': Resultado de la función original.
                  - 'time': Tiempo de ejecución en segundos.
                  - 'memory': Pico de memoria utilizado en bytes.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        memstart()  # Inicia el rastreo de memoria
        start_time = time.perf_counter()  # Tiempo de inicio
        result = {}
        try:
            result.update(func(*args, **kwargs))  # Ejecuta la función
        finally:
            end_time = time.perf_counter()  # Tiempo de fin
            peak = tm.get_traced_memory()[1]  # Obtiene el uso máximo de memoria
            memstop()  # Detiene el rastreo de memoria
        result.update({'time': end_time - start_time, 'memory': peak})
        return result
    return wrapper

def save(data, filename):
    """
    Guarda datos en un archivo utilizando pickle.
    
    Args:
        data (any): Datos a guardar.
        filename (str): Nombre del archivo donde se guardarán los datos.
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def load(filename):
    """
    Carga datos desde un archivo utilizando pickle.
    
    Args:
        filename (str): Nombre del archivo desde donde se cargarán los datos.
    
    Returns:
        any: Datos cargados desde el archivo.
    """
    with open(filename, 'r') as f:
        return json.load(f)


def load_temporal_series(path: Union[str, bytes]):
    """
    Carga datos desde un archivo utilizando pickle.
    
    Args:
        path (str): Nombre del archivo desde donde se cargarán los datos.
    
    Returns:
        any: Datos cargados desde el archivo.
    """
    df = pl.read_csv(path)
    index = df.columns[0] if len(df.columns) == 1 else df.columns[1]
    return df[index].to_list()

def cwd():
    return path.dirname(__file__)