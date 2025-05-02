import polars as pl
import json

from os import path
from functools import wraps

from typing import Union

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