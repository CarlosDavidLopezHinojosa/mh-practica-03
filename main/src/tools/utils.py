import polars as pl

from os import path
from typing import Union


def load_temporal_series(path: Union[str, bytes], numpy: bool = False):
    """
    Carga datos desde un archivo utilizando pickle.
    
    Args:
        path (str): Nombre del archivo desde donde se cargarán los datos.
        numpy (bool): Si se debe devolver un array de numpy o una lista. Por defecto es False.
    
    Returns:
        any: Datos cargados desde el archivo.
    """
    df = pl.read_csv(path)
    index = df.columns[0] if len(df.columns) == 1 else df.columns[1]
    return df[index].to_numpy() if numpy else df[index].to_list()

def cwd():
    return path.dirname(__file__)