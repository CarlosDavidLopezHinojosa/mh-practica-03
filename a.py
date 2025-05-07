import numpy as np
import time
from occ import fastfitness
from main.src.algorithm.pso import occurrences
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


import polars as pl

def load_temporal_series(path):
    """
    Carga datos desde un archivo utilizando pickle.
    
    Args:
        path (str): Nombre del archivo desde donde se cargarán los datos.
    
    Returns:
        any: Datos cargados desde el archivo.
    """
    df = pl.read_csv(path)
    index = df.columns[0] if len(df.columns) == 1 else df.columns[1]
    return df[index].to_numpy()

def compare_fastfitness_and_occurrences():
    sizes = [10_000]
    threshold = 0.8
    pattern_length = 2
    np.random.seed(42)
    pattern = np.random.rand(pattern_length).astype(np.float64)

    S = load_temporal_series("main/charts/ecg-sintetico.csv")
    for size in sizes:
        # Accuracy comparison
        python_result = occurrences(S, pattern, threshold)
        rust_result = fastfitness(S, np.concatenate(([pattern_length], pattern)), threshold)

        print(f"Python result: {python_result}")
        print(f"Rust result: {rust_result}")

        # assert np.isclose(rust_result, python_result, atol=1e-6), \
        #     f"Accuracy mismatch: fastfitness={rust_result}, occurrences={python_result}"

        # Performance comparison
        start = time.time()
        occurrences(S, pattern, threshold)
        python_time = time.time() - start

        start = time.time()
        fastfitness(S, np.concatenate(([pattern_length], pattern)), threshold)
        rust_time = time.time() - start

        print(f"Size: {size}")
        print(f"occurrences (Python): {python_time:.6f}s")
        print(f"fastfitness (Rust): {rust_time:.6f}s")
        print("-" * 40)

if __name__ == "__main__":
    compare_fastfitness_and_occurrences()

