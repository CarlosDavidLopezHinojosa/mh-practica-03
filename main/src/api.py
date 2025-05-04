from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.tools import utils
from src.algorithm import pso


app = FastAPI()
MAINDIR = utils.cwd().split("src")[0]

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a Pydantic model for the request body
class PatternRequest(BaseModel):
    temporal_series: list[float]
    max_lenght: int
    min_lenght: int
    threshold: float
    swarm_size: int
    iterations: int
    omega: float
    c1: float
    c2: float
    merge_thresh: float


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/datasets")
async def get_datasets():
    """
    Devuelve una lista de datasets disponibles.
    """
    names = ["ECG Real", 'ECG Real con anomalías', 'ECG Sintético', 'Senoidal', 'Sintético de tres patrones']
    paths = [
        MAINDIR + "charts/ecg-real.csv",
        MAINDIR + "charts/ecg-real-con-anomalia.csv",
        MAINDIR + "charts/ecg-sintetico.csv",
        MAINDIR + "charts/Seno-Different-Factors.csv",
        MAINDIR + "charts/Synthetic-three-patterns-with-noise.csv"
    ]

    datasets = {}

    for name, path in zip(names, paths):
        try:
            datasets[name] = utils.load_temporal_series(path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al procesar el fichero: {str(e)}")
        
    return datasets

@app.post("/dataframe")
async def parse_csv(file: UploadFile):
    """
    Parsea un fichero CSV y devuelve el contenido como una lista.
    """
    try:
        # Leer el archivo CSV desde el contenido del archivo
        data = utils.load_temporal_series(file.file)
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar el fichero: {str(e)}")
    
@app.post("/pattern")
async def find_pattern(request: PatternRequest):
    """
    Encuentra patrones en una serie temporal utilizando el algoritmo PSO.
    """
    try:
        temporal_series = request.temporal_series

        # Validar que la serie temporal no esté vacía
        if not temporal_series:
            raise HTTPException(status_code=400, detail="La serie temporal está vacía.")

        # Parámetros del algoritmo
        max_lenght = request.max_lenght
        min_lenght = request.min_lenght
        threshold = request.threshold
        swarm_size = request.swarm_size
        iterations = request.iterations
        omega = request.omega
        c1 = request.c1
        c2 = request.c2

        # Ejecutar el algoritmo PSO para encontrar patrones
        best_pattern = pso.pso(temporal_series, max_lenght, min_lenght, threshold, swarm_size, iterations, omega, c1, c2)
        L = int(best_pattern[0])
        coeffs = best_pattern[1:L+1]
        raw_occ = pso.find_occurrences(temporal_series, coeffs, threshold)
        merge_thresh = 2
        best_pattern = pso.filter_and_merge_occurrences(raw_occ, L, merge_thresh)
        return {"best_pattern": best_pattern}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la serie temporal: {str(e)}")