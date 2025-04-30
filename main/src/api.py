from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import polars as pl
from os import path

app = FastAPI()
ROOT_DIR = path.dirname(__file__).split("main")[0]

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Cambia esto si tu cliente está en otro dominio o puerto
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/datasets")
async def get_datasets():
    """
    Devuelve una lista de datasets disponibles.
    """
    print(ROOT_DIR)
    names = ["ECG Real", 'ECG Real con anomalías', 'ECG Sintético', 'Senoidal', 'Senoidal con ruido']
    paths = [
        ROOT_DIR + "charts/ecg-real.csv",
        ROOT_DIR + "charts/ecg-real-con-anomalia.csv",
        ROOT_DIR + "charts/ecg-sintetico.csv",
        ROOT_DIR + "charts/Seno-Different-Factors.csv",
        ROOT_DIR + "charts/Synthetic-three-patterns-with-noise.csv"
    ]

    datasets = {}

    for name, path in zip(names, paths):
        try:
            # Leer el archivo CSV
            df = pl.read_csv(path)
            # Obtener el primer índice de la columna
            index = df.columns[0] if len(df.columns) == 1 else df.columns[1]
            # Convertir la columna a una lista
            data = df[index].to_list()
            datasets[name] = data
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
        df = pl.read_csv(file.file)
        index = df.columns[0] if len(df.columns) == 1 else df.columns[1]
        return {"data": df[index].to_list()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar el fichero: {str(e)}")