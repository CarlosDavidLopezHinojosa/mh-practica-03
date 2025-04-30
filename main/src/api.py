from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from src.tools import utils
from src.algorithm import pattern


app = FastAPI()
ROOT_DIR = utils.cwd().split("src")[0]

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambia esto si tu cliente está en otro dominio o puerto
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