# Usa una imagen oficial de Python
FROM python:3.10-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos del backend
COPY main/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main/ .

# Copia los archivos CSV
COPY main/charts/ ./charts/


# Expone el puerto usado por Uvicorn
EXPOSE 8000

# Comando para ejecutar FastAPI con autoreload (puede cambiarse en prod)
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
