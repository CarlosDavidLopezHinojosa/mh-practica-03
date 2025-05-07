# Usa una imagen base de Python
FROM python:3.10-slim

# Instala dependencias del sistema necesarias para Rust y OpenSSL
RUN apt-get update && apt-get install -y \
    curl build-essential gcc pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Instala Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Instala maturin
RUN pip install maturin

# Establece el directorio de trabajo
WORKDIR /app

# Copia archivos de Python y requisitos
COPY main/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo el proyecto
COPY main/ .

# Entra a la carpeta del módulo Rust y compila el paquete
WORKDIR /app/rust
RUN maturin build --release

# Instala el paquete generado
RUN pip install target/wheels/*.whl

# Vuelve a la raíz de la app
WORKDIR /app

# Copia los archivos CSV (opcional si ya están en COPY main/.)
COPY main/charts/ ./charts/

# Expone el puerto de FastAPI
EXPOSE 8000

# Comando para ejecutar el servidor
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
