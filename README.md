# Metaheurísticas: Práctica 3

Este proyecto es una aplicación web para la visualización y análisis de series temporales, desarrollada como parte de la asignatura de Metaheurísticas.

## Características

- **Carga de CSV**: Permite cargar archivos CSV para visualizar datos.
- **Selección de datasets**: Incluye datasets predefinidos que se pueden seleccionar desde la interfaz.
- **Visualización gráfica**: Utiliza gráficos de líneas para representar las series temporales.
- **Interactividad**: Ajuste dinámico del rango de visualización en los gráficos.

## Tecnologías utilizadas

- **Frontend**: React, TypeScript, Vite, TailwindCSS.
- **Backend**: FastAPI.
- **Gráficos**: Recharts.

## Requisitos previos

- Node.js (versión 18 o superior).
- Python (versión 3.8 o superior).
- Gestor de paquetes `pnpm` para el frontend.

## Instalación

1. Clona este repositorio:
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd metaheuristicas/mh-practica-03
   ```

2. Configura el backend:
   ```bash
   cd main
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   pip install -r requirements.txt
   uvicorn src.api:app --reload
   ```

3. Configura el frontend:
   ```bash
   cd ../web
   pnpm install
   pnpm dev
   ```

## Uso

1. Inicia el backend con `uvicorn` en el puerto 8000.
2. Inicia el frontend con `pnpm dev` en el puerto 5173.
3. Abre el navegador en `http://localhost:5173` para acceder a la aplicación.

## Estructura del proyecto

- **main**: Contiene el backend desarrollado con FastAPI.
- **web**: Contiene el frontend desarrollado con React y TypeScript.
- **docs**: Documentación adicional del proyecto.

## Autores

- Javier Gómez Aparicio
- Carlos David López Hinojosa
- Alejandro Luque Núñez

Profesor: José María Luna Ariza

## Licencia

Este proyecto es parte de un curso académico y no tiene licencia explícita para distribución pública.
