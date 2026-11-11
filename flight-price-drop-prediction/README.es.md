# Predicción de bajadas de precio en vuelos

Práctica 1 de Aprendizaje Automático a Gran Escala. Jaime Morales Castro y Nicolás Pernas Gómez.

Clasificación binaria con Spark MLlib: dada una búsqueda de vuelo, predice si el precio bajará al menos un 5 % en los siete días siguientes (ESPERAR) o no (COMPRAR YA).

## Requisitos

- Python 3.11 y `pip install -r requirements.txt`
- JDK 17
- Solo en Windows: `winutils.exe` y `hadoop.dll` de Hadoop 3.3.6 en `C:\hadoop\bin`, de https://github.com/cdarlint/winutils

`src/config.py` usa las variables `JAVA_HOME` y `HADOOP_HOME` del sistema si son válidas; si no, recurre a rutas fijas que habrá que ajustar. Para comprobar que Spark arranca:

```
python src/config.py
```

## Datos

No se incluyen: el fichero original ocupa 31 GB. Se descarga de Kaggle (*Expedia Flight Prices*) y se deja como `data/itineraries.csv`.

El subset utilizado son los vuelos entre {LAX, SFO} y {JFK, EWR, BOS} en ambos sentidos, 12 rutas dirigidas y 6.147.167 observaciones.

## Ejecución

```
python src/data_processing.py   # etiqueta, variables y particiones
python src/main.py              # entrena, evalúa y escribe results/
```

El primero construye `data/corredor.parquet` desde el CSV del subset si no existe, y genera `data/dataset_modelado.parquet` (3.145.954 filas). El segundo tarda entre 35 y 60 minutos; `python src/main.py --muestra 0.05` hace una pasada rápida de comprobación.

El CSV del subset se obtiene a partir de `itineraries.csv` ejecutando `notebooks/data_selection.ipynb`.

Opcional, genera las cifras de la sección de limitaciones de la memoria:

```
python src/analisis_supervivencia.py
```

## Estructura

```
src/config.py                   rutas, sesión de Spark y parámetros
src/data_processing.py          selección, etiquetado, variables y particiones
src/models.py                   pipelines, rejilla de hiperparámetros y entrenamiento
src/evaluation.py               métricas, umbral de decisión y simulación económica
src/main.py                     ejecución completa
src/analisis_supervivencia.py   cuantificación del sesgo de supervivencia
notebooks/data_selection.ipynb  selección del subset
notebooks/02_eda.ipynb          análisis exploratorio
results/                        métricas de la ejecución final
```
