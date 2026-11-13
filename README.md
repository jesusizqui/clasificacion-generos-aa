# ¿Qué hace que una canción suene a rock?

Clasificación de géneros musicales del Free Music Archive (FMA) mediante regresión logística, k vecinos más próximos y algoritmos genéticos.

Proyecto final de **Aprendizaje Automático** · Grado en Ingeniería Matemática · Universidad Francisco de Vitoria · Curso 2025-2026.

## Estructura

```
proyecto/
├── datos/            # aquí se descomprime fma_metadata (no se sube al repositorio)
├── src/
│   ├── datos.py                 # carga del subconjunto small y estandarización
│   └── regresion_logistica.py   # regresión logística desde cero (Newton-Raphson + ridge)
├── notebooks/
│   ├── 01_datos_y_exploracion.ipynb         # comprobaciones, EDA y estandarización
│   ├── 02_regresion_logistica_binaria.ipynb  # ejemplo de clase, Rock vs Hip-Hop, comparación con scikit-learn
│   ├── 03_regularizacion_y_multiclase.ipynb  # 8 géneros (uno contra el resto) y curva frente a λ
│   └── 04_knn.ipynb                          # k-NN y comparación preliminar
├── figuras/          # las figuras que generan los notebooks
└── resultados/       # tablas y resultados en CSV/JSON
```

## Cómo reproducir los resultados

1. Instalar las dependencias:
   ```
   pip install -r requirements.txt
   ```
2. Descargar `fma_metadata.zip` desde el enlace del repositorio oficial (https://github.com/mdeff/fma) y descomprimirlo dentro de `datos/`, de forma que exista `datos/fma_metadata/tracks.csv`.
3. Ejecutar los notebooks **en orden** (01 → 04) desde la carpeta `notebooks/`. El notebook 01 genera `datos/procesados.npz`, que usan los demás.

Ningún paso depende del azar, así que los resultados son reproducibles.

## Datos

FMA: A Dataset for Music Analysis (Defferrard, Benzi, Vandergheynst y Bresson, ISMIR 2017). Solo se usan los metadatos y las características precalculadas (licencia CC BY 4.0); no se descarga ni se redistribuye audio.
