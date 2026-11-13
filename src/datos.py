"""
Funciones para cargar el dataset FMA y preparar los datos.

Aquí se agrupa todo lo relacionado con los datos para no repetir el mismo
código en cada notebook.
"""
import numpy as np
import pandas as pd

# Nombres que usa FMA para la partición oficial y nombres cortos que usaremos nosotros
NOMBRES_PARTICION = {"training": "train", "validation": "val", "test": "test"}


def cargar_fma_small(carpeta):
    """
    Lee tracks.csv y features.csv y se queda solo con las 8.000 canciones
    del subconjunto 'small'.

    Devuelve:
        X         -> tabla (DataFrame) de 8.000 filas x 518 características
        genero    -> género de cada canción (lo que queremos predecir)
        particion -> 'train', 'val' o 'test' según la partición oficial
        artista   -> identificador del artista (para comprobar la partición)
    """
    # tracks.csv tiene dos filas de cabecera (p. ej. 'track' / 'genre_top'),
    # por eso usamos header=[0, 1]. La primera columna es el id de la canción.
    tracks = pd.read_csv(f"{carpeta}/tracks.csv", index_col=0, header=[0, 1], low_memory=False)

    # features.csv tiene tres filas de cabecera: característica, estadístico y número
    # (p. ej. 'mfcc' / 'mean' / '01').
    features = pd.read_csv(f"{carpeta}/features.csv", index_col=0, header=[0, 1, 2])

    # Nos quedamos solo con las canciones del subconjunto 'small'
    es_small = tracks[("set", "subset")] == "small"
    tracks_small = tracks[es_small]
    X = features.loc[tracks_small.index].copy()

    # Convertimos la cabecera de tres niveles en un único nombre legible,
    # por ejemplo ('mfcc', 'mean', '01') -> 'mfcc_mean_01'
    X.columns = ["_".join(col) for col in X.columns]

    genero = tracks_small[("track", "genre_top")].astype(str)
    particion = tracks_small[("set", "split")].map(NOMBRES_PARTICION)
    artista = tracks_small[("artist", "id")]

    # Ponemos nombre a las series para que las tablas se lean mejor
    genero.name, particion.name, artista.name = "genero", "particion", "artista"
    return X, genero, particion, artista


class Estandarizador:
    """
    Pasa cada característica a media 0 y desviación típica 1.

    La media y la desviación se calculan SOLO con el conjunto de entrenamiento
    y luego se aplican igual a validación y test. Si usáramos también los datos
    de test para calcularlas, el modelo "vería" información del test antes de
    ser evaluado y el resultado sería demasiado optimista.
    """

    def ajustar(self, X_train):
        self.media = X_train.mean(axis=0)
        self.desviacion = X_train.std(axis=0)
        # Si alguna característica fuera constante, su desviación sería 0 y
        # dividiríamos entre 0. En ese caso la dejamos en 1 (no cambia nada).
        self.desviacion[self.desviacion == 0] = 1.0
        return self

    def transformar(self, X):
        return (X - self.media) / self.desviacion


def guardar_procesados(ruta, **datos):
    """Guarda varios arrays de NumPy en un único archivo .npz."""
    np.savez_compressed(ruta, **datos)


def cargar_procesados(ruta):
    """
    Carga el archivo generado en el notebook 01 y lo devuelve como
    diccionario: X_train, y_train, X_val, y_val, X_test, y_test, nombres, generos.
    """
    archivo = np.load(ruta, allow_pickle=True)
    return {clave: archivo[clave] for clave in archivo.files}
