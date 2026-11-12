"""Muestreo de datos sin dependencias de PyTorch/Transformers."""

import csv
import random

import pandas as pd

import config


def muestrear_archivo_direcciones(ruta, n_por_direccion, semilla):
    """Muestreo de reservorio: evita cargar el TSV de ~1.1 GB en RAM."""
    muestras = {tuple(par.split("-")): [] for par in config.PARES}
    vistos = {par: 0 for par in muestras}
    generadores = {
        par: random.Random(f"{semilla}-{par[0]}-{par[1]}") for par in muestras
    }
    with open(ruta, encoding="utf-8", newline="") as f:
        lector = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        columnas = {"idioma_a", "idioma_b", "texto_a", "texto_b"}
        if not columnas.issubset(lector.fieldnames or []):
            raise ValueError(f"{ruta} no contiene las columnas requeridas: {sorted(columnas)}")
        for fila in lector:
            par = (fila["idioma_a"], fila["idioma_b"])
            if par not in muestras:
                raise ValueError(f"Par inesperado en entrenamiento.tsv: {par}")
            vistos[par] += 1
            muestra = muestras[par]
            registro = (fila["texto_a"], fila["texto_b"])
            if len(muestra) < n_por_direccion:
                muestra.append(registro)
            else:
                indice = generadores[par].randrange(vistos[par])
                if indice < n_por_direccion:
                    muestra[indice] = registro

    faltantes = [
        f"{a}-{b}: {vistos[(a, b)]}"
        for a, b in muestras
        if len(muestras[(a, b)]) < n_por_direccion
    ]
    if faltantes:
        raise ValueError("No hay suficientes ejemplos por dirección: " + ", ".join(faltantes))
    filas = []
    for (a, b), muestra in muestras.items():
        for texto_a, texto_b in muestra:
            filas.append(
                {"origen": a, "destino": b, "texto_origen": texto_a, "texto_destino": texto_b}
            )
            filas.append(
                {"origen": b, "destino": a, "texto_origen": texto_b, "texto_destino": texto_a}
            )
    return pd.DataFrame(filas)
