"""Inyecta modismos (traducción por equivalencia, no literal) al afinamiento.

TED2020 y Europarl casi no enseñan "modismo -> modismo equivalente": enseñan
traducción general. Por eso el modelo tiende a traducir modismos palabra por
palabra (ver salidas/modismos_revision_antes.csv). Este módulo no depende de
PyTorch/Transformers, igual que muestreo.py, para que se pueda importar y
probar sin instalar el resto de dependencias.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

import config


def cargar_modismos(ruta: Path) -> list[dict]:
    """Lee un TSV con columnas id, de, en, es, ro (una fila por variante de
    un modismo, con su equivalente real -no literal- en cada idioma)."""
    with open(ruta, encoding="utf-8", newline="") as f:
        lector = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        columnas = {"id", *config.IDIOMAS}
        if not columnas.issubset(lector.fieldnames or []):
            raise ValueError(f"{ruta} no contiene las columnas requeridas: {sorted(columnas)}")
        filas = list(lector)
    if not filas:
        raise ValueError(f"{ruta} no tiene ninguna fila.")
    return filas


def expandir_pares_entrenamiento(filas: list[dict], repeticiones: int) -> pd.DataFrame:
    """De cada fila multi-paralela (un modismo en los 4 idiomas) arma pares
    (origen, destino, texto_origen, texto_destino) para las 6 combinaciones
    de config.PARES, repetidos `repeticiones` veces para que el muestreo por
    reservorio del resto del corpus (millones de pares) no los deje fuera.

    El factor de repetición es una perilla, no una verdad revelada: con muy
    pocas repeticiones el modelo casi no ve estos ejemplos: con demasiadas,
    puede sobreajustarse a estas 12 oraciones exactas en vez de generalizar.
    Vale la pena probar un par de valores (config.REPETICIONES_MODISMOS) y
    comparar salidas/modismos_revision_despues.csv entre corridas.
    """
    if repeticiones < 1:
        raise ValueError("repeticiones debe ser >= 1")
    registros = []
    for fila in filas:
        for par in config.PARES:
            a, b = par.split("-")
            registros.append(
                {"origen": a, "destino": b, "texto_origen": fila[a], "texto_destino": fila[b]}
            )
            registros.append(
                {"origen": b, "destino": a, "texto_origen": fila[b], "texto_destino": fila[a]}
            )
    return pd.DataFrame(registros * repeticiones)
