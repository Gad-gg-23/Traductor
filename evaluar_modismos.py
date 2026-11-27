"""Evalúa modismos específicamente: ¿el afinamiento con modismos.tsv sirvió?

    python Traductor.py evaluar-modismos

Dos conjuntos, guardados en salidas/:

1. Entrenados (data/modismos/modismos.tsv, el que SÍ se repite en el
   afinamiento -ver entrenar.py-): compara contra la versión "antes" que ya
   existe (modismos_directos_antes.csv / modismos_revision_antes.csv) para
   ver si mejoró. Esto mide memorización de estos modismos puntuales, no
   generalización.

2. Held-out (data/modismos/modismos_heldout.tsv, NUNCA usado en el
   afinamiento): mide si el modelo generaliza a modismos que nunca vio, no
   solo si memorizó los 12 de arriba. Es la prueba más honesta de las dos.
"""

import itertools

import pandas as pd
from tqdm import tqdm

import config
from evaluar import bleu_chrf, cargar_modelo, guardar_csv_atomico
from modismos import cargar_modismos


def evaluar_conjunto(filas, motor, descripcion):
    direcciones = list(itertools.permutations(config.IDIOMAS, 2))
    agregados, detalle = [], []
    for origen, destino in tqdm(direcciones, desc=f"  {descripcion}"):
        entradas = [fila[origen] for fila in filas]
        referencias = [fila[destino] for fila in filas]
        hipotesis = motor.traducir_lote(entradas, origen, destino)
        bleu, chrf = bleu_chrf(hipotesis, referencias)
        agregados.append({
            "origen": origen, "destino": destino, "n": len(filas),
            "bleu": round(bleu, 2), "chrf": round(chrf, 2),
        })
        for entrada, referencia, hipo in zip(entradas, referencias, hipotesis):
            detalle.append({
                "origen": origen, "destino": destino,
                "entrada": entrada, "referencia": referencia, "hipotesis": hipo,
            })
    return pd.DataFrame(agregados), pd.DataFrame(detalle)


def main():
    print("Cargando modelo afinado...")
    motor = cargar_modelo()

    print(f"\nModismos entrenados ({config.RUTA_MODISMOS.name}, 12 direcciones)...")
    filas = cargar_modismos(config.RUTA_MODISMOS)
    directos, revision = evaluar_conjunto(filas, motor, "entrenados")
    guardar_csv_atomico(directos, config.RUTA_SALIDAS / "modismos_directos_despues.csv")
    guardar_csv_atomico(revision, config.RUTA_SALIDAS / "modismos_revision_despues.csv")
    print(directos.to_string(index=False))

    print(f"\nModismos held-out ({config.RUTA_MODISMOS_HELDOUT.name}, nunca vistos en el afinamiento)...")
    filas_heldout = cargar_modismos(config.RUTA_MODISMOS_HELDOUT)
    directos_h, revision_h = evaluar_conjunto(filas_heldout, motor, "held-out")
    guardar_csv_atomico(directos_h, config.RUTA_SALIDAS / "modismos_heldout_directos.csv")
    guardar_csv_atomico(revision_h, config.RUTA_SALIDAS / "modismos_heldout_revision.csv")
    print(directos_h.to_string(index=False))

    print(
        "\nOK: compara modismos_directos_antes.csv vs modismos_directos_despues.csv "
        "(memorización) y revisa modismos_heldout_revision.csv a mano (generalización; "
        "BLEU/chrF no bastan para juzgar un modismo nunca visto)."
    )


if __name__ == "__main__":
    main()
