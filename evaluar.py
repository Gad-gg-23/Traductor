"""
Evalúa el modelo afinado sobre prueba.tsv (2,000 oraciones multi-paralelas,
que el afinamiento nunca vio -- ver preparar_datos.py).

    python evaluar.py

Dos evaluaciones, guardadas en salidas/:

1. Directa (resultados_directos.csv): las 12 direcciones de traducción,
   comparando contra la traducción real de TED2020 con BLEU y chrF (las dos
   métricas estándar de traducción automática; sacrebleu las calcula igual
   en cualquier computadora, sin ambigüedad de tokenización).

2. En cadena (resultados_cadenas.csv): la tarea real de la práctica. Desde
   cada uno de los 4 idiomas de inicio se prueban las 6 permutaciones
   posibles de los otros 3 (4 saltos de traducción por cadena) y se compara
   el texto final contra el ORIGINAL -- no hace falta traducción humana de
   referencia, es la misma oración de donde salió la cadena.
"""

import csv
import itertools
import json
import os

import pandas as pd
import sacrebleu
from tqdm import tqdm

import config
from motor_traduccion import MotorTraduccion


def cargar_modelo():
    motor = MotorTraduccion(permitir_modelo_base=False)
    motor.cargar()
    return motor


def traducir_lote(textos, origen, destino, motor):
    return motor.traducir_lote(textos, origen, destino)


def bleu_chrf(hipotesis, referencias):
    bleu = sacrebleu.corpus_bleu(hipotesis, [referencias])
    chrf = sacrebleu.corpus_chrf(hipotesis, [referencias])
    return bleu.score, chrf.score


def evaluar_directo(df_prueba, motor):
    filas = []
    direcciones = list(itertools.permutations(config.IDIOMAS, 2))  # las 12, sin repetir origen=destino
    for origen, destino in tqdm(direcciones, desc="  direcciones"):
        hipotesis = traducir_lote(list(df_prueba[origen]), origen, destino, motor)
        bleu, chrf = bleu_chrf(hipotesis, list(df_prueba[destino]))
        filas.append({"origen": origen, "destino": destino, "bleu": round(bleu, 2), "chrf": round(chrf, 2)})
    return pd.DataFrame(filas)


def evaluar_cadenas(df_prueba, motor):
    muestra = df_prueba.sample(n=min(config.N_ORACIONES_CADENAS, len(df_prueba)), random_state=config.SEMILLA)
    filas = []
    combinaciones = [
        (origen, orden)
        for origen in config.IDIOMAS
        for orden in itertools.permutations(i for i in config.IDIOMAS if i != origen)
    ]
    for origen, orden in tqdm(combinaciones, desc="  cadenas"):
        originales = list(muestra[origen])
        actuales, idioma_actual = originales, origen
        for siguiente in list(orden) + [origen]:
            actuales = traducir_lote(actuales, idioma_actual, siguiente, motor)
            idioma_actual = siguiente
        bleu, chrf = bleu_chrf(actuales, originales)
        exactas = 100 * sum(a.casefold().strip() == b.casefold().strip() for a, b in zip(actuales, originales)) / len(originales)
        filas.append({"idioma_inicio": origen, "orden": "->".join(orden),
                      "bleu": round(bleu, 2), "chrf": round(chrf, 2),
                      "coincidencia_exacta_pct": round(exactas, 2)})
    return pd.DataFrame(filas)


def guardar_csv_atomico(df, ruta):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(ruta.suffix + ".part")
    df.to_csv(temporal, index=False)
    os.replace(temporal, ruta)


def main():
    print("Cargando modelo afinado...")
    motor = cargar_modelo()

    df_prueba = pd.read_csv(config.RUTA_PROCESADOS / "prueba.tsv", sep="\t",
                            quoting=csv.QUOTE_NONE, escapechar="\\", keep_default_na=False)
    faltantes = set(config.IDIOMAS) - set(df_prueba.columns)
    if faltantes:
        raise ValueError("prueba.tsv no contiene las columnas: " + ", ".join(sorted(faltantes)))
    if df_prueba.empty:
        raise ValueError("prueba.tsv está vacío.")

    print(f"\nEvaluación directa ({len(df_prueba):,} oraciones por dirección, 12 direcciones)...")
    directos = evaluar_directo(df_prueba, motor)
    guardar_csv_atomico(directos, config.RUTA_RESULTADOS_DIRECTOS_CSV)
    print(directos.to_string(index=False))

    n_cadenas = min(config.N_ORACIONES_CADENAS, len(df_prueba))
    print(f"\nEvaluación en cadena ({n_cadenas} oraciones, 4 idiomas x 6 órdenes = 24 cadenas)...")
    cadenas = evaluar_cadenas(df_prueba, motor)
    guardar_csv_atomico(cadenas, config.RUTA_RESULTADOS_CADENAS_CSV)
    print(cadenas.to_string(index=False))

    print("\nResumen por idioma de inicio (promedio de las 6 permutaciones):")
    print(cadenas.groupby("idioma_inicio")[["bleu", "chrf", "coincidencia_exacta_pct"]].mean().round(2).to_string())

    ruta_metadata = config.RUTA_SALIDAS / "metadata_evaluacion.json"
    temporal = ruta_metadata.with_suffix(".json.part")
    with open(temporal, "w", encoding="utf-8") as f:
        json.dump(
            {
                "modelo": motor.informacion.como_dict(),
                "oraciones_directas": len(df_prueba),
                "oraciones_cadenas": n_cadenas,
                "semilla": config.SEMILLA,
                "perfil_generacion": config.PERFIL_GENERACION_PREDETERMINADO,
                "nota": "BLEU/chrF miden coincidencia textual; deben complementarse con revisión humana.",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    os.replace(temporal, ruta_metadata)

    print(f"\nOK: resultados en {config.RUTA_RESULTADOS_DIRECTOS_CSV} y {config.RUTA_RESULTADOS_CADENAS_CSV}")


if __name__ == "__main__":
    main()
