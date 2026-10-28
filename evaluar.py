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

import pandas as pd
import sacrebleu
import torch
from tqdm import tqdm
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

import config


def cargar_modelo():
    if not config.RUTA_MODELO_DIR.exists():
        raise SystemExit(f"No hay modelo afinado en {config.RUTA_MODELO_DIR}. Corre primero: python entrenar.py")
    tokenizer = M2M100Tokenizer.from_pretrained(config.RUTA_MODELO_DIR)
    modelo = M2M100ForConditionalGeneration.from_pretrained(config.RUTA_MODELO_DIR)
    modelo.eval()
    if torch.cuda.is_available():
        modelo.to("cuda")
    return modelo, tokenizer


@torch.inference_mode()
def traducir_lote(textos, origen, destino, modelo, tokenizer):
    tokenizer.src_lang = origen
    salidas = []
    for i in range(0, len(textos), config.TAMANO_LOTE_GENERACION):
        trozo = textos[i:i + config.TAMANO_LOTE_GENERACION]
        entradas = tokenizer(trozo, return_tensors="pt", padding=True, truncation=True,
                             max_length=config.MAX_TOKENS).to(modelo.device)
        generados = modelo.generate(**entradas, forced_bos_token_id=tokenizer.get_lang_id(destino),
                                    max_length=config.LONGITUD_MAXIMA_GENERACION)
        salidas.extend(tokenizer.batch_decode(generados, skip_special_tokens=True))
    return salidas


def bleu_chrf(hipotesis, referencias):
    bleu = sacrebleu.corpus_bleu(hipotesis, [referencias])
    chrf = sacrebleu.corpus_chrf(hipotesis, [referencias])
    return bleu.score, chrf.score


def evaluar_directo(df_prueba, modelo, tokenizer):
    filas = []
    direcciones = list(itertools.permutations(config.IDIOMAS, 2))  # las 12, sin repetir origen=destino
    for origen, destino in tqdm(direcciones, desc="  direcciones"):
        hipotesis = traducir_lote(list(df_prueba[origen]), origen, destino, modelo, tokenizer)
        bleu, chrf = bleu_chrf(hipotesis, list(df_prueba[destino]))
        filas.append({"origen": origen, "destino": destino, "bleu": round(bleu, 2), "chrf": round(chrf, 2)})
    return pd.DataFrame(filas)


def evaluar_cadenas(df_prueba, modelo, tokenizer):
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
            actuales = traducir_lote(actuales, idioma_actual, siguiente, modelo, tokenizer)
            idioma_actual = siguiente
        bleu, chrf = bleu_chrf(actuales, originales)
        filas.append({"idioma_inicio": origen, "orden": "->".join(orden),
                      "bleu": round(bleu, 2), "chrf": round(chrf, 2)})
    return pd.DataFrame(filas)


def main():
    print("Cargando modelo afinado...")
    modelo, tokenizer = cargar_modelo()

    df_prueba = pd.read_csv(config.RUTA_PROCESADOS / "prueba.tsv", sep="\t",
                            quoting=csv.QUOTE_NONE, escapechar="\\", keep_default_na=False)

    print(f"\nEvaluación directa ({len(df_prueba):,} oraciones por dirección, 12 direcciones)...")
    directos = evaluar_directo(df_prueba, modelo, tokenizer)
    directos.to_csv(config.RUTA_RESULTADOS_DIRECTOS_CSV, index=False)
    print(directos.to_string(index=False))

    n_cadenas = min(config.N_ORACIONES_CADENAS, len(df_prueba))
    print(f"\nEvaluación en cadena ({n_cadenas} oraciones, 4 idiomas x 6 órdenes = 24 cadenas)...")
    cadenas = evaluar_cadenas(df_prueba, modelo, tokenizer)
    cadenas.to_csv(config.RUTA_RESULTADOS_CADENAS_CSV, index=False)
    print(cadenas.to_string(index=False))

    print("\nResumen por idioma de inicio (promedio de las 6 permutaciones):")
    print(cadenas.groupby("idioma_inicio")[["bleu", "chrf"]].mean().round(2).to_string())

    print(f"\nOK: resultados en {config.RUTA_RESULTADOS_DIRECTOS_CSV} y {config.RUTA_RESULTADOS_CADENAS_CSV}")


if __name__ == "__main__":
    main()
