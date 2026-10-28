"""
Limpia los corpus de data/crudos/ y arma los conjuntos de entrenamiento,
validación y prueba en data/procesados/.

    python preparar_datos.py

Salidas:
    entrenamiento.tsv  columnas: idioma_a, idioma_b, texto_a, texto_b   (NO va a Git, pesa cientos de MB)
    validacion.tsv     columnas: de, en, es, ro   (la MISMA oración en los 4 idiomas)
    prueba.tsv         columnas: de, en, es, ro
    resumen.json       conteos por corpus/par, para el reporte

validacion.tsv y prueba.tsv sí van a Git: así todo el equipo evalúa contra
exactamente las mismas oraciones aunque cada quien regenere el entrenamiento.

Por qué validación/prueba son multi-paralelas: la práctica es una cadena
es -> en -> ro -> de -> es. Para medir cuánto se pierde en cada salto hace
falta tener la traducción "real" de cada paso, no solo la del primer par.
TED2020 permite armarlas usando el inglés como pivote: si la misma oración en
inglés aparece en en-de, en-es y en-ro, se juntan sus tres traducciones.

Fuga de datos: cualquier oración que esté en validación/prueba (en cualquier
idioma) se elimina de TODOS los pares de entrenamiento. Si no, el modelo
podría haber visto "Hola, ¿cómo estás?" <-> "Salut, ce faci?" entrenando
es-ro y luego "acertarla" en la prueba por memoria.
"""

import csv
import json
import random
import re
import unicodedata
from collections import Counter

import config

ESPACIOS = re.compile(r"\s+")

# Para validación/prueba se piden oraciones "de verdad": ni "Gracias." ni
# párrafos enteros. El filtro de entrenamiento es más permisivo.
MIN_PALABRAS_EVAL = 4
MAX_PALABRAS_EVAL = 40
# Entre las 4 versiones de la misma oración. Más estricto que MAX_RATIO_LONGITUD
# porque TED a veces corta los subtítulos distinto en cada idioma (ej. el inglés
# trae "(Laughter) Graduated high school, ..." y el español empieza a la mitad).
MAX_RATIO_EVAL = 1.6
ANOTACIONES = re.compile(r"[()\[\]♫♪]")  # (Risas), (Aplausos), [música], letras de canciones


def normalizar(texto):
    texto = unicodedata.normalize("NFC", texto)
    return ESPACIOS.sub(" ", texto).strip()  # también quita tabs, que romperían el TSV


def clave(texto):
    """Forma comparable de una oración, para detectar duplicados y fugas."""
    return texto.casefold()


def par_valido(a, b, min_palabras=config.MIN_PALABRAS, max_palabras=config.MAX_PALABRAS):
    if not a or not b:
        return False
    na, nb = len(a.split()), len(b.split())
    if not (min_palabras <= na <= max_palabras and min_palabras <= nb <= max_palabras):
        return False
    if max(na, nb) / min(na, nb) > config.MAX_RATIO_LONGITUD:
        return False
    if clave(a) == clave(b):  # el "traductor" humano dejó la oración sin traducir
        return False
    return True


def leer_par(corpus, par):
    a, b = par.split("-")
    carpeta = config.RUTA_CRUDOS / corpus / par
    with open(carpeta / f"{corpus}.{par}.{a}", encoding="utf-8") as fa, \
         open(carpeta / f"{corpus}.{par}.{b}", encoding="utf-8") as fb:
        for la, lb in zip(fa, fb):
            yield normalizar(la), normalizar(lb)


# =========================================================
# VALIDACIÓN Y PRUEBA MULTI-PARALELAS (pivote inglés en TED2020)
# =========================================================

def traducciones_unicas_desde_ingles(idioma):
    """Dict ingles -> traducción, solo para oraciones en inglés que aparecen UNA vez.

    Si "Thank you." aparece 500 veces con traducciones distintas, no hay forma
    de saber cuál corresponde a cuál en los otros pares, así que se descarta.
    """
    par = "-".join(sorted(["en", idioma]))
    pos_en = par.split("-").index("en")
    vistos = Counter()
    traduccion = {}
    for a, b in leer_par("TED2020", par):
        en, otro = (a, b) if pos_en == 0 else (b, a)
        if not par_valido(en, otro, MIN_PALABRAS_EVAL, MAX_PALABRAS_EVAL):
            continue
        vistos[en] += 1
        traduccion[en] = otro
    return {en: t for en, t in traduccion.items() if vistos[en] == 1}


def construir_eval():
    otros = [i for i in config.IDIOMAS if i != "en"]
    por_idioma = {i: traducciones_unicas_desde_ingles(i) for i in otros}

    comunes = set.intersection(*(set(d) for d in por_idioma.values()))
    filas = []
    for en in sorted(comunes):  # sorted para que el resultado no dependa del orden del set
        fila = {"en": en, **{i: por_idioma[i][en] for i in otros}}
        # Las 4 versiones deben ser distintas entre sí (si no, alguna quedó sin traducir)
        if len({clave(t) for t in fila.values()}) != len(config.IDIOMAS):
            continue
        if any(ANOTACIONES.search(t) for t in fila.values()):
            continue
        largos = [len(t.split()) for t in fila.values()]
        if max(largos) / min(largos) > MAX_RATIO_EVAL:
            continue
        filas.append(fila)

    random.Random(config.SEMILLA).shuffle(filas)
    n_val, n_test = config.N_VALIDACION, config.N_PRUEBA
    if len(filas) < n_val + n_test:
        raise RuntimeError(f"Solo hay {len(filas)} oraciones multi-paralelas, se piden {n_val + n_test}")
    return filas[:n_val], filas[n_val:n_val + n_test], len(filas)


def guardar_eval(filas, ruta):
    with open(ruta, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=config.IDIOMAS, delimiter="\t",
                           quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writeheader()
        w.writerows({i: fila[i] for i in config.IDIOMAS} for fila in filas)


# =========================================================
# ENTRENAMIENTO
# =========================================================

def construir_entrenamiento(prohibidas, ruta_salida):
    resumen = {}
    total = 0
    with open(ruta_salida, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writerow(["idioma_a", "idioma_b", "texto_a", "texto_b"])

        for corpus in config.CORPUS:
            for par in config.PARES:
                a, b = par.split("-")
                crudos = descartados_fuga = 0
                vistos = set()
                pares = []
                for ta, tb in leer_par(corpus, par):
                    crudos += 1
                    if not par_valido(ta, tb):
                        continue
                    ka, kb = clave(ta), clave(tb)
                    if ka in prohibidas or kb in prohibidas:
                        descartados_fuga += 1
                        continue
                    if (ka, kb) in vistos:
                        continue
                    vistos.add((ka, kb))
                    pares.append((ta, tb))

                limpios = len(pares)
                random.Random(f"{config.SEMILLA}-{corpus}-{par}").shuffle(pares)
                pares = pares[:config.MAX_PARES_POR_CORPUS_Y_PAR]
                w.writerows((a, b, ta, tb) for ta, tb in pares)

                resumen[f"{corpus} {par}"] = {
                    "crudos": crudos, "limpios": limpios,
                    "descartados_por_fuga": descartados_fuga, "usados": len(pares),
                }
                total += len(pares)
                print(f"  {corpus:9s} {par}: {crudos:>9,} crudos -> {limpios:>9,} limpios -> {len(pares):>7,} usados")
    return resumen, total


def main():
    config.RUTA_PROCESADOS.mkdir(parents=True, exist_ok=True)

    print("Armando validación/prueba multi-paralelas (TED2020, pivote inglés)...")
    val, test, disponibles = construir_eval()
    guardar_eval(val, config.RUTA_PROCESADOS / "validacion.tsv")
    guardar_eval(test, config.RUTA_PROCESADOS / "prueba.tsv")
    print(f"  {disponibles:,} oraciones disponibles en los 4 idiomas -> "
          f"{len(val)} validación, {len(test)} prueba")

    prohibidas = {clave(fila[i]) for fila in val + test for i in config.IDIOMAS}

    print("Limpiando corpus de entrenamiento...")
    resumen, total = construir_entrenamiento(prohibidas, config.RUTA_PROCESADOS / "entrenamiento.tsv")
    print(f"  Total entrenamiento: {total:,} pares ({2 * total:,} ejemplos contando ambas direcciones)")

    with open(config.RUTA_PROCESADOS / "resumen.json", "w", encoding="utf-8") as f:
        json.dump({
            "multiparalelas_disponibles": disponibles,
            "validacion": len(val), "prueba": len(test),
            "entrenamiento_total": total, "por_corpus_y_par": resumen,
        }, f, ensure_ascii=False, indent=2)
    print(f"OK: datos en {config.RUTA_PROCESADOS}")


if __name__ == "__main__":
    main()
