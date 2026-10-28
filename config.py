"""
Configuración central del traductor multilingüe (español, inglés, rumano, alemán).

Un solo lugar con rutas, idiomas, corpus y parámetros de limpieza, para que
descargar_datos.py, preparar_datos.py y el entrenamiento no se desincronicen.

La idea de la práctica: entra una oración en un idioma, pasa por los otros
tres (en cualquier orden) y regresa al idioma original. Eso son 4 idiomas x
3 destinos = 12 direcciones de traducción. En vez de 12 modelos, el plan es
UN solo Transformer multilingüe al que se le indica el idioma destino con una
etiqueta (ej. "<2ro>") al inicio de la oración de entrada.
"""

from itertools import combinations
from pathlib import Path

# =========================================================
# RUTAS
# =========================================================

RAIZ = Path(__file__).resolve().parent

RUTA_DATOS = RAIZ / "data"
RUTA_CRUDOS = RUTA_DATOS / "crudos"        # .zip y .txt tal cual vienen de OPUS (NO van a Git)
RUTA_PROCESADOS = RUTA_DATOS / "procesados"  # train/val/test ya limpios

RUTA_SALIDAS = RAIZ / "salidas"

# =========================================================
# IDIOMAS
# =========================================================

IDIOMAS = ["de", "en", "es", "ro"]  # orden alfabético, fijo (OPUS nombra los pares así)
NOMBRES_IDIOMAS = {"de": "alemán", "en": "inglés", "es": "español", "ro": "rumano"}

# Los 6 pares sin dirección; cada uno da 2 direcciones (de->en y en->de).
PARES = [f"{a}-{b}" for a, b in combinations(IDIOMAS, 2)]

# =========================================================
# CORPUS DE OPUS (https://opus.nlpl.eu)
# =========================================================
# Se usa el formato "moses": un .zip por par con dos .txt alineados línea a
# línea (línea i del .de = traducción de la línea i del .en).
#
# - TED2020: subtítulos de charlas TED. Oraciones habladas y cotidianas, lo
#   más parecido a lo que el profe va a escribir. ~270k-420k por par y
#   equilibrado entre los 6 pares (rumano no queda en desventaja).
# - Europarl: debates del Parlamento Europeo. Formal, pero bien traducido.
#   ~390k pares con rumano y ~2M entre de/en/es.
#
# alineaciones = número de líneas que reporta la API de OPUS; descargar_datos.py
# lo usa (con 1% de tolerancia) para verificar que el archivo llegó completo.
CORPUS = {
    "TED2020": {
        "version": "v1",
        "alineaciones": {
            "de-en": 296288, "de-es": 295366, "de-ro": 267246,
            "en-es": 417195, "en-ro": 328195, "es-ro": 326409,
        },
    },
    "Europarl": {
        "version": "v8",
        "alineaciones": {
            "de-en": 1961119, "de-es": 1887879, "de-ro": 385663,
            "en-es": 2009073, "en-ro": 400356, "es-ro": 387653,
        },
    },
}

URL_OPUS = "https://object.pouta.csc.fi/OPUS-{corpus}/{version}/moses/{par}.txt.zip"

# =========================================================
# LIMPIEZA Y PARTICIÓN (preparar_datos.py)
# =========================================================

SEMILLA = 42

MIN_PALABRAS = 1
MAX_PALABRAS = 80          # oraciones más largas casi siempre son párrafos mal alineados
MAX_RATIO_LONGITUD = 2.5   # si una lado es 2.5x más largo que el otro, la alineación es sospechosa

# Europarl tiene ~2M pares en de/en/es pero solo ~390k con rumano. Se recorta
# para que ningún par domine el entrenamiento y el rumano no quede relegado.
MAX_PARES_POR_CORPUS_Y_PAR = 400_000

# Validación y prueba salen de TED2020 y son MULTI-PARALELAS: la misma oración
# existe en los 4 idiomas. Así se puede evaluar la cadena completa
# es -> en -> ro -> de -> es comparando contra la oración original real.
N_VALIDACION = 1000
N_PRUEBA = 2000
