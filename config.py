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

# =========================================================
# MODELO: Transformer preentrenado multilingüe (M2M100)
# =========================================================
# M2M100 traduce directamente entre cualquier par de sus 100 idiomas sin
# pasar por inglés como intermediario (a diferencia de MarianMT/opus-mt, que
# son modelos bilingües, uno por par), y sus códigos de idioma son de 2
# letras idénticos a los de IDIOMAS (de, en, es, ro) -- no hace falta
# ninguna tabla de conversión entre nuestros códigos y los del modelo.
MODELO_BASE = "facebook/m2m100_418M"

RUTA_MODELO_DIR = RUTA_SALIDAS / "modelo_m2m100"  # modelo afinado (NO va a Git, ~1.9 GB)
RUTA_HISTORICO_CSV = RUTA_SALIDAS / "historico_entrenamiento.csv"
RUTA_RESULTADOS_DIRECTOS_CSV = RUTA_SALIDAS / "resultados_directos.csv"
RUTA_RESULTADOS_CADENAS_CSV = RUTA_SALIDAS / "resultados_cadenas.csv"

# Repositorio de Hugging Face donde se comparte el modelo ya afinado: los
# pesos pesan ~1.9 GB y GitHub rechaza archivos de más de 100 MB. Mismo
# patrón que el dataset de Identificador_Rostros, pero aquí es un repo de
# MODELOS, no de datasets, y a diferencia de GitHub, push_to_hub() puede
# crearlo solo (no hace falta crearlo a mano antes).
HF_REPO_MODELO = "GadiGG/traductor-escom"

# =========================================================
# AFINAMIENTO (entrenar.py)
# =========================================================
# 4.18M pares de entrenamiento x 2 direcciones son demasiados para afinar en
# una sesión de Colab gratuita. Se toma una muestra pareja por dirección para
# que ninguna dirección domine el afinamiento (igual que el recorte de
# Europarl en preparar_datos.py, misma idea aplicada aquí).
EJEMPLOS_POR_DIRECCION = 15_000  # x 12 direcciones = 180,000 ejemplos de afinamiento
MAX_TOKENS = 96  # en subpalabras, no en palabras; 80 palabras (MAX_PALABRAS) caben holgado

EPOCAS = 2
TAMANO_LOTE = 16
ACUMULACION_GRADIENTE = 2  # lote efectivo = 32
TASA_APRENDIZAJE = 3e-5
PASOS_CALENTAMIENTO = 500
EJEMPLOS_EVAL_POR_DIRECCION = 40  # solo para medir pérdida durante el afinamiento, no el BLEU real

# =========================================================
# EVALUACIÓN (evaluar.py)
# =========================================================
# Directa: se traduce cada columna de prueba.tsv a cada una de las otras 3
# idiomas (12 direcciones) y se compara con la traducción real de TED2020
# usando BLEU y chrF (ambas métricas estándar de traducción automática).
TAMANO_LOTE_GENERACION = 32
LONGITUD_MAXIMA_GENERACION = 96

# En cadena (round-trip): la tarea real de la práctica. Desde cada uno de
# los 4 idiomas de inicio se prueban las 6 permutaciones posibles de los
# otros 3 (4 saltos de traducción por cadena) y se compara el texto final
# contra el ORIGINAL -- no hace falta traducción humana de referencia, es
# la misma oración de donde salió la cadena. Se usa un subconjunto de
# prueba.tsv porque son 4 idiomas x 6 órdenes x 4 saltos = 96 traducciones
# por cada oración de prueba.
N_ORACIONES_CADENAS = 100
