# Traductor multilingüe: español · inglés · rumano · alemán

Práctica de Machine Learning (ESCOM, IPN). Entra una oración en un idioma,
se traduce a los otros tres en cualquier orden y se regresa al idioma
original. Ejemplo:

```
es -> en -> ro -> de -> es
ro -> de -> es -> en -> ro
... (todas las permutaciones)
```

Son 4 idiomas × 3 destinos = **12 direcciones de traducción**. El plan es un
solo Transformer multilingüe al que se le indica el idioma destino con una
etiqueta al inicio de la entrada (ej. `<2ro> Hola, ¿cómo estás?`), en vez de
12 modelos separados.

## Inicio rápido

```bash
pip install -r requirements.txt
python Traductor.py descargar   # ~875 MB desde OPUS -> data/crudos/ (unos minutos)
python Traductor.py preparar    # limpia y arma los conjuntos -> data/procesados/ (~3 min)
python Traductor.py entrenar    # afina M2M100 (necesita GPU, ver colab/Traductor_Colab.ipynb)
python Traductor.py evaluar     # BLEU/chrF directo y en cadena sobre prueba.tsv
python Traductor.py predecir --texto "Hola, ¿cómo estás?" --origen es --todas-las-cadenas
```

**El equipo entrena en Colab, no localmente** (no hay GPU en esta máquina ni se espera que la haya en las del equipo): ver [`colab/Traductor_Colab.ipynb`](colab/Traductor_Colab.ipynb) y la guía paso a paso en [GUIA_EQUIPO.md](GUIA_EQUIPO.md), que cubre también cómo compartir el modelo entrenado.

Los corpus **no están en GitHub** (pesan 2.6 GB extraídos y GitHub no acepta
archivos de más de 100 MB). Cada quien los baja con el script; OPUS es
público, no hace falta cuenta ni token.

## Datos

Fuente: [OPUS](https://opus.nlpl.eu), formato *moses* (dos `.txt` alineados
línea a línea por par de idiomas).

| Corpus | Qué es | Pares por idioma |
|---|---|---|
| **TED2020** v1 | Subtítulos de charlas TED: oraciones habladas y cotidianas | 255k – 400k |
| **Europarl** v8 | Debates del Parlamento Europeo: formal y bien traducido | 370k – 1.9M (recortado a 400k) |

Europarl se recorta a 400k por par para que de/en/es (que tienen ~1.9M) no
dominen el entrenamiento y el rumano no quede relegado.

### Limpieza (`preparar_datos.py`)

- Normalización Unicode (NFC) y de espacios.
- Se descartan pares vacíos, de más de 80 palabras, con un lado 2.5× más largo
  que el otro, o sin traducir (texto idéntico en ambos idiomas).
- Se eliminan pares duplicados.

### Conjuntos resultantes (`data/procesados/`)

| Archivo | Contenido | ¿En Git? |
|---|---|---|
| `entrenamiento.tsv` | 4,178,522 pares (`idioma_a, idioma_b, texto_a, texto_b`); cada par sirve en ambas direcciones → 8.36M ejemplos | No (1.1 GB) |
| `validacion.tsv` | 1,000 oraciones, columnas `de, en, es, ro` | Sí |
| `prueba.tsv` | 2,000 oraciones, columnas `de, en, es, ro` | Sí |
| `resumen.json` | Conteos por corpus y par (crudos / limpios / usados) | Sí |

Pares de entrenamiento por par de idiomas (ambos corpus):

| de-en | de-es | de-ro | en-es | en-ro | es-ro |
|---|---|---|---|---|---|
| 683,834 | 682,941 | 628,823 | 799,743 | 699,062 | 684,119 |

**Validación y prueba son multi-paralelas:** cada fila es la *misma* oración en
los 4 idiomas. Eso permite medir la cadena completa (es → en → ro → de → es)
y cuánto se pierde en cada salto. Se arman desde TED2020 usando el inglés como
pivote, con filtros extra: 4–40 palabras, sin anotaciones tipo "(Risas)" o ♫,
y las 4 versiones de longitud parecida.

**Sin fuga de datos:** cualquier oración de validación/prueba (en cualquier
idioma) se elimina de todos los pares de entrenamiento. Verificado: 0
coincidencias.

Para leer los TSV con pandas:

```python
import csv, pandas as pd
df = pd.read_csv("data/procesados/prueba.tsv", sep="\t", quoting=csv.QUOTE_NONE,
                 escapechar="\\", keep_default_na=False)
```

(`keep_default_na=False` importa: si no, pandas convierte oraciones como "NA"
o "null" en valores vacíos.)

## Estructura

```
Traductor.py         punto de entrada (descargar / preparar / entrenar / evaluar / predecir)
config.py            idiomas, corpus, modelo e hiperparámetros, todo centralizado
descargar_datos.py   baja y verifica los corpus de OPUS
preparar_datos.py    limpieza, deduplicación y particiones
entrenar.py           afina M2M100 sobre data/procesados/entrenamiento.tsv
evaluar.py            BLEU/chrF directo (12 direcciones) y en cadena (round-trip) sobre prueba.tsv
predecir.py           traduce una oración: directa, en una cadena específica, o en las 6 permutaciones
colab/Traductor_Colab.ipynb   notebook para correr todo con GPU gratuita
GUIA_EQUIPO.md        guía paso a paso para el equipo (Colab, Hugging Face, troubleshooting)
data/crudos/          corpus tal cual de OPUS             (ignorado por Git)
data/procesados/      conjuntos listos para entrenar
salidas/               modelo afinado y métricas            (el modelo, ignorado por Git)
```

## Modelo

**`facebook/m2m100_418M`** (Transformer encoder-decoder preentrenado, muchos-a-muchos entre
100 idiomas), afinado con nuestros datos de OPUS. Se eligió un modelo preentrenado y no uno
entrenado desde cero porque ya sabe traducir razonablemente bien antes de empezar — en las
pocas horas de GPU gratuita de Colab, afinarlo da resultados muchísimo mejores que entrenar un
Transformer desde cero con los mismos datos. Es la misma idea de *transfer learning* que
`Frutas_Verduras`/`Comida` usan con MobileNetV2, aplicada aquí a traducción.

Por qué M2M100 y no MarianMT (`Helsinki-NLP/opus-mt-*`): MarianMT es **bilingüe**, un modelo
distinto por cada par de idiomas (necesitaríamos cargar varios modelos para cubrir las 12
direcciones). M2M100 traduce directo entre cualquier par de sus 100 idiomas con un solo modelo
— justo lo que pide la práctica — y sus códigos de idioma (`de`, `en`, `es`, `ro`) son
exactamente los mismos que usamos en todo el proyecto, sin ninguna tabla de conversión.

### Afinamiento (`entrenar.py`)

4,178,522 pares de `entrenamiento.tsv` (x2 direcciones) son demasiados para una sesión de
Colab gratuita, así que se toma una muestra pareja: **15,000 ejemplos por cada una de las 12
direcciones** (180,000 en total), 2 épocas, lote efectivo 32, `fp16` si hay GPU. Todo ajustable
en `config.py` (`EJEMPLOS_POR_DIRECCION`, `EPOCAS`, etc.) si el equipo quiere más calidad a
cambio de más tiempo de Colab.

El modelo afinado se guarda en `salidas/modelo_m2m100/` — **no va a GitHub** (pesa ~1.9 GB,
el límite de GitHub es 100 MB por archivo). Se comparte en Hugging Face:
**[huggingface.co/GadiGG/traductor-escom](https://huggingface.co/GadiGG/traductor-escom)**
(mismo patrón que el dataset de `Identificador_Rostros`, pero aquí es un repo de modelo). Ver
Parte 6 de [GUIA_EQUIPO.md](GUIA_EQUIPO.md).

### Evaluación (`evaluar.py`)

Sobre las 2,000 oraciones de `prueba.tsv`, nunca vistas en el afinamiento:

1. **Directa** (`salidas/resultados_directos.csv`): las 12 direcciones, traducción vs.
   traducción real de TED2020, con **BLEU** y **chrF** (métricas estándar de traducción
   automática, calculadas con `sacrebleu` para que el número no dependa de cómo cada uno
   tokenice el texto).
2. **En cadena** (`salidas/resultados_cadenas.csv`) — la tarea real de la práctica: desde cada
   uno de los 4 idiomas de inicio se prueban las **6 permutaciones** posibles de los otros 3
   (4 saltos de traducción por cadena) y se compara el texto final contra el **original** — no
   hace falta traducción humana de referencia para esta parte, es la misma oración de donde
   salió la cadena. Usa un subconjunto de 100 oraciones de prueba por costo (4 × 6 × 4 = 96
   traducciones por oración).

### Inferencia (`predecir.py`)

```bash
python predecir.py --texto "Hola, ¿cómo estás?" --origen es --destino en              # directa
python predecir.py --texto "Hola, ¿cómo estás?" --origen es --cadena en,ro,de         # cadena específica
python predecir.py --texto "Hola, ¿cómo estás?" --origen es --todas-las-cadenas       # las 6 permutaciones
```
