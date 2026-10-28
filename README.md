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
```

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
Traductor.py         punto de entrada (descargar / preparar)
config.py            idiomas, corpus, rutas y parámetros de limpieza
descargar_datos.py   baja y verifica los corpus de OPUS
preparar_datos.py    limpieza, deduplicación y particiones
data/crudos/         corpus tal cual de OPUS            (ignorado por Git)
data/procesados/     conjuntos listos para entrenar
```

## Modelo

Pendiente.
