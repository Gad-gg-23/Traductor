# Cuatro Voces — traductor multilingüe

Traductor local para **español, inglés, rumano y alemán** construido sobre
`facebook/m2m100_418M`. Un solo modelo cubre las 12 direcciones y puede recorrer
los otros tres idiomas en cualquier orden antes de volver al original.

```text
es → en → ro → de → es
ro → de → es → en → ro
```

Incluye interfaz gráfica, CLI, preparación reproducible de OPUS, entrenamiento
reanudable, evaluación BLEU/chrF, diagnóstico y pruebas automatizadas.

## Probar el traductor

Se recomienda Python 3.10–3.13 y al menos 4 GB libres. El primer uso descarga el
modelo afinado (~1.94 GB); después queda en la caché de Hugging Face.

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python Traductor.py diagnostico
python Traductor.py interfaz
```

La interfaz se abre en `http://127.0.0.1:7860` y ofrece:

- traducción directa entre dos idiomas;
- una cadena en el orden elegido;
- comparación de las seis rutas posibles;
- perfiles rápido, equilibrado y máxima calidad;
- estado verificable del modelo cargado;
- ejecución local sin enviar el texto a una API externa.

Para iniciar sin abrir automáticamente el navegador:

```bash
python Traductor.py interfaz --no-abrir
```

## Uso por terminal

```bash
# Directa
python Traductor.py predecir --texto "Hola, ¿cómo estás?" --origen es --destino en

# es → en → ro → de → es
python Traductor.py predecir --texto "Hola, ¿cómo estás?" --origen es --cadena en,ro,de

# Las seis permutaciones
python Traductor.py predecir --texto "Hola, ¿cómo estás?" --origen es --todas-las-cadenas

# Menor latencia en CPU
python Traductor.py predecir --texto "Hola" --origen es --destino ro --perfil rapida

# Salida estructurada
python Traductor.py predecir --texto "Hola" --origen es --destino de --json

# Modelo, revisión y dispositivo realmente cargados
python Traductor.py predecir --info-modelo
```

El programa **no sustituye silenciosamente** el modelo afinado por el modelo
base. Si se desea aceptar esa degradación de calidad, debe autorizarse de forma
explícita con `--permitir-modelo-base`.

El perfil predeterminado usa dos candidatos. `--perfil rapida` usa generación
greedy y es especialmente útil para las cadenas; `--perfil maxima` usa cinco
candidatos y puede ser varias veces más lento. Las seis rutas reutilizan los
primeros saltos compartidos para reducir de 24 a 21 generaciones efectivas.

## Diagnóstico

```bash
python Traductor.py diagnostico                 # dependencias, datos y modelo remoto
python Traductor.py diagnostico --cargar-modelo # prueba también los pesos completos
python Traductor.py diagnostico --json          # utilizable desde scripts/CI
```

El diagnóstico revisa versiones, espacio disponible, forma de los TSV y la
revisión inmutable del modelo en Hugging Face.

## Datos y entrenamiento

Los corpus provienen de [OPUS](https://opus.nlpl.eu): TED2020 v1 y Europarl v8.
No se guardan en Git porque ocupan aproximadamente 2.6 GB extraídos.

```bash
python Traductor.py descargar
python Traductor.py preparar
```

La descarga usa reintentos, archivos temporales, validación de tamaño y CRC del
ZIP. Los archivos definitivos solo se reemplazan cuando la extracción termina
correctamente. La preparación también escribe de forma atómica: un corte no
destruye conjuntos válidos anteriores.

### Limpieza

- normalización Unicode NFC y de espacios;
- descarte de vacíos, textos demasiado largos, relaciones de longitud
  sospechosas y pares sin traducir;
- deduplicación por corpus/par;
- validación y prueba multi-paralelas mediante pivote inglés;
- exclusión de las 3,000 oraciones de evaluación de todos los pares de
  entrenamiento.

| Archivo | Contenido |
|---|---|
| `entrenamiento.tsv` | 4.18 M pares aproximados; se regenera localmente |
| `validacion.tsv` | 1,000 oraciones en los cuatro idiomas |
| `prueba.tsv` | 2,000 oraciones en los cuatro idiomas |
| `resumen.json` | conteos y SHA-256 de los conjuntos de evaluación |

### Afinamiento

El entrenamiento está pensado para GPU en
[`colab/Traductor_Colab.ipynb`](colab/Traductor_Colab.ipynb):

```bash
python Traductor.py entrenar
```

- muestreo de reservorio: no carga el TSV de 1.1 GB completo en RAM;
- 15,000 ejemplos por dirección, 180,000 en total;
- tokenización por dirección y en fragmentos de 1,000;
- precisión mixta en CUDA y gradient checkpointing;
- checkpoint por época y reanudación automática;
- selección del menor `eval_loss`;
- guardado transaccional del modelo y metadatos de versiones.

Para conservar checkpoints en Google Drive u otra ubicación persistente:

```bash
export TRADUCTOR_SALIDAS=/ruta/persistente/salidas
python Traductor.py entrenar
```

El tiempo real depende de la GPU y puede ser considerable; no se debe asumir
que 180,000 ejemplos terminarán en pocos minutos.

## Evaluación

```bash
python Traductor.py evaluar
```

Genera atómicamente:

- `salidas/resultados_directos.csv`: BLEU y chrF para las 12 direcciones;
- `salidas/resultados_cadenas.csv`: 24 cadenas, BLEU, chrF y coincidencia exacta;
- `salidas/metadata_evaluacion.json`: modelo/revisión, semilla y tamaño de muestra.

BLEU y chrF miden coincidencia textual, no equivalencia semántica perfecta. Para
una entrega importante se deben conservar estos resultados y complementarlos
con revisión humana de una muestra por dirección.

## Reproducibilidad del modelo

El modelo compartido es
[`GadiGG/traductor-escom`](https://huggingface.co/GadiGG/traductor-escom). El
código fija una revisión concreta en `config.HF_REVISION_MODELO`; una subida
nueva no altera silenciosamente resultados anteriores.

Transformers está fijado a **5.21.0**, la versión con la que se publicó el
artefacto actual. El tokenizador guardado por esa versión no es compatible con
Transformers 4.x; por eso no se debe relajar esa restricción sin ejecutar todas
las pruebas y volver a validar el modelo.

## Pruebas

```bash
python -m compileall -q -x '(^|/)(\.venv|\.git)/' .
python -m unittest discover -v
```

Las pruebas cubren validación de entradas, cadenas, muestreo balanceado, API de
la interfaz y enrutamiento de la CLI. GitHub Actions las ejecuta en Python 3.11
y 3.12.

## Estructura

```text
Traductor.py              entrada única
motor_traduccion.py       carga segura, dispositivos e inferencia
interfaz.py               servidor web local
web/                      interfaz HTML/CSS/JS offline
descargar_datos.py        descarga transaccional de OPUS
preparar_datos.py         limpieza y particiones
muestreo.py               muestreo de reservorio
entrenar.py               afinamiento y checkpoints
evaluar.py                BLEU/chrF directo y round-trip
diagnostico.py            verificación de instalación
tests/                    pruebas automatizadas
```

## Límites conocidos

- Ningún traductor automático garantiza traducciones perfectas.
- En CPU, cargar y generar con un modelo de ~484 M parámetros puede ser lento.
- Las seis cadenas realizan 24 traducciones y son mucho más costosas que una
  traducción directa.
- El contenido legal, médico o académico crítico requiere revisión humana.
