# Guía para el equipo — Traductor multilingüe (es/en/ro/de)

Repositorio: **[github.com/Gad-gg-23/Traductor](https://github.com/Gad-gg-23/Traductor)**

Notebook de Colab (ábrelo directo con este link, no hace falta buscarlo):
**[colab.research.google.com/.../Traductor_Colab.ipynb](https://colab.research.google.com/github/Gad-gg-23/Traductor/blob/main/colab/Traductor_Colab.ipynb)**

Modelo afinado, compartido en Hugging Face (no en GitHub, pesa ~1.9 GB):
**[huggingface.co/GadiGG/traductor-escom](https://huggingface.co/GadiGG/traductor-escom)**

---

## Qué hace esta práctica, en una frase

Entra una oración en español, inglés, rumano o alemán; el modelo la traduce a los otros tres idiomas en **cualquier orden** y siempre puede regresarla al idioma original (es → en → ro → de → es, ro → de → es → en → ro, etc. — las 12 direcciones de traducción, 1 solo modelo).

---

## Parte 0 — Lo que hace el admin, una sola vez

1. **GitHub:** entra al repo → **Settings → Collaborators → Add people** → escribe el usuario/correo de GitHub de cada compañero → **Add**. Cada quien acepta la invitación antes de poder `git push`.
2. **Hugging Face (solo si alguien más va a subir el modelo entrenado, Parte 6):** entra a [huggingface.co/GadiGG/traductor-escom](https://huggingface.co/GadiGG/traductor-escom) → **Settings → Collaborators** → agrega su usuario de Hugging Face. Si el repo todavía no existe la primera vez que alguien corre la Parte 6, `push_to_hub` lo crea solo — no hace falta crearlo a mano antes, a diferencia de GitHub.
3. Comparte esta guía y el link del notebook de arriba con el equipo.

El resto de la guía es para **cada compañero**.

---

## Parte 1 — Correr el notebook (cada persona, todas las veces que entrene)

**No se necesita cuenta de Kaggle ni de OPUS.** Solo necesitas una cuenta de Hugging Face (gratis, con un *token*) si quieres subir tu modelo entrenado al repo compartido (Parte 6, opcional).

### 1.1 Abrir el notebook y activar GPU

1. Abre el link del notebook de arriba, inicia sesión con tu cuenta de Google si te lo pide.
2. `Entorno de ejecución` (menú de arriba) → `Cambiar tipo de entorno de ejecución` → **GPU** (la T4 gratuita) → **Guardar**.

### 1.2 Correr todo

`Entorno de ejecución → Ejecutar todas` (o celda por celda con ▶️, de arriba hacia abajo). Lo que vas a ver pasar, en orden:

1. **Clona el repo** — rápido, segundos.
2. **Instala lo que falta y verifica la GPU** — `transformers`, `sentencepiece`, `sacrebleu`, `accelerate`. Debe imprimir `GPU disponible: True`. Si dice `False`, te faltó el paso 1.1.2.
3. **Descarga los corpus de OPUS** (`Traductor.py descargar`) — ~875 MB, tarda unos minutos. Verifica que cada idioma tenga el mismo número de líneas que su pareja; si algo quedó a medias, lo vuelve a bajar.
4. **Prepara los datos** (`Traductor.py preparar`) — limpia, quita duplicados, separa validación/prueba (multi-paralelas: la misma oración en los 4 idiomas) y se asegura de que ninguna de esas oraciones esté también en el entrenamiento. Tarda 2-4 minutos.
5. **Afina el modelo** (`Traductor.py entrenar`) — parte de `facebook/m2m100_418M` (ya sabe traducir entre 100 idiomas) y lo afina con una muestra de nuestros datos de OPUS (180,000 ejemplos repartidos entre las 12 direcciones). Va a imprimir el progreso; **espera a que termine por completo** — al final debe decir `OK: modelo afinado en .../salidas/modelo_m2m100`. Tarda varios minutos en GPU.
6. **Evalúa** (`Traductor.py evaluar`) — esta es la prueba real de la práctica:
   - **Directa:** traduce las 2,000 oraciones de prueba en las 12 direcciones y las compara con la traducción real de TED2020 (BLEU y chrF, las métricas estándar de traducción automática — más alto es mejor, BLEU de 100 sería traducción perfecta palabra por palabra).
   - **En cadena (round-trip):** desde cada uno de los 4 idiomas se recorren los otros 3 en sus 6 órdenes posibles y se regresa al idioma original, comparando el resultado final contra la oración de partida — sin necesitar ninguna traducción humana de referencia para esta parte.
   Tarda varios minutos más (son 12 + 24 = 36 evaluaciones con generación de texto).
7. **Ver resultados** — imprime las tablas de BLEU/chrF por dirección y por cadena, más el promedio por idioma de inicio.
8. **Descargar o subir el modelo** (opcional) — ver Parte 6.
9. **Probar tu propia oración** (opcional) — ver Parte 5.

---

## Parte 2 — Cómo correrlo en tu computadora en vez de Colab (opcional)

Solo para los pasos de datos (sin GPU, el afinamiento sería lentísimo):

```bash
cd Traductor
pip install -r requirements.txt
python Traductor.py descargar
python Traductor.py preparar
```

`entrenar.py` y `evaluar.py` sí necesitan GPU para terminar en un tiempo razonable — para eso está el notebook.

---

## Parte 3 — De dónde salen los datos (por si alguien pregunta en la exposición)

- **OPUS** (opus.nlpl.eu), dos corpus: **TED2020** (subtítulos de charlas TED, lenguaje hablado/cotidiano) y **Europarl** (debates del Parlamento Europeo, lenguaje formal).
- Se limpian oraciones vacías, demasiado largas, con un lado mucho más largo que el otro, o que quedaron sin traducir.
- **Validación y prueba son multi-paralelas**: para que la misma oración exista en los 4 idiomas a la vez (necesario para evaluar la cadena completa), se construyen desde TED2020 usando el inglés como "puente" entre los otros tres.
- Esas 3,000 oraciones (1,000 validación + 2,000 prueba) se **sacan del entrenamiento en los 4 idiomas**, para que el modelo nunca las haya visto. Verificado: cero coincidencias.
- Todos los detalles y números exactos están en [README.md](README.md).

---

## Parte 4 — Por qué un Transformer preentrenado y no uno desde cero

`facebook/m2m100_418M` ya sabe traducir entre 100 idiomas porque Meta lo entrenó con muchísimos más datos de los que nosotros podríamos juntar o procesar en Colab. Lo afinamos (*fine-tuning*) con nuestros propios datos de OPUS para que mejore específicamente en nuestras 12 direcciones, en vez de entrenar un Transformer desde cero, que con unas horas de GPU gratuita produciría traducciones mucho peores. Es la misma idea de *transfer learning* que ya usamos con MobileNetV2 en Frutas_Verduras y Comida, aplicada a traducción en vez de clasificación de imágenes.

M2M100 traduce directo entre cualquier par de sus idiomas (no pasa por inglés como paso intermedio obligatorio), y sus códigos de idioma (`de`, `en`, `es`, `ro`) son exactamente los mismos que usamos en todo el proyecto.

---

## Parte 5 — Probar con tu propia oración

Desde la terminal (local, con el modelo ya afinado en `salidas/modelo_m2m100/`, o en la última celda del notebook):

```bash
# Traducción directa a un solo idioma
python predecir.py --texto "Hola, ¿cómo estás?" --origen es --destino en

# Una cadena específica: es -> en -> ro -> de -> es
python predecir.py --texto "Hola, ¿cómo estás?" --origen es --cadena en,ro,de

# Las 6 permutaciones posibles de los otros 3 idiomas, de una sola vez
python predecir.py --texto "Hola, ¿cómo estás?" --origen es --todas-las-cadenas
```

---

## Parte 6 — Compartir el modelo entrenado con el equipo (opcional)

El modelo afinado pesa **~1.9 GB** (M2M100 completo, no solo una cabeza de clasificación) — **no va a GitHub**, que rechaza archivos de más de 100 MB. Por eso se comparte en Hugging Face, igual que el dataset de fotos de Identificador_Rostros, pero aquí es un repo de *modelo*, no de *dataset*.

**Opción A — descargarlo a tu computadora:** la celda de la Parte 6 del notebook comprime `salidas/modelo_m2m100/` en un .zip y lo descarga. Tarda un rato por el tamaño.

**Opción B — subirlo a Hugging Face** (recomendado, así el equipo comparte el último modelo entrenado):
1. Ve a https://huggingface.co/settings/tokens
2. **Create new token** → tipo **Write** → nómbralo (ej. "Colab Traductor") → **Create token**.
3. Cópialo — solo se muestra una vez.
4. Pégalo en el campo de la celda (no se ve mientras escribes, es normal).

A diferencia de un repo de GitHub, **no hace falta que el admin lo cree primero**: `push_to_hub` crea `GadiGG/traductor-escom` automáticamente la primera vez que alguien lo corre con un token válido. Si tu cuenta no es colaboradora todavía, pídele al admin que te agregue (Parte 0.2).

⚠️ El token es como una contraseña: no lo compartas ni lo pegues en chats. Si se expone, revócalo en esa misma página y genera uno nuevo.

Para que cualquiera use el modelo ya subido (sin tener que entrenarlo de nuevo):

```python
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

modelo = M2M100ForConditionalGeneration.from_pretrained("GadiGG/traductor-escom")
tokenizer = M2M100Tokenizer.from_pretrained("GadiGG/traductor-escom")
```

---

## Problemas comunes

| Error | Causa | Solución |
|---|---|---|
| `GPU disponible: False` | No activaste GPU | `Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU` |
| `Error en <corpus> <par>: incompleto: ...` | La descarga de OPUS se cortó a medias (conexión) | Vuelve a correr `Traductor.py descargar`, retoma solo lo que falta |
| `Error en <corpus> <par>: desalineado: ...` | El .zip de OPUS llegó corrupto | Borra `data/crudos/<corpus>/<par>/` y vuelve a descargar ese par |
| `No hay modelo afinado en .../modelo_m2m100` (al correr `evaluar.py` o `predecir.py`) | No corriste `entrenar.py` todavía, o se corrió en otra sesión de Colab que ya se cerró | Corre `python Traductor.py entrenar` primero |
| `[Errno 2] No such file or directory: '/content/repo'` | Colab reinició la sesión (VM nueva), se perdió todo lo descargado | `Entorno de ejecución → Ejecutar todas` desde el principio |
| El `push_to_hub` de la Parte 6 falla con error de permisos | Tu cuenta de Hugging Face no es colaboradora del repo, o el token no tiene permiso Write | Pide que te agreguen como colaborador (Parte 0.2), o genera un token nuevo con permiso **Write** |
| BLEU muy bajo (cercano a 0) pero chrF razonable | Normal en frases muy cortas o con pocas épocas de afinamiento — BLEU penaliza mucho cuando casi no hay coincidencias exactas de 4 palabras seguidas; chrF (a nivel de caracteres) es más indulgente y suele ser más informativo aquí | No es un bug; si quieres mejorar BLEU, sube `EPOCAS` o `EJEMPLOS_POR_DIRECCION` en `config.py` (tarda más en Colab) |
