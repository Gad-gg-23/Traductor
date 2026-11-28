---
library_name: transformers
pipeline_tag: translation
license: mit
base_model: facebook/m2m100_418M
language:
  - de
  - en
  - es
  - ro
tags:
  - m2m100
  - multilingual-translation
  - opus
---

# Cuatro Voces — M2M100 afinado

Modelo muchos-a-muchos para traducción directa entre alemán (`de`), inglés
(`en`), español (`es`) y rumano (`ro`). Es un afinamiento de
`facebook/m2m100_418M` con pares de TED2020 y Europarl obtenidos de OPUS.

## Uso

```python
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

repo = "GadiGG/traductor-escom"
tokenizer = M2M100Tokenizer.from_pretrained(repo)
model = M2M100ForConditionalGeneration.from_pretrained(repo)

tokenizer.src_lang = "es"
inputs = tokenizer("Hola, ¿cómo estás?", return_tensors="pt")
output = model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.get_lang_id("en"),
    max_new_tokens=96,
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

El repositorio de código incluye una interfaz gráfica local, CLI, preparación
de datos, evaluación y pruebas:
https://github.com/Gad-gg-23/Traductor

## Entrenamiento

- 12 direcciones equilibradas.
- 15,000 ejemplos por dirección; 180,000 en total.
- 2 épocas, lote efectivo 32.
- Longitud máxima de 96 subpalabras.
- Validación multi-paralela separada del entrenamiento.
- Transformers 5.21.0.

Los conteos exactos, semilla y versiones se guardan junto al modelo en
`metadata_entrenamiento.json`.

## Evaluación

El proyecto calcula BLEU y chrF sobre 2,000 oraciones multi-paralelas para las
12 direcciones, además de 24 recorridos completos de cuatro saltos. Los CSV y
`metadata_evaluacion.json` se generan con `python Traductor.py evaluar` y deben
publicarse junto con cada versión validada.

Promedio de BLEU/chrF por idioma de inicio sobre las 24 cadenas de cuatro
saltos (round-trip):

| Idioma de inicio | BLEU | chrF |
|---|---|---|
| de | 28.62 | 55.85 |
| en | 33.90 | 57.72 |
| es | 32.42 | 57.99 |
| ro | 24.68 | 52.84 |

El rumano rinde más bajo de forma consistente, esperable por tener menos datos
en OPUS para ese idioma. El detalle completo por dirección y por cadena está en
`resultados_directos.csv` y `resultados_cadenas.csv`, publicados junto al
modelo.

### Modismos (afinamiento dirigido)

`data/modismos/modismos.tsv` agrega 12 oraciones (6 modismos × 2 variantes) con
su equivalente real -no literal- en los 4 idiomas, repetidas durante el
afinamiento. Resultado:

- **Modismos entrenados:** BLEU subió de ~19–30 a ~99–100 frente al modelo
  anterior. Es memorización esperada (se repitieron 30x), no evidencia de
  generalización por sí sola.
- **Modismos held-out** (2 modismos nunca vistos en el afinamiento, evaluados
  en las 12 direcciones): la generalización es **parcial**. Cuando la frase de
  origen es semánticamente transparente el modelo produce un equivalente real
  razonable (p. ej. alemán "ein Vermögen kosten" → español "costó una
  fortuna"), pero en la mayoría de direcciones todavía traduce el modismo
  palabra por palabra (p. ej. "un ojo de la cara" → inglés literal "one eye on
  his face" en vez de "an arm and a leg"). Con solo 12 ejemplos de
  entrenamiento no se debe esperar generalización robusta a modismos
  completamente nuevos; eso requeriría un conjunto mucho más grande.

## Limitaciones

- Puede producir errores, omisiones o traducciones demasiado literales,
  especialmente en modismos que no estuvieron en el conjunto de afinamiento
  dirigido (ver sección Modismos).
- BLEU/chrF no garantizan equivalencia semántica.
- TED2020 y Europarl no representan todos los registros ni variedades de cada
  idioma.
- El rumano "nouă" ("nueva"/"nueve") es ambiguo para el modelo: se observó al
  menos un caso de confusión real (p. ej. "Mașina aceea nouă..." tradujo
  "nouă" como el número en vez del adjetivo). Es una limitación heredada del
  modelo base, no introducida por este afinamiento.
- Textos médicos, legales o de alto impacto requieren revisión humana.

## Licencia y fuentes

La arquitectura y pesos base se publican con licencia MIT. Los corpus proceden
de OPUS y conservan sus condiciones y atribuciones originales.
