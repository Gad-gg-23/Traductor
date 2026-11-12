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

No se declaran cifras en esta ficha hasta que los resultados de la revisión
correspondiente estén publicados; existir como archivo no demuestra calidad.

## Limitaciones

- Puede producir errores, omisiones o traducciones demasiado literales.
- BLEU/chrF no garantizan equivalencia semántica.
- TED2020 y Europarl no representan todos los registros ni variedades de cada
  idioma.
- Textos médicos, legales o de alto impacto requieren revisión humana.

## Licencia y fuentes

La arquitectura y pesos base se publican con licencia MIT. Los corpus proceden
de OPUS y conservan sus condiciones y atribuciones originales.
