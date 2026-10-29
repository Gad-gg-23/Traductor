"""
Traduce con el modelo ya afinado. Busca, en orden: el modelo local
(salidas/modelo_m2m100/, el que deja entrenar.py), el modelo compartido del
equipo en Hugging Face (config.HF_REPO_MODELO, se descarga solo la primera
vez y queda en caché), y por último el M2M100 preentrenado sin afinar.

Traducción directa entre dos idiomas:
    python predecir.py --texto "Hola, ¿cómo estás?" --origen es --destino en

Cadena completa (round-trip): pasa por los otros 3 idiomas en el orden dado
y regresa al idioma de origen al final:
    python predecir.py --texto "Hola, ¿cómo estás?" --origen es --cadena en,ro,de

Las 6 permutaciones posibles de los otros 3 idiomas, de una sola vez -- esto
es exactamente la tarea de la práctica (que entre cualquier orden, siempre
se pueda regresar al idioma original):
    python predecir.py --texto "Hola, ¿cómo estás?" --origen es --todas-las-cadenas
"""

import argparse
from itertools import permutations

import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

import config

_modelo = None
_tokenizer = None


def cargar_modelo():
    global _modelo, _tokenizer
    if _modelo is None:
        if config.RUTA_MODELO_DIR.exists():
            ruta = config.RUTA_MODELO_DIR
        else:
            ruta = config.HF_REPO_MODELO
            print(f"Aviso: no hay modelo afinado local en {config.RUTA_MODELO_DIR}. "
                  f"Descargando el del equipo desde https://huggingface.co/{config.HF_REPO_MODELO} "
                  f"(solo la primera vez, luego queda en caché)...")
        try:
            _tokenizer = M2M100Tokenizer.from_pretrained(ruta)
            _modelo = M2M100ForConditionalGeneration.from_pretrained(ruta)
        except Exception as error:
            print(f"Aviso: no se pudo cargar {ruta} ({error}). Usando {config.MODELO_BASE} sin afinar.")
            _tokenizer = M2M100Tokenizer.from_pretrained(config.MODELO_BASE)
            _modelo = M2M100ForConditionalGeneration.from_pretrained(config.MODELO_BASE)
        _modelo.eval()
        if torch.cuda.is_available():
            _modelo.to("cuda")
    return _modelo, _tokenizer


@torch.inference_mode()
def traducir(texto, origen, destino):
    modelo, tokenizer = cargar_modelo()
    tokenizer.src_lang = origen
    entradas = tokenizer(texto, return_tensors="pt", truncation=True, max_length=config.MAX_TOKENS)
    entradas = {k: v.to(modelo.device) for k, v in entradas.items()}
    generados = modelo.generate(
        **entradas,
        forced_bos_token_id=tokenizer.get_lang_id(destino),
        max_length=config.LONGITUD_MAXIMA_GENERACION,
    )
    return tokenizer.batch_decode(generados, skip_special_tokens=True)[0]


def traducir_cadena(texto, origen, orden):
    """orden: los otros 3 idiomas en el orden en que se visitan. Regresa a
    origen al final. Resultado: lista [(idioma, texto), ...] con cada paso,
    empezando por el texto original."""
    pasos = [(origen, texto)]
    idioma_actual, texto_actual = origen, texto
    for siguiente in list(orden) + [origen]:
        texto_actual = traducir(texto_actual, idioma_actual, siguiente)
        idioma_actual = siguiente
        pasos.append((idioma_actual, texto_actual))
    return pasos


def imprimir_cadena(pasos):
    print("  " + " -> ".join(idioma for idioma, _ in pasos))
    for idioma, texto in pasos:
        print(f"    [{idioma}] {texto}")


def main():
    parser = argparse.ArgumentParser(description="Traduce con el M2M100 afinado del equipo.")
    parser.add_argument("--texto", required=True)
    parser.add_argument("--origen", required=True, choices=config.IDIOMAS)
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--destino", choices=config.IDIOMAS, help="Traducción directa a un solo idioma.")
    grupo.add_argument("--cadena", help="Idiomas separados por coma, en el orden a visitar "
                                        "(regresa a --origen al final). Ej: en,ro,de")
    grupo.add_argument("--todas-las-cadenas", action="store_true",
                       help="Prueba las 6 permutaciones posibles de los otros 3 idiomas.")
    args = parser.parse_args()

    otros = [i for i in config.IDIOMAS if i != args.origen]

    if args.destino:
        print(traducir(args.texto, args.origen, args.destino))
    elif args.cadena:
        orden = args.cadena.split(",")
        if sorted(orden) != sorted(otros):
            parser.error(f"--cadena debe listar exactamente estos idiomas (en cualquier orden): {otros}")
        imprimir_cadena(traducir_cadena(args.texto, args.origen, orden))
    else:
        for orden in permutations(otros):
            imprimir_cadena(traducir_cadena(args.texto, args.origen, orden))
            print()


if __name__ == "__main__":
    main()
