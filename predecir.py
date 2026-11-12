"""CLI de inferencia del traductor multilingüe."""

from __future__ import annotations

import argparse
import json
import sys

import config
from motor_traduccion import ErrorTraductor, MotorTraduccion

_motor: MotorTraduccion | None = None


def obtener_motor(permitir_modelo_base: bool = False, dispositivo: str = "auto") -> MotorTraduccion:
    global _motor
    if _motor is None:
        _motor = MotorTraduccion(permitir_modelo_base=permitir_modelo_base, dispositivo=dispositivo)
    return _motor


def traducir(texto, origen, destino):
    """Compatibilidad con quienes importaban la función de versiones anteriores."""
    return obtener_motor().traducir(texto, origen, destino)


def traducir_cadena(texto, origen, orden):
    pasos = obtener_motor().traducir_cadena(texto, origen, orden)
    return [(paso["idioma"], paso["texto"]) for paso in pasos]


def imprimir_cadena(pasos):
    normalizados = [
        paso if isinstance(paso, dict) else {"idioma": paso[0], "texto": paso[1]}
        for paso in pasos
    ]
    print("  " + " -> ".join(paso["idioma"] for paso in normalizados))
    for paso in normalizados:
        print(f"    [{paso['idioma']}] {paso['texto']}")


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Traductor.py predecir",
        description="Traduce con el modelo M2M100 afinado y validado del equipo.",
    )
    parser.add_argument("--texto", help="Texto que se va a traducir.")
    parser.add_argument("--origen", choices=config.IDIOMAS, help="Idioma del texto de entrada.")
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--destino", choices=config.IDIOMAS, help="Traducción directa.")
    grupo.add_argument("--cadena", help="Los otros tres idiomas separados por coma; ej. en,ro,de.")
    grupo.add_argument(
        "--todas-las-cadenas", action="store_true", help="Ejecuta las seis permutaciones posibles."
    )
    parser.add_argument("--info-modelo", action="store_true", help="Carga y muestra el modelo en uso.")
    parser.add_argument("--json", action="store_true", help="Entrega un resultado JSON para automatización.")
    parser.add_argument(
        "--dispositivo", choices=["auto", "cpu", "cuda", "mps"], default="auto"
    )
    parser.add_argument(
        "--perfil",
        choices=list(config.PERFILES_GENERACION),
        default=config.PERFIL_GENERACION_PREDETERMINADO,
        help="rapida prioriza latencia; maxima explora más candidatos.",
    )
    parser.add_argument(
        "--permitir-modelo-base",
        action="store_true",
        help="Autoriza explícitamente usar M2M100 sin afinar si falla el modelo del equipo.",
    )
    return parser


def _validar_argumentos(parser, args):
    if args.info_modelo and not args.texto:
        return
    if not args.texto:
        parser.error("--texto es obligatorio para traducir")
    if not args.origen:
        parser.error("--origen es obligatorio para traducir")
    if not (args.destino or args.cadena or args.todas_las_cadenas):
        parser.error("elige --destino, --cadena o --todas-las-cadenas")


def main(argv=None) -> int:
    parser = construir_parser()
    args = parser.parse_args(argv)
    _validar_argumentos(parser, args)
    motor = MotorTraduccion(args.permitir_modelo_base, args.dispositivo)

    try:
        if args.info_modelo and not args.texto:
            resultado = {"modelo": motor.cargar().como_dict()}
        elif args.destino:
            traduccion = motor.traducir(args.texto, args.origen, args.destino, args.perfil)
            resultado = {
                "modo": "directa",
                "origen": args.origen,
                "destino": args.destino,
                "entrada": args.texto,
                "traduccion": traduccion,
                "perfil": args.perfil,
                "modelo": motor.informacion.como_dict(),
            }
        elif args.cadena:
            orden = [idioma.strip() for idioma in args.cadena.split(",")]
            pasos = motor.traducir_cadena(args.texto, args.origen, orden, args.perfil)
            resultado = {
                "modo": "cadena",
                "origen": args.origen,
                "orden": orden,
                "pasos": pasos,
                "perfil": args.perfil,
                "modelo": motor.informacion.como_dict(),
            }
        else:
            cadenas = motor.traducir_todas_las_cadenas(args.texto, args.origen, args.perfil)
            resultado = {
                "modo": "todas_las_cadenas",
                "origen": args.origen,
                "cadenas": cadenas,
                "perfil": args.perfil,
                "modelo": motor.informacion.como_dict(),
            }
    except ErrorTraductor as error:
        if args.json:
            print(json.dumps({"error": str(error)}, ensure_ascii=False))
        else:
            print(f"Error: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(resultado, ensure_ascii=False, indent=2))
    elif resultado.get("modo") == "directa":
        print(resultado["traduccion"])
    elif resultado.get("modo") == "cadena":
        imprimir_cadena(resultado["pasos"])
    elif resultado.get("modo") == "todas_las_cadenas":
        for cadena in resultado["cadenas"]:
            imprimir_cadena(cadena["pasos"])
            print()
    else:
        print(json.dumps(resultado, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
