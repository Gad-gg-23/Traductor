"""Punto de entrada único del proyecto."""

from __future__ import annotations

import argparse
import sys


DESCRIPCIONES = {
    "interfaz": "Abre la interfaz gráfica local en el navegador.",
    "predecir": "Traduce una oración directa o en cadena.",
    "diagnostico": "Comprueba dependencias, datos y modelo.",
    "descargar": "Descarga y verifica los corpus de OPUS.",
    "preparar": "Limpia los corpus y construye los conjuntos.",
    "entrenar": "Afina M2M100 sobre los datos preparados.",
    "evaluar": "Calcula BLEU/chrF directo y en cadena.",
}


def construir_parser_raiz() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Traductor multilingüe robusto: español, inglés, rumano y alemán.",
        epilog="Usa 'python Traductor.py <comando> --help' para ver las opciones de un comando.",
    )
    subparsers = parser.add_subparsers(dest="comando", metavar="comando")
    for comando, descripcion in DESCRIPCIONES.items():
        subparsers.add_parser(comando, add_help=False, help=descripcion)
    return parser


def _sin_argumentos(comando: str, argumentos: list[str]):
    if argumentos:
        raise SystemExit(f"El comando '{comando}' no acepta argumentos: {' '.join(argumentos)}")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = construir_parser_raiz()
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0

    comando, restantes = argv[0], argv[1:]
    if comando not in DESCRIPCIONES:
        parser.error(f"comando desconocido: {comando}")

    if comando == "interfaz":
        import interfaz

        return interfaz.main(restantes)
    if comando == "predecir":
        import predecir

        return predecir.main(restantes)
    if comando == "diagnostico":
        import diagnostico

        return diagnostico.main(restantes)
    if comando == "descargar":
        import descargar_datos

        return descargar_datos.main(restantes) or 0
    if comando == "preparar":
        _sin_argumentos(comando, restantes)
        import preparar_datos

        preparar_datos.main()
        return 0
    if comando == "entrenar":
        _sin_argumentos(comando, restantes)
        import entrenar

        entrenar.entrenar()
        return 0
    if comando == "evaluar":
        _sin_argumentos(comando, restantes)
        import evaluar

        evaluar.main()
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
