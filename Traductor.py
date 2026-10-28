"""
Punto de entrada único del traductor multilingüe (es, en, ro, de).

    python Traductor.py descargar   -> baja los corpus de OPUS a data/crudos/
    python Traductor.py preparar    -> limpia y arma entrenamiento/validación/prueba
"""

import argparse
import sys


def comando_descargar(args_restantes):
    import descargar_datos

    sys.argv = [sys.argv[0]] + args_restantes
    descargar_datos.main()


def comando_preparar():
    import preparar_datos

    preparar_datos.main()


def main():
    parser = argparse.ArgumentParser(description="Traductor multilingüe español/inglés/rumano/alemán.")
    subparsers = parser.add_subparsers(dest="comando", required=True)
    subparsers.add_parser("descargar", help="Descarga los corpus de OPUS.")
    subparsers.add_parser("preparar", help="Limpia los corpus y arma los conjuntos.")

    args, restantes = parser.parse_known_args()

    if args.comando == "descargar":
        comando_descargar(restantes)
    elif args.comando == "preparar":
        comando_preparar()


if __name__ == "__main__":
    main()
