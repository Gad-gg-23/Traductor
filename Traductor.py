"""
Punto de entrada único del traductor multilingüe (es, en, ro, de).

    python Traductor.py descargar   -> baja los corpus de OPUS a data/crudos/
    python Traductor.py preparar    -> limpia y arma entrenamiento/validación/prueba
    python Traductor.py entrenar    -> afina M2M100 sobre los datos preparados
    python Traductor.py evaluar     -> BLEU/chrF directo y en cadena (round-trip) sobre prueba.tsv
    python Traductor.py predecir --texto "..." --origen es --todas-las-cadenas
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


def comando_entrenar():
    import entrenar

    entrenar.entrenar()


def comando_evaluar():
    import evaluar

    evaluar.main()


def comando_predecir(args_restantes):
    import predecir

    sys.argv = [sys.argv[0]] + args_restantes
    predecir.main()


def main():
    parser = argparse.ArgumentParser(description="Traductor multilingüe español/inglés/rumano/alemán.")
    subparsers = parser.add_subparsers(dest="comando", required=True)
    subparsers.add_parser("descargar", help="Descarga los corpus de OPUS.")
    subparsers.add_parser("preparar", help="Limpia los corpus y arma los conjuntos.")
    subparsers.add_parser("entrenar", help="Afina M2M100 sobre los datos preparados.")
    subparsers.add_parser("evaluar", help="BLEU/chrF directo y en cadena sobre prueba.tsv.")
    subparsers.add_parser("predecir", help="Traduce una oración (directa o en cadena).")

    args, restantes = parser.parse_known_args()

    if args.comando == "descargar":
        comando_descargar(restantes)
    elif args.comando == "preparar":
        comando_preparar()
    elif args.comando == "entrenar":
        comando_entrenar()
    elif args.comando == "evaluar":
        comando_evaluar()
    elif args.comando == "predecir":
        comando_predecir(restantes)


if __name__ == "__main__":
    main()
