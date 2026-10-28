"""
Descarga los corpus paralelos de OPUS a data/crudos/.

No hace falta cuenta ni token: OPUS es público. Los archivos pesan ~875 MB
comprimidos (~2.5 GB extraídos), por eso NO se suben a GitHub; cada quien los
baja con este script (en Colab tarda unos minutos).

Uso:
    python descargar_datos.py                    # todos los corpus de config.CORPUS
    python descargar_datos.py --corpus TED2020   # solo uno

Resultado:
    data/crudos/<corpus>/<par>/<corpus>.<par>.<idioma>   (ej. TED2020.de-en.de)
"""

import argparse
import sys
import zipfile

import requests
from tqdm import tqdm

import config


def contar_lineas(ruta):
    with open(ruta, "rb") as f:
        return sum(1 for _ in f)


def archivos_esperados(corpus, par):
    carpeta = config.RUTA_CRUDOS / corpus / par
    a, b = par.split("-")
    return [carpeta / f"{corpus}.{par}.{a}", carpeta / f"{corpus}.{par}.{b}"]


def problema_de_alineacion(corpus, par):
    """None si el par está completo; si no, un texto que explica qué falla.

    La API de OPUS reporta un conteo que a veces difiere por unas decenas de
    líneas del archivo real (ej. TED2020 de-en: 296,288 vs 296,354), así que
    se tolera un 1%. Lo que sí es obligatorio es que ambos idiomas tengan el
    MISMO número de líneas: si no, la línea i ya no es traducción de la línea i.
    """
    archivos = archivos_esperados(corpus, par)
    if not all(r.exists() for r in archivos):
        return "faltan archivos"
    n_a, n_b = (contar_lineas(r) for r in archivos)
    if n_a != n_b:
        return f"desalineado: {n_a} vs {n_b} líneas"
    esperadas = config.CORPUS[corpus]["alineaciones"][par]
    if abs(n_a - esperadas) > 0.01 * esperadas:
        return f"incompleto: {n_a} líneas, OPUS reporta {esperadas}"
    return None


def ya_descargado(corpus, par):
    return problema_de_alineacion(corpus, par) is None


def descargar(corpus, par):
    if ya_descargado(corpus, par):
        print(f"  {corpus} {par}: ya estaba, se omite")
        return

    version = config.CORPUS[corpus]["version"]
    url = config.URL_OPUS.format(corpus=corpus, version=version, par=par)
    carpeta = config.RUTA_CRUDOS / corpus / par
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_zip = carpeta / f"{par}.txt.zip"

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(ruta_zip, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                             desc=f"  {corpus} {par}") as barra:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                barra.update(len(chunk))

    # Solo se extraen los dos .txt de idiomas (el zip trae también README/LICENSE/.xml)
    nombres = {r.name for r in archivos_esperados(corpus, par)}
    with zipfile.ZipFile(ruta_zip) as z:
        for miembro in z.namelist():
            if miembro in nombres:
                z.extract(miembro, carpeta)
    ruta_zip.unlink()

    problema = problema_de_alineacion(corpus, par)
    if problema:
        sys.exit(f"Error en {corpus} {par}: {problema}")


def main():
    parser = argparse.ArgumentParser(description="Descarga los corpus de OPUS.")
    parser.add_argument("--corpus", choices=list(config.CORPUS), help="Descargar solo este corpus.")
    args = parser.parse_args()

    corpus_elegidos = [args.corpus] if args.corpus else list(config.CORPUS)
    for corpus in corpus_elegidos:
        print(f"{corpus}:")
        for par in config.PARES:
            descargar(corpus, par)

    print(f"OK: corpus en {config.RUTA_CRUDOS}")


if __name__ == "__main__":
    main()
