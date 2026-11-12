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
import os
import shutil
import sys
import time
import zipfile

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

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
    ruta_parcial = ruta_zip.with_suffix(ruta_zip.suffix + ".part")

    reintentos = Retry(
        total=config.INTENTOS_DESCARGA,
        connect=config.INTENTOS_DESCARGA,
        status=config.INTENTOS_DESCARGA,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    sesion = requests.Session()
    sesion.mount("https://", HTTPAdapter(max_retries=reintentos))
    for intento in range(1, config.INTENTOS_DESCARGA + 1):
        try:
            with sesion.get(
                url,
                stream=True,
                timeout=(config.TIMEOUT_CONEXION, config.TIMEOUT_LECTURA),
            ) as respuesta:
                respuesta.raise_for_status()
                total = int(respuesta.headers.get("content-length", 0))
                descargados = 0
                with open(ruta_parcial, "wb") as f, tqdm(
                    total=total or None,
                    unit="B",
                    unit_scale=True,
                    desc=f"  {corpus} {par}",
                ) as barra:
                    for chunk in respuesta.iter_content(chunk_size=1 << 20):
                        if not chunk:
                            continue
                        f.write(chunk)
                        descargados += len(chunk)
                        barra.update(len(chunk))
                    f.flush()
                    os.fsync(f.fileno())
                if total and descargados != total:
                    raise IOError(f"descarga incompleta: {descargados} de {total} bytes")
            os.replace(ruta_parcial, ruta_zip)
            break
        except (OSError, requests.RequestException) as error:
            ruta_parcial.unlink(missing_ok=True)
            if intento == config.INTENTOS_DESCARGA:
                raise RuntimeError(
                    f"No se pudo descargar {corpus} {par} tras {intento} intentos: {error}"
                ) from error
            espera = min(2 ** (intento - 1), 8)
            print(f"  intento {intento} falló ({error}); reintentando en {espera}s")
            time.sleep(espera)
    sesion.close()

    # Se valida el CRC completo antes de reemplazar ningún corpus existente.
    nombres = {ruta.name for ruta in archivos_esperados(corpus, par)}
    temporales = []
    try:
        with zipfile.ZipFile(ruta_zip) as z:
            corrupto = z.testzip()
            if corrupto:
                raise zipfile.BadZipFile(f"CRC inválido en {corrupto}")
            disponibles = set(z.namelist())
            faltantes = nombres - disponibles
            if faltantes:
                raise zipfile.BadZipFile("faltan miembros: " + ", ".join(sorted(faltantes)))
            for nombre in sorted(nombres):
                destino = carpeta / nombre
                temporal = destino.with_suffix(destino.suffix + ".part")
                temporales.append(temporal)
                with z.open(nombre) as origen, open(temporal, "wb") as salida:
                    shutil.copyfileobj(origen, salida, length=1 << 20)
                    salida.flush()
                    os.fsync(salida.fileno())
            for temporal in temporales:
                os.replace(temporal, temporal.with_suffix(""))
    finally:
        for temporal in temporales:
            temporal.unlink(missing_ok=True)
        ruta_zip.unlink(missing_ok=True)

    problema = problema_de_alineacion(corpus, par)
    if problema:
        sys.exit(f"Error en {corpus} {par}: {problema}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Descarga los corpus de OPUS.")
    parser.add_argument("--corpus", choices=list(config.CORPUS), help="Descargar solo este corpus.")
    args = parser.parse_args(argv)

    corpus_elegidos = [args.corpus] if args.corpus else list(config.CORPUS)
    for corpus in corpus_elegidos:
        print(f"{corpus}:")
        for par in config.PARES:
            descargar(corpus, par)

    print(f"OK: corpus en {config.RUTA_CRUDOS}")


if __name__ == "__main__":
    main()
