"""Diagnóstico reproducible de instalación, datos y modelo."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version

import config


PAQUETES = {
    "requests": None,
    "tqdm": None,
    "pandas": None,
    "torch": None,
    "transformers": "5.21.0",
    "datasets": None,
    "sentencepiece": None,
    "sacrebleu": None,
    "accelerate": None,
    "huggingface_hub": None,
    "safetensors": None,
}


def resultado(nombre, estado, detalle):
    return {"comprobacion": nombre, "estado": estado, "detalle": detalle}


def sha256_archivo(ruta):
    digest = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()


def ejecutar(sin_red=False, cargar_modelo=False):
    resultados = []
    python_ok = sys.version_info >= (3, 10)
    resultados.append(resultado("Python", "ok" if python_ok else "error", sys.version.split()[0]))

    for paquete, exacta in PAQUETES.items():
        try:
            instalada = version(paquete)
            coincide = exacta is None or instalada == exacta
            detalle = instalada if coincide or exacta is None else f"{instalada}; se requiere {exacta}"
            resultados.append(resultado(f"Paquete {paquete}", "ok" if coincide else "error", detalle))
        except PackageNotFoundError:
            resultados.append(resultado(f"Paquete {paquete}", "error", "no instalado"))

    try:
        import torch

        if torch.cuda.is_available():
            disp = f"cuda ({torch.cuda.get_device_name(0)})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            disp = "mps (Apple Silicon)"
        else:
            disp = "cpu"
        resultados.append(resultado("Aceleración", "ok", disp))
    except ImportError:
        pass

    libres = shutil.disk_usage(config.RAIZ).free
    resultados.append(
        resultado(
            "Espacio libre",
            "ok" if libres >= 4 * 1024**3 else "aviso",
            f"{libres / 1024**3:.1f} GiB disponibles",
        )
    )

    for nombre, esperadas in (("validacion.tsv", config.N_VALIDACION), ("prueba.tsv", config.N_PRUEBA)):
        ruta = config.RUTA_PROCESADOS / nombre
        if not ruta.is_file():
            resultados.append(resultado(nombre, "error", "archivo ausente"))
            continue
        with open(ruta, encoding="utf-8", newline="") as f:
            lector = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
            filas = sum(1 for _ in lector)
            columnas_ok = lector.fieldnames == config.IDIOMAS
        estado = "ok" if filas == esperadas and columnas_ok else "error"
        resultados.append(resultado(nombre, estado, f"{filas} filas; columnas {lector.fieldnames}"))

    ruta_resumen = config.RUTA_PROCESADOS / "resumen.json"
    if ruta_resumen.is_file():
        try:
            resumen = json.loads(ruta_resumen.read_text(encoding="utf-8"))
            hashes = resumen.get("sha256", {})
            for nombre in ("validacion.tsv", "prueba.tsv"):
                ruta = config.RUTA_PROCESADOS / nombre
                esperado = hashes.get(nombre)
                if esperado and ruta.is_file():
                    actual = sha256_archivo(ruta)
                    resultados.append(
                        resultado(
                            f"Integridad {nombre}",
                            "ok" if actual == esperado else "error",
                            "SHA-256 correcto" if actual == esperado else "SHA-256 no coincide",
                        )
                    )
        except (OSError, json.JSONDecodeError) as error:
            resultados.append(resultado("resumen.json", "error", str(error)))

    local = config.RUTA_MODELO_DIR
    if local.is_dir():
        pesos = any((local / nombre).is_file() for nombre in ("model.safetensors", "pytorch_model.bin"))
        resultados.append(resultado("Modelo local", "ok" if pesos else "error", str(local)))
    elif sin_red:
        resultados.append(resultado("Modelo", "aviso", "sin copia local; red no comprobada"))
    else:
        try:
            import requests

            respuesta = requests.get(
                f"https://huggingface.co/api/models/{config.HF_REPO_MODELO}/revision/{config.HF_REVISION_MODELO}",
                timeout=(config.TIMEOUT_CONEXION, config.TIMEOUT_LECTURA),
            )
            respuesta.raise_for_status()
            resultados.append(
                resultado("Modelo remoto", "ok", f"{config.HF_REPO_MODELO}@{config.HF_REVISION_MODELO[:12]}")
            )
        except Exception as error:
            resultados.append(resultado("Modelo remoto", "error", f"{type(error).__name__}: {error}"))

    if cargar_modelo:
        try:
            from motor_traduccion import MotorTraduccion

            info = MotorTraduccion().cargar()
            resultados.append(resultado("Carga completa", "ok", json.dumps(info.como_dict(), ensure_ascii=False)))
        except Exception as error:
            resultados.append(resultado("Carga completa", "error", f"{type(error).__name__}: {error}"))
    return resultados


def main(argv=None):
    parser = argparse.ArgumentParser(description="Comprueba instalación, datos y disponibilidad del modelo.")
    parser.add_argument("--sin-red", action="store_true")
    parser.add_argument("--cargar-modelo", action="store_true", help="Descarga/carga también los pesos completos.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    resultados = ejecutar(args.sin_red, args.cargar_modelo)
    if args.json:
        print(json.dumps(resultados, ensure_ascii=False, indent=2))
    else:
        simbolos = {"ok": "✓", "aviso": "!", "error": "✗"}
        for item in resultados:
            print(f"{simbolos[item['estado']]} {item['comprobacion']}: {item['detalle']}")
    return 1 if any(item["estado"] == "error" for item in resultados) else 0


if __name__ == "__main__":
    raise SystemExit(main())
