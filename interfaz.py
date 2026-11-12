"""Interfaz gráfica web local, sin framework adicional.

Uso:
    python Traductor.py interfaz
    python interfaz.py --no-abrir
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import config
from motor_traduccion import ErrorTraductor, MotorTraduccion

LOGGER = logging.getLogger("traductor.web")
CARPETA_WEB = config.RAIZ / "web"
ARCHIVOS_PUBLICOS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/favicon.svg": ("favicon.svg", "image/svg+xml"),
}


def procesar_traduccion(motor: MotorTraduccion, datos: dict) -> dict:
    if not isinstance(datos, dict):
        raise ErrorTraductor("La petición debe ser un objeto JSON.")
    modo = datos.get("modo")
    texto = datos.get("texto", "")
    origen = datos.get("origen", "")
    perfil = datos.get("perfil", config.PERFIL_GENERACION_PREDETERMINADO)

    if modo == "directa":
        destino = datos.get("destino", "")
        traduccion = motor.traducir(texto, origen, destino, perfil)
        return {
            "modo": modo,
            "entrada": texto,
            "origen": origen,
            "destino": destino,
            "traduccion": traduccion,
            "perfil": perfil,
            "modelo": motor.informacion.como_dict(),
        }
    if modo == "cadena":
        orden = datos.get("orden")
        if not isinstance(orden, list):
            raise ErrorTraductor("El orden de la cadena debe ser una lista de idiomas.")
        return {
            "modo": modo,
            "origen": origen,
            "orden": orden,
            "pasos": motor.traducir_cadena(texto, origen, orden, perfil),
            "perfil": perfil,
            "modelo": motor.informacion.como_dict(),
        }
    if modo == "todas":
        return {
            "modo": modo,
            "origen": origen,
            "cadenas": motor.traducir_todas_las_cadenas(texto, origen, perfil),
            "perfil": perfil,
            "modelo": motor.informacion.como_dict(),
        }
    raise ErrorTraductor("El modo debe ser directa, cadena o todas.")


def crear_manejador(motor: MotorTraduccion):
    class Manejador(BaseHTTPRequestHandler):
        server_version = "TraductorLocal/1.0"

        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def log_message(self, formato, *args):
            LOGGER.info("%s - %s", self.address_string(), formato % args)

        def _cabeceras_seguridad(self):
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self'; script-src 'self'; "
                "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'",
            )
            self.send_header("Cache-Control", "no-store")

        def _enviar(self, estado: int, cuerpo: bytes, tipo: str):
            self.send_response(estado)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(cuerpo)))
            self._cabeceras_seguridad()
            self.end_headers()
            try:
                self.wfile.write(cuerpo)
            except BrokenPipeError:
                LOGGER.info("El navegador cerró la conexión antes de recibir la respuesta.")

        def _json(self, estado: int, datos: dict):
            cuerpo = json.dumps(datos, ensure_ascii=False).encode("utf-8")
            self._enviar(estado, cuerpo, "application/json; charset=utf-8")

        def _leer_json(self) -> dict:
            tipo = self.headers.get_content_type()
            if tipo != "application/json":
                raise ErrorTraductor("Content-Type debe ser application/json.")
            try:
                longitud = int(self.headers.get("Content-Length", "0"))
            except ValueError as error:
                raise ErrorTraductor("Content-Length no es válido.") from error
            if longitud <= 0:
                raise ErrorTraductor("La petición está vacía.")
            if longitud > config.MAX_BYTES_PETICION_WEB:
                raise ErrorTraductor("La petición supera el tamaño máximo permitido.")
            try:
                return json.loads(self.rfile.read(longitud).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ErrorTraductor("El cuerpo no contiene JSON UTF-8 válido.") from error

        def do_GET(self):
            ruta = urlparse(self.path).path
            if ruta == "/api/status":
                self._json(HTTPStatus.OK, {"modelo": motor.informacion.como_dict()})
                return
            publico = ARCHIVOS_PUBLICOS.get(ruta)
            if publico is None:
                self._json(HTTPStatus.NOT_FOUND, {"error": "Ruta no encontrada."})
                return
            nombre, tipo = publico
            ruta_archivo = CARPETA_WEB / nombre
            try:
                cuerpo = ruta_archivo.read_bytes()
            except OSError:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Falta un recurso de la interfaz."})
                return
            self._enviar(HTTPStatus.OK, cuerpo, tipo)

        def do_POST(self):
            ruta = urlparse(self.path).path
            try:
                datos = self._leer_json()
                if ruta == "/api/cargar":
                    resultado = {"modelo": motor.cargar().como_dict()}
                elif ruta == "/api/traducir":
                    resultado = procesar_traduccion(motor, datos)
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "Ruta no encontrada."})
                    return
                self._json(HTTPStatus.OK, resultado)
            except ErrorTraductor as error:
                self._json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": str(error), "modelo": motor.informacion.como_dict()},
                )
            except Exception:
                LOGGER.exception("Fallo inesperado atendiendo %s", ruta)
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "Ocurrió un fallo interno. Revisa la terminal para ver el detalle."},
                )

    return Manejador


class ServidorTraductor(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def construir_parser():
    parser = argparse.ArgumentParser(description="Abre la interfaz gráfica local del traductor.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--puerto", type=int, default=7860)
    parser.add_argument("--no-abrir", action="store_true", help="No abre el navegador automáticamente.")
    parser.add_argument("--precargar", action="store_true", help="Carga el modelo antes de abrir la interfaz.")
    parser.add_argument("--permitir-red", action="store_true", help="Permite escuchar fuera de localhost.")
    parser.add_argument("--permitir-modelo-base", action="store_true")
    parser.add_argument("--dispositivo", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser


def main(argv=None) -> int:
    args = construir_parser().parse_args(argv)
    if not 1 <= args.puerto <= 65535:
        raise SystemExit("El puerto debe estar entre 1 y 65535.")
    if args.host not in {"127.0.0.1", "localhost"} and not args.permitir_red:
        raise SystemExit("Para exponer la interfaz en red debes agregar --permitir-red.")
    faltantes = [nombre for nombre, _ in ARCHIVOS_PUBLICOS.values() if not (CARPETA_WEB / nombre).is_file()]
    if faltantes:
        raise SystemExit("Faltan recursos web: " + ", ".join(sorted(set(faltantes))))

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    LOGGER.setLevel(logging.INFO)
    motor = MotorTraduccion(args.permitir_modelo_base, args.dispositivo)
    if args.precargar:
        try:
            info = motor.cargar()
            print(f"Modelo listo: {info.fuente} en {info.dispositivo}")
        except ErrorTraductor as error:
            raise SystemExit(f"No se pudo precargar el modelo: {error}") from error

    try:
        servidor = ServidorTraductor((args.host, args.puerto), crear_manejador(motor))
    except OSError as error:
        raise SystemExit(
            f"No se pudo abrir {args.host}:{args.puerto}. Prueba otro puerto con --puerto. Detalle: {error}"
        ) from error
    host_visible = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    url = f"http://{host_visible}:{servidor.server_port}"
    print(f"Interfaz disponible en {url}")
    print("Presiona Ctrl+C para detenerla.")
    if not args.no_abrir:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        servidor.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nCerrando interfaz...")
    finally:
        servidor.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
