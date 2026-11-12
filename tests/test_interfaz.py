import http.client
import json
import threading
import unittest

from interfaz import ServidorTraductor, crear_manejador, procesar_traduccion
from motor_traduccion import InformacionModelo


class MotorSimulado:
    informacion = InformacionModelo(estado="listo", fuente="simulado", dispositivo="cpu")

    def traducir(self, texto, origen, destino, perfil=None):
        return f"{destino}:{texto}"

    def traducir_cadena(self, texto, origen, orden, perfil=None):
        return [{"idioma": origen, "texto": texto}] + [
            {"idioma": idioma, "texto": f"{idioma}:{texto}"} for idioma in [*orden, origen]
        ]

    def traducir_todas_las_cadenas(self, texto, origen, perfil=None):
        return [{"orden": ["de", "en", "ro"], "pasos": self.traducir_cadena(texto, origen, ["de", "en", "ro"], perfil)}]


class PruebasInterfaz(unittest.TestCase):
    def setUp(self):
        self.motor = MotorSimulado()

    def test_peticion_directa(self):
        resultado = procesar_traduccion(
            self.motor, {"modo": "directa", "texto": "Hola", "origen": "es", "destino": "en"}
        )
        self.assertEqual(resultado["traduccion"], "en:Hola")
        self.assertEqual(resultado["modelo"]["fuente"], "simulado")

    def test_peticion_cadena(self):
        resultado = procesar_traduccion(
            self.motor,
            {"modo": "cadena", "texto": "Hola", "origen": "es", "orden": ["en", "ro", "de"]},
        )
        self.assertEqual(len(resultado["pasos"]), 5)

    def test_peticion_todas_las_cadenas(self):
        resultado = procesar_traduccion(
            self.motor,
            {"modo": "todas", "texto": "Hola", "origen": "es"},
        )
        self.assertEqual(resultado["modo"], "todas")
        self.assertEqual(len(resultado["cadenas"]), 1)

    def test_rechaza_modo_invalido(self):
        from motor_traduccion import ErrorTraductor

        with self.assertRaises(ErrorTraductor):
            procesar_traduccion(self.motor, {"modo": "inexistente", "texto": "Hola", "origen": "es"})


class PruebasServidorHTTP(unittest.TestCase):
    def setUp(self):
        self.servidor = ServidorTraductor(("127.0.0.1", 0), crear_manejador(MotorSimulado()))
        self.hilo = threading.Thread(target=self.servidor.serve_forever, daemon=True)
        self.hilo.start()
        self.conexion = http.client.HTTPConnection("127.0.0.1", self.servidor.server_port, timeout=3)

    def tearDown(self):
        self.conexion.close()
        self.servidor.shutdown()
        self.servidor.server_close()
        self.hilo.join(timeout=3)

    def test_estado_incluye_cabeceras_de_seguridad(self):
        self.conexion.request("GET", "/api/status")
        respuesta = self.conexion.getresponse()
        datos = json.loads(respuesta.read())
        self.assertEqual(respuesta.status, 200)
        self.assertEqual(datos["modelo"]["fuente"], "simulado")
        self.assertEqual(respuesta.getheader("X-Frame-Options"), "DENY")

    def test_rechaza_json_invalido(self):
        self.conexion.request(
            "POST",
            "/api/traducir",
            body=b"no-json",
            headers={"Content-Type": "application/json", "Content-Length": "7"},
        )
        respuesta = self.conexion.getresponse()
        datos = json.loads(respuesta.read())
        self.assertEqual(respuesta.status, 400)
        self.assertIn("JSON", datos["error"])


if __name__ == "__main__":
    unittest.main()
