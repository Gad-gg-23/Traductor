import unittest
from unittest.mock import Mock

from motor_traduccion import (
    ErrorConfiguracion,
    ErrorTraductor,
    InformacionModelo,
    MotorTraduccion,
    limpiar_texto,
    validar_direccion,
    validar_orden,
    validar_perfil,
)


class PruebasValidacion(unittest.TestCase):
    def test_limpia_bordes_sin_destruir_contenido(self):
        self.assertEqual(limpiar_texto("  Hola\nmundo  "), "Hola\nmundo")

    def test_rechaza_texto_vacio(self):
        with self.assertRaises(ErrorConfiguracion):
            limpiar_texto("   ")

    def test_rechaza_misma_direccion(self):
        with self.assertRaises(ErrorConfiguracion):
            validar_direccion("es", "es")

    def test_valida_los_tres_idiomas_de_cadena(self):
        self.assertEqual(validar_orden("es", ["en", "ro", "de"]), ("en", "ro", "de"))
        with self.assertRaises(ErrorConfiguracion):
            validar_orden("es", ["en", "en", "de"])

    def test_valida_perfil_de_velocidad(self):
        self.assertEqual(validar_perfil(None), "equilibrada")
        with self.assertRaises(ErrorConfiguracion):
            validar_perfil("inventada")


class PruebasMotorSinModelo(unittest.TestCase):
    def setUp(self):
        self.motor = MotorTraduccion()
        self.motor._generar = Mock(
            side_effect=lambda textos, origen, destino, perfil=None: [f"{destino}:{t}" for t in textos]
        )
        self.motor._info = InformacionModelo(estado="listo", fuente="simulado", dispositivo="cpu")

    def test_traduccion_directa(self):
        self.assertEqual(self.motor.traducir("Hola", "es", "en"), "en:Hola")

    def test_cadena_incluye_original_y_regreso(self):
        pasos = self.motor.traducir_cadena("Hola", "es", ["en", "ro", "de"])
        self.assertEqual([p["idioma"] for p in pasos], ["es", "en", "ro", "de", "es"])
        self.assertEqual(len(pasos), 5)

    def test_rechaza_salida_vacia(self):
        self.motor._generar = Mock(return_value=[""])
        with self.assertRaises(ErrorTraductor):
            self.motor.traducir("Hola", "es", "en")

    def test_seis_rutas_reutilizan_primeros_saltos(self):
        cadenas = self.motor.traducir_todas_las_cadenas("Hola", "es", "rapida")
        self.assertEqual(len(cadenas), 6)
        self.assertEqual(self.motor._generar.call_count, 21)


if __name__ == "__main__":
    unittest.main()
