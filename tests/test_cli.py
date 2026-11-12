import subprocess
import sys
import unittest


class PruebasCLI(unittest.TestCase):
    def ejecutar(self, *argumentos):
        return subprocess.run(
            [sys.executable, "Traductor.py", *argumentos],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_ayuda_predecir_muestra_opciones_reales(self):
        resultado = self.ejecutar("predecir", "--help")
        self.assertEqual(resultado.returncode, 0)
        self.assertIn("--todas-las-cadenas", resultado.stdout)
        self.assertIn("--permitir-modelo-base", resultado.stdout)

    def test_comando_sin_argumentos_rechaza_extras(self):
        resultado = self.ejecutar("preparar", "--desconocido")
        self.assertNotEqual(resultado.returncode, 0)
        self.assertIn("no acepta argumentos", resultado.stderr)


if __name__ == "__main__":
    unittest.main()
