import csv
import tempfile
import unittest
from pathlib import Path

import config
from muestreo import muestrear_archivo_direcciones
from preparar_datos import normalizar, par_valido


class PruebasPreparacion(unittest.TestCase):
    def test_normaliza_unicode_espacios_y_tabs(self):
        self.assertEqual(normalizar("  café\t\n  con   leche  "), "café con leche")

    def test_rechaza_par_identico(self):
        self.assertFalse(par_valido("Hola mundo", "hola mundo"))

    def test_muestreo_reservorio_equilibra_12_direcciones(self):
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "entrenamiento.tsv"
            with open(ruta, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
                writer.writerow(["idioma_a", "idioma_b", "texto_a", "texto_b"])
                for par in config.PARES:
                    a, b = par.split("-")
                    for indice in range(8):
                        writer.writerow([a, b, f"{a}-{indice}", f"{b}-{indice}"])
            muestra = muestrear_archivo_direcciones(ruta, 4, 42)
            conteos = muestra.groupby(["origen", "destino"]).size()
            self.assertEqual(len(conteos), 12)
            self.assertTrue((conteos == 4).all())
            self.assertEqual(len(muestra), 48)


if __name__ == "__main__":
    unittest.main()
