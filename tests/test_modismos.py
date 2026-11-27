import csv
import tempfile
import unittest
from pathlib import Path

import config
from modismos import cargar_modismos, expandir_pares_entrenamiento


class PruebasModismos(unittest.TestCase):
    def _escribir_tsv(self, ruta, filas):
        with open(ruta, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", *config.IDIOMAS], delimiter="\t",
                                quoting=csv.QUOTE_NONE, escapechar="\\", lineterminator="\n")
            w.writeheader()
            w.writerows(filas)

    def test_carga_valida_columnas(self):
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "modismos.tsv"
            self._escribir_tsv(ruta, [{"id": "x1", "de": "A", "en": "B", "es": "C", "ro": "D"}])
            filas = cargar_modismos(ruta)
            self.assertEqual(filas, [{"id": "x1", "de": "A", "en": "B", "es": "C", "ro": "D"}])

    def test_rechaza_columnas_faltantes(self):
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "modismos.tsv"
            with open(ruta, "w", encoding="utf-8", newline="") as f:
                f.write("id\tde\ten\tes\n")
                f.write("x1\tA\tB\tC\n")
            with self.assertRaises(ValueError):
                cargar_modismos(ruta)

    def test_expande_las_6_direcciones_con_ambos_sentidos(self):
        filas = [{"id": "x1", "de": "A", "en": "B", "es": "C", "ro": "D"}]
        df = expandir_pares_entrenamiento(filas, repeticiones=1)
        # 6 pares x 2 sentidos = 12 filas por cada modismo
        self.assertEqual(len(df), 12)
        direcciones = set(zip(df.origen, df.destino))
        esperadas = set()
        for par in config.PARES:
            a, b = par.split("-")
            esperadas.add((a, b))
            esperadas.add((b, a))
        self.assertEqual(direcciones, esperadas)
        fila = df[(df.origen == "de") & (df.destino == "en")].iloc[0]
        self.assertEqual(fila.texto_origen, "A")
        self.assertEqual(fila.texto_destino, "B")

    def test_repeticiones_multiplica_el_total(self):
        filas = [{"id": "x1", "de": "A", "en": "B", "es": "C", "ro": "D"}]
        df = expandir_pares_entrenamiento(filas, repeticiones=5)
        self.assertEqual(len(df), 12 * 5)

    def test_rechaza_repeticiones_invalidas(self):
        filas = [{"id": "x1", "de": "A", "en": "B", "es": "C", "ro": "D"}]
        with self.assertRaises(ValueError):
            expandir_pares_entrenamiento(filas, repeticiones=0)


if __name__ == "__main__":
    unittest.main()
