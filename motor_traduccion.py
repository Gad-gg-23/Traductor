"""Núcleo robusto y reutilizable de inferencia.

Este módulo no importa PyTorch ni Transformers al cargarse. Así, la ayuda de
la CLI, las pruebas y la interfaz pueden arrancar y explicar qué dependencia
falta en vez de fallar con un traceback antes de mostrar información útil.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path
from typing import Iterable, Sequence

import config

LOGGER = logging.getLogger(__name__)


class ErrorTraductor(RuntimeError):
    """Error esperado que puede mostrarse directamente a una persona."""


class ErrorConfiguracion(ErrorTraductor):
    """La instalación o un argumento no permiten continuar."""


class ErrorCargaModelo(ErrorTraductor):
    """No fue posible cargar un modelo válido."""


@dataclass(frozen=True)
class InformacionModelo:
    estado: str
    fuente: str | None = None
    revision: str | None = None
    dispositivo: str | None = None
    modelo_base: bool = False
    advertencia: str | None = None

    def como_dict(self) -> dict:
        return asdict(self)


def limpiar_texto(texto: str) -> str:
    if not isinstance(texto, str):
        raise ErrorConfiguracion("El texto debe ser una cadena de caracteres.")
    texto = texto.strip()
    if not texto:
        raise ErrorConfiguracion("Escribe una frase antes de traducir.")
    if "\x00" in texto:
        raise ErrorConfiguracion("El texto contiene un carácter nulo no permitido.")
    if len(texto) > config.MAX_CARACTERES_ENTRADA:
        raise ErrorConfiguracion(
            f"El texto supera el límite de {config.MAX_CARACTERES_ENTRADA:,} caracteres."
        )
    return texto


def validar_idioma(idioma: str, nombre: str = "idioma") -> str:
    if idioma not in config.IDIOMAS:
        permitidos = ", ".join(config.IDIOMAS)
        raise ErrorConfiguracion(f"{nombre} debe ser uno de: {permitidos}.")
    return idioma


def validar_direccion(origen: str, destino: str) -> tuple[str, str]:
    validar_idioma(origen, "El idioma de origen")
    validar_idioma(destino, "El idioma de destino")
    if origen == destino:
        raise ErrorConfiguracion("El idioma de destino debe ser diferente al de origen.")
    return origen, destino


def validar_orden(origen: str, orden: Sequence[str]) -> tuple[str, ...]:
    validar_idioma(origen, "El idioma de origen")
    otros = sorted(i for i in config.IDIOMAS if i != origen)
    orden_limpio = tuple(i.strip() for i in orden)
    if sorted(orden_limpio) != otros:
        raise ErrorConfiguracion(
            "La cadena debe incluir exactamente una vez los otros tres idiomas: "
            + ", ".join(otros)
            + "."
        )
    return orden_limpio


def validar_perfil(perfil: str | None) -> str:
    perfil = perfil or config.PERFIL_GENERACION_PREDETERMINADO
    if perfil not in config.PERFILES_GENERACION:
        raise ErrorConfiguracion(
            "El perfil de generación debe ser uno de: "
            + ", ".join(config.PERFILES_GENERACION)
            + "."
        )
    return perfil


def _modelo_local_completo(ruta: Path) -> bool:
    if not ruta.is_dir() or not (ruta / "config.json").is_file():
        return False
    pesos = (
        ruta / "model.safetensors",
        ruta / "pytorch_model.bin",
        ruta / "model.safetensors.index.json",
        ruta / "pytorch_model.bin.index.json",
    )
    return any(p.is_file() for p in pesos)


class MotorTraduccion:
    """Carga perezosa, validación y traducción segura para CLI, web y evaluación."""

    def __init__(self, permitir_modelo_base: bool = False, dispositivo: str = "auto"):
        if dispositivo not in {"auto", "cpu", "cuda", "mps"}:
            raise ErrorConfiguracion("El dispositivo debe ser auto, cpu, cuda o mps.")
        self.permitir_modelo_base = permitir_modelo_base
        self.dispositivo_solicitado = dispositivo
        self._modelo = None
        self._tokenizer = None
        self._torch = None
        self._candado = threading.RLock()
        self._cache: OrderedDict[tuple[str, str, str, str], str] = OrderedDict()
        self._info = InformacionModelo(estado="sin_cargar")

    @property
    def informacion(self) -> InformacionModelo:
        return self._info

    @property
    def cargado(self) -> bool:
        return self._modelo is not None and self._tokenizer is not None

    def _elegir_dispositivo(self, torch) -> str:
        solicitado = self.dispositivo_solicitado
        if solicitado == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if solicitado == "cuda" and not torch.cuda.is_available():
            raise ErrorConfiguracion("Se pidió CUDA, pero PyTorch no detecta una GPU CUDA.")
        if solicitado == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise ErrorConfiguracion("Se pidió MPS, pero PyTorch no detecta Apple Silicon compatible.")
        return solicitado

    @staticmethod
    def _importar_dependencias():
        try:
            # PyTorch emite una advertencia irrelevante en macOS al importar
            # utilidades de redirección que este proyecto no usa.
            logging.getLogger(
                "torch.distributed.elastic.multiprocessing.redirects"
            ).setLevel(logging.ERROR)
            import torch
            from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        except ImportError as error:
            raise ErrorConfiguracion(
                "Faltan dependencias de inferencia. Ejecuta: pip install -r requirements.txt"
            ) from error
        return torch, M2M100ForConditionalGeneration, M2M100Tokenizer

    def _cargar_desde(self, fuente, revision, es_base: bool):
        torch, clase_modelo, clase_tokenizer = self._importar_dependencias()
        dispositivo = self._elegir_dispositivo(torch)

        kwargs = {"revision": revision} if revision else {}
        tokenizer = clase_tokenizer.from_pretrained(fuente, **kwargs)

        # En CPU se conserva float32. En GPU/MPS, float16 reduce a la mitad la
        # memoria sin modificar los pesos guardados en disco.
        dtype = torch.float16 if dispositivo in {"cuda", "mps"} else torch.float32
        modelo = clase_modelo.from_pretrained(fuente, dtype=dtype, **kwargs)
        modelo.to(dispositivo)
        modelo.eval()
        modelo.config.use_cache = True
        if getattr(modelo, "generation_config", None) is not None:
            modelo.generation_config.use_cache = True

        self._torch = torch
        self._tokenizer = tokenizer
        self._modelo = modelo
        self._info = InformacionModelo(
            estado="listo",
            fuente=str(fuente),
            revision=revision,
            dispositivo=dispositivo,
            modelo_base=es_base,
        )

    def cargar(self) -> InformacionModelo:
        with self._candado:
            if self.cargado:
                return self._info

            self._info = InformacionModelo(estado="cargando")
            local = config.RUTA_MODELO_DIR
            anterior = config.RUTA_SALIDAS / "modelo_m2m100.anterior"
            if not local.exists() and _modelo_local_completo(anterior):
                LOGGER.warning("Recuperando modelo local anterior desde %s", anterior)
                os.replace(anterior, local)
            if local.exists() and not _modelo_local_completo(local):
                self._info = InformacionModelo(estado="error")
                raise ErrorCargaModelo(
                    f"Hay un modelo local incompleto en {local}. Elimínalo o vuelve a entrenarlo."
                )

            fuente = local if _modelo_local_completo(local) else config.HF_REPO_MODELO
            revision = None if fuente == local else config.HF_REVISION_MODELO
            try:
                self._cargar_desde(fuente, revision, es_base=False)
                return self._info
            except Exception as error:
                detalle = f"{type(error).__name__}: {error}"
                LOGGER.exception("No se pudo cargar el modelo afinado desde %s", fuente)
                if not self.permitir_modelo_base:
                    self._info = InformacionModelo(
                        estado="error", fuente=str(fuente), revision=revision, advertencia=detalle
                    )
                    raise ErrorCargaModelo(
                        "No se pudo cargar el modelo afinado. No se usó el modelo base para evitar "
                        f"resultados engañosos. Detalle: {detalle}"
                    ) from error

            try:
                self._cargar_desde(config.MODELO_BASE, revision=None, es_base=True)
                self._info = InformacionModelo(
                    **{
                        **self._info.como_dict(),
                        "advertencia": "Se usa el modelo base porque se autorizó explícitamente.",
                    }
                )
                return self._info
            except Exception as error:
                detalle = f"{type(error).__name__}: {error}"
                self._info = InformacionModelo(
                    estado="error", fuente=config.MODELO_BASE, modelo_base=True, advertencia=detalle
                )
                raise ErrorCargaModelo(f"Tampoco se pudo cargar el modelo base: {detalle}") from error

    def _generar(
        self, textos: Sequence[str], origen: str, destino: str, perfil: str | None = None
    ) -> list[str]:
        self.cargar()
        perfil = validar_perfil(perfil)
        parametros = config.PERFILES_GENERACION[perfil]
        tokenizer = self._tokenizer
        modelo = self._modelo
        torch = self._torch
        tokenizer.src_lang = origen
        salidas: list[str] = []

        with torch.inference_mode():
            for inicio in range(0, len(textos), config.TAMANO_LOTE_GENERACION):
                trozo = list(textos[inicio : inicio + config.TAMANO_LOTE_GENERACION])
                entradas = tokenizer(
                    trozo,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=getattr(config, "MAX_TOKENS_ENTRADA", config.MAX_TOKENS),
                )
                entradas = {clave: valor.to(modelo.device) for clave, valor in entradas.items()}
                opciones_generacion = {
                    "forced_bos_token_id": tokenizer.get_lang_id(destino),
                    "max_length": config.LONGITUD_MAXIMA_GENERACION,
                    "num_beams": parametros["num_beams"],
                    "no_repeat_ngram_size": config.NO_REPEAT_NGRAM_SIZE,
                    "use_cache": True,
                }
                if parametros["num_beams"] > 1:
                    opciones_generacion["early_stopping"] = True
                generados = modelo.generate(
                    **entradas,
                    **opciones_generacion,
                )
                salidas.extend(tokenizer.batch_decode(generados, skip_special_tokens=True))
        return [salida.strip() for salida in salidas]

    def traducir(
        self, texto: str, origen: str, destino: str, perfil: str | None = None
    ) -> str:
        texto = limpiar_texto(texto)
        validar_direccion(origen, destino)
        perfil = validar_perfil(perfil)
        clave_cache = (texto, origen, destino, perfil)
        with self._candado:
            if clave_cache in self._cache:
                self._cache.move_to_end(clave_cache)
                return self._cache[clave_cache]
            salida = self._generar([texto], origen, destino, perfil)[0]
            if not salida:
                raise ErrorTraductor("El modelo devolvió una traducción vacía.")
            self._cache[clave_cache] = salida
            while len(self._cache) > config.MAX_TRADUCCIONES_EN_CACHE:
                self._cache.popitem(last=False)
        return salida

    def traducir_lote(
        self,
        textos: Iterable[str],
        origen: str,
        destino: str,
        perfil: str | None = None,
    ) -> list[str]:
        validar_direccion(origen, destino)
        limpios = [limpiar_texto(texto) for texto in textos]
        if not limpios:
            return []
        with self._candado:
            salidas = self._generar(limpios, origen, destino, perfil)
        if len(salidas) != len(limpios) or any(not salida for salida in salidas):
            raise ErrorTraductor("El modelo no devolvió una salida válida para cada entrada.")
        return salidas

    def traducir_cadena(
        self,
        texto: str,
        origen: str,
        orden: Sequence[str],
        perfil: str | None = None,
    ) -> list[dict[str, str]]:
        texto = limpiar_texto(texto)
        orden = validar_orden(origen, orden)
        pasos = [{"idioma": origen, "texto": texto}]
        idioma_actual, texto_actual = origen, texto
        for siguiente in (*orden, origen):
            texto_actual = self.traducir(texto_actual, idioma_actual, siguiente, perfil)
            idioma_actual = siguiente
            pasos.append({"idioma": idioma_actual, "texto": texto_actual})
        return pasos

    def traducir_todas_las_cadenas(
        self, texto: str, origen: str, perfil: str | None = None
    ) -> list[dict]:
        texto = limpiar_texto(texto)
        validar_idioma(origen, "El idioma de origen")
        perfil = validar_perfil(perfil)
        otros = [idioma for idioma in config.IDIOMAS if idioma != origen]
        cache: dict[tuple[str, str, str], str] = {}
        resultados = []
        for orden in permutations(otros):
            pasos = [{"idioma": origen, "texto": texto}]
            idioma_actual, texto_actual = origen, texto
            for siguiente in (*orden, origen):
                clave = (texto_actual, idioma_actual, siguiente)
                if clave not in cache:
                    cache[clave] = self.traducir(
                        texto_actual, idioma_actual, siguiente, perfil
                    )
                texto_actual = cache[clave]
                idioma_actual = siguiente
                pasos.append({"idioma": idioma_actual, "texto": texto_actual})
            resultados.append({"orden": list(orden), "pasos": pasos})
        return resultados
