"""
Afina M2M100 (facebook/m2m100_418M) sobre los corpus de OPUS ya preparados.

    python entrenar.py

Requiere haber corrido antes `python preparar_datos.py` (genera
data/procesados/entrenamiento.tsv y validacion.tsv). Pensado para correr en
Colab con GPU: en CPU tardaría horas en vez de minutos.

Qué hace:
1. Toma una muestra pareja de entrenamiento.tsv para las 12 direcciones de
   traducción (ver config.EJEMPLOS_POR_DIRECCION).
2. Tokeniza cada dirección por separado: M2M100Tokenizer necesita que
   src_lang/tgt_lang estén fijados ANTES de tokenizar, así que no se puede
   tokenizar de un jalón un lote con idiomas mezclados.
3. Afina con Seq2SeqTrainer (entropía cruzada estándar de traducción, igual
   que cualquier Transformer encoder-decoder).
4. Guarda el modelo afinado en salidas/modelo_m2m100/ y el historial de
   pérdida por época en salidas/historico_entrenamiento.csv.

La evaluación real (BLEU/chrF por dirección y en cadenas de 4 idiomas) es
un paso separado: python evaluar.py
"""

import csv
import json
import os
import shutil
from importlib.metadata import PackageNotFoundError, version

import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets
from transformers import (
    DataCollatorForSeq2Seq,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

import config
from muestreo import muestrear_archivo_direcciones


def _leer_tsv(ruta):
    return pd.read_csv(ruta, sep="\t", quoting=csv.QUOTE_NONE, escapechar="\\", keep_default_na=False)


def muestrear_direcciones(df_pares, n_por_direccion, semilla):
    """De pares (idioma_a, idioma_b, texto_a, texto_b) saca ejemplos
    (origen, destino, texto_origen, texto_destino) en AMBAS direcciones,
    hasta n_por_direccion de cada una."""
    filas = []
    for (a, b), grupo in df_pares.groupby(["idioma_a", "idioma_b"]):
        grupo = grupo.sample(n=min(n_por_direccion, len(grupo)), random_state=semilla)
        for texto_a, texto_b in zip(grupo.texto_a, grupo.texto_b):
            filas.append({"origen": a, "destino": b, "texto_origen": texto_a, "texto_destino": texto_b})
            filas.append({"origen": b, "destino": a, "texto_origen": texto_b, "texto_destino": texto_a})
    return pd.DataFrame(filas)


def tokenizar_por_direccion(df, tokenizer):
    """Tokeniza cada (origen, destino) en su propio lote (ver docstring del
    módulo) y junta todo en un solo Dataset de HuggingFace, mezclado."""
    trozos = []
    for (origen, destino), grupo in df.groupby(["origen", "destino"]):
        tokenizer.src_lang = origen
        tokenizer.tgt_lang = destino
        for inicio in range(0, len(grupo), config.TAMANO_CHUNK_TOKENIZACION):
            chunk = grupo.iloc[inicio : inicio + config.TAMANO_CHUNK_TOKENIZACION]
            codificado = tokenizer(
                list(chunk.texto_origen),
                text_target=list(chunk.texto_destino),
                truncation=True,
                max_length=config.MAX_TOKENS,
            )
            trozos.append(Dataset.from_dict(dict(codificado)))
    if not trozos:
        raise ValueError("No se generó ningún ejemplo tokenizado.")
    return concatenate_datasets(trozos).shuffle(seed=config.SEMILLA)


def construir_datasets(tokenizer):
    ruta_entrenamiento = config.RUTA_PROCESADOS / "entrenamiento.tsv"
    if not ruta_entrenamiento.is_file():
        raise FileNotFoundError(
            f"Falta {ruta_entrenamiento}. Ejecuta primero: python Traductor.py preparar"
        )
    entrenamiento = muestrear_archivo_direcciones(
        ruta_entrenamiento, config.EJEMPLOS_POR_DIRECCION, config.SEMILLA,
    )
    print(f"  Afinamiento: {len(entrenamiento):,} ejemplos "
          f"(~{len(entrenamiento) // 12:,} por cada una de las 12 direcciones)")

    # validacion.tsv es multi-paralela (una columna por idioma): se
    # reconstruyen las 12 direcciones desde ahí para medir la pérdida
    # durante el afinamiento. No es la evaluación real (eso es evaluar.py).
    val_multi = _leer_tsv(config.RUTA_PROCESADOS / "validacion.tsv").sample(
        n=config.EJEMPLOS_EVAL_POR_DIRECCION, random_state=config.SEMILLA
    )
    filas_val = [
        {"origen": origen, "destino": destino, "texto_origen": texto_o, "texto_destino": texto_d}
        for origen in config.IDIOMAS
        for destino in config.IDIOMAS
        if origen != destino
        for texto_o, texto_d in zip(val_multi[origen], val_multi[destino])
    ]
    validacion = pd.DataFrame(filas_val)
    print(f"  Validación (solo pérdida, no BLEU): {len(validacion):,} ejemplos")

    return tokenizar_por_direccion(entrenamiento, tokenizer), tokenizar_por_direccion(validacion, tokenizer)


def crear_argumentos_entrenamiento(**kwargs):
    """Seq2SeqTrainingArguments renombró evaluation_strategy -> eval_strategy
    en versiones recientes de transformers. Probar los dos nombres evita que
    el afinamiento falle solo por la versión instalada en Colab ese día."""
    try:
        return Seq2SeqTrainingArguments(eval_strategy="epoch", **kwargs)
    except TypeError:
        return Seq2SeqTrainingArguments(evaluation_strategy="epoch", **kwargs)


def entrenar():
    config.RUTA_SALIDAS.mkdir(parents=True, exist_ok=True)
    ruta_anterior = config.RUTA_SALIDAS / "modelo_m2m100.anterior"
    if ruta_anterior.exists() and not config.RUTA_MODELO_DIR.exists():
        print("Recuperando el modelo anterior después de una interrupción de guardado...")
        os.replace(ruta_anterior, config.RUTA_MODELO_DIR)

    print(f"Cargando {config.MODELO_BASE} (preentrenado)...")
    tokenizer = M2M100Tokenizer.from_pretrained(config.MODELO_BASE)
    modelo = M2M100ForConditionalGeneration.from_pretrained(config.MODELO_BASE)
    modelo.config.use_cache = False

    print("Preparando datos de afinamiento...")
    ds_train, ds_val = construir_datasets(tokenizer)

    ruta_checkpoints = config.RUTA_SALIDAS / "checkpoints_temporales"
    argumentos = crear_argumentos_entrenamiento(
        output_dir=str(ruta_checkpoints),
        num_train_epochs=config.EPOCAS,
        per_device_train_batch_size=config.TAMANO_LOTE,
        per_device_eval_batch_size=config.TAMANO_LOTE,
        gradient_accumulation_steps=config.ACUMULACION_GRADIENTE,
        learning_rate=config.TASA_APRENDIZAJE,
        warmup_steps=config.PASOS_CALENTAMIENTO,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        logging_steps=50,
        report_to=[],
        seed=config.SEMILLA,
    )

    entrenador = Seq2SeqTrainer(
        model=modelo,
        args=argumentos,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=modelo),
    )

    print("Afinando (esto tarda varios minutos, va a imprimir el progreso)...")
    ultimo_checkpoint = get_last_checkpoint(str(ruta_checkpoints)) if ruta_checkpoints.exists() else None
    if ultimo_checkpoint:
        print(f"Reanudando desde {ultimo_checkpoint}")
    resultado = entrenador.train(resume_from_checkpoint=ultimo_checkpoint)

    historico_temporal = config.RUTA_HISTORICO_CSV.with_suffix(".csv.part")
    pd.DataFrame(entrenador.state.log_history).to_csv(historico_temporal, index=False)
    os.replace(historico_temporal, config.RUTA_HISTORICO_CSV)

    if hasattr(modelo, "gradient_checkpointing_disable"):
        modelo.gradient_checkpointing_disable()
    modelo.config.use_cache = True
    if getattr(modelo, "generation_config", None) is not None:
        modelo.generation_config.use_cache = True

    ruta_nueva = config.RUTA_SALIDAS / "modelo_m2m100.nuevo"
    shutil.rmtree(ruta_nueva, ignore_errors=True)
    shutil.rmtree(ruta_anterior, ignore_errors=True)
    ruta_nueva.mkdir(parents=True)
    modelo.save_pretrained(ruta_nueva, safe_serialization=True)
    tokenizer.save_pretrained(ruta_nueva)

    versiones = {}
    for paquete in ("torch", "transformers", "datasets", "accelerate", "pandas"):
        try:
            versiones[paquete] = version(paquete)
        except PackageNotFoundError:
            versiones[paquete] = "no disponible"
    with open(ruta_nueva / "metadata_entrenamiento.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "modelo_base": config.MODELO_BASE,
                "semilla": config.SEMILLA,
                "epocas": config.EPOCAS,
                "ejemplos_por_direccion": config.EJEMPLOS_POR_DIRECCION,
                "max_tokens": config.MAX_TOKENS,
                "training_loss": resultado.training_loss,
                "versiones": versiones,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.flush()
        os.fsync(f.fileno())

    if config.RUTA_MODELO_DIR.exists():
        os.replace(config.RUTA_MODELO_DIR, ruta_anterior)
    try:
        os.replace(ruta_nueva, config.RUTA_MODELO_DIR)
    except Exception:
        if ruta_anterior.exists() and not config.RUTA_MODELO_DIR.exists():
            os.replace(ruta_anterior, config.RUTA_MODELO_DIR)
        raise
    shutil.rmtree(ruta_anterior, ignore_errors=True)
    shutil.rmtree(ruta_checkpoints, ignore_errors=True)

    print(f"OK: modelo afinado en {config.RUTA_MODELO_DIR}")
    print(f"    pérdida final de entrenamiento: {resultado.training_loss:.4f}")


if __name__ == "__main__":
    entrenar()
