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
import shutil

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

import config


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
        codificado = tokenizer(
            list(grupo.texto_origen),
            text_target=list(grupo.texto_destino),
            truncation=True,
            max_length=config.MAX_TOKENS,
        )
        trozos.append(Dataset.from_dict(dict(codificado)))
    return concatenate_datasets(trozos).shuffle(seed=config.SEMILLA)


def construir_datasets(tokenizer):
    entrenamiento = muestrear_direcciones(
        _leer_tsv(config.RUTA_PROCESADOS / "entrenamiento.tsv"),
        config.EJEMPLOS_POR_DIRECCION, config.SEMILLA,
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
    print(f"Cargando {config.MODELO_BASE} (preentrenado)...")
    tokenizer = M2M100Tokenizer.from_pretrained(config.MODELO_BASE)
    modelo = M2M100ForConditionalGeneration.from_pretrained(config.MODELO_BASE)

    print("Preparando datos de afinamiento...")
    ds_train, ds_val = construir_datasets(tokenizer)

    config.RUTA_SALIDAS.mkdir(parents=True, exist_ok=True)
    ruta_checkpoints = config.RUTA_SALIDAS / "checkpoints_temporales"
    argumentos = crear_argumentos_entrenamiento(
        output_dir=str(ruta_checkpoints),
        num_train_epochs=config.EPOCAS,
        per_device_train_batch_size=config.TAMANO_LOTE,
        per_device_eval_batch_size=config.TAMANO_LOTE,
        gradient_accumulation_steps=config.ACUMULACION_GRADIENTE,
        learning_rate=config.TASA_APRENDIZAJE,
        warmup_steps=config.PASOS_CALENTAMIENTO,
        save_strategy="no",  # el único modelo que importa es el afinado completo, al final
        fp16=torch.cuda.is_available(),
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
    resultado = entrenador.train()

    pd.DataFrame(entrenador.state.log_history).to_csv(config.RUTA_HISTORICO_CSV, index=False)

    config.RUTA_MODELO_DIR.mkdir(parents=True, exist_ok=True)
    modelo.save_pretrained(config.RUTA_MODELO_DIR)
    tokenizer.save_pretrained(config.RUTA_MODELO_DIR)
    shutil.rmtree(ruta_checkpoints, ignore_errors=True)

    print(f"OK: modelo afinado en {config.RUTA_MODELO_DIR}")
    print(f"    pérdida final de entrenamiento: {resultado.training_loss:.4f}")


if __name__ == "__main__":
    entrenar()
