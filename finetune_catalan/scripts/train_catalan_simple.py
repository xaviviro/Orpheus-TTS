"""
Script de entrenamiento para Orpheus TTS en Catalán.
Basado exactamente en el script original de ./finetune/train.py
"""

from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer, DataCollatorForLanguageModeling
import numpy as np
import yaml
import wandb
import os
import sys

def main():
    # Cargar configuración
    config_file = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == '--config' else "configs/config_catalan.yaml"

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # Extraer parámetros del config
    dsn = config["TTS_dataset"]
    model_name = config["model_name"]

    # Training config
    training_config = config["training"]
    epochs = training_config["epochs"]
    batch_size = training_config["batch_size"]
    save_steps = training_config["save_steps"]
    learning_rate = training_config["learning_rate"]

    # Paths
    paths_config = config["paths"]
    output_dir = paths_config["output_dir"]

    # WandB config
    wandb_config = config.get("wandb", {})
    project_name = wandb_config.get("project_name", "orpheus-catalan-tts")
    run_name = wandb_config.get("run_name", "catalan-tts-training")

    # Model config
    model_config = config.get("model_config", {})
    attn_implementation = model_config.get("attn_implementation", "sdpa")

    # Cargar tokenizer y modelo
    print(f"Cargando modelo: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Configurar pad_token si no existe
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation=attn_implementation
    )

    # Cargar dataset
    print(f"Cargando dataset: {dsn}")
    if os.path.exists(dsn):
        # Dataset local
        ds = load_from_disk(dsn)
        train_ds = ds["train"]
    else:
        # Dataset de HuggingFace
        train_ds = load_dataset(dsn, split="train")

    print(f"  - Train: {len(train_ds)} ejemplos")

    # Inicializar WandB
    wandb.init(project=project_name, name=run_name)

    # Crear data collator para padding dinámico
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, no masked LM
    )

    # Configurar argumentos de entrenamiento (exactamente como el original)
    training_args = TrainingArguments(
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        logging_steps=training_config.get("logging_steps", 1),
        bf16=training_config.get("bf16", True),
        output_dir=output_dir,
        report_to=training_config.get("report_to", "wandb"),
        save_steps=save_steps,
        remove_unused_columns=True,
        learning_rate=learning_rate,
    )

    # Crear trainer con data_collator para manejar padding
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
    )

    # Entrenar
    print("\nIniciando entrenamiento...")
    trainer.train()

    print("\n✅ Entrenamiento completado!")

if __name__ == '__main__':
    main()
