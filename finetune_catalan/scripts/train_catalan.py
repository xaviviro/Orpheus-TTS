"""
Script de entrenamiento para Orpheus TTS en Catalán con variantes dialectales.

Basado en el script de entrenamiento original de Orpheus pero adaptado para
manejar múltiples dialectos catalanes.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import wandb


def parse_args():
    parser = argparse.ArgumentParser(description='Entrenar Orpheus TTS para catalán')
    parser.add_argument(
        '--config',
        type=str,
        default='../configs/config_catalan.yaml',
        help='Ruta al archivo de configuración'
    )
    parser.add_argument(
        '--resume_from_checkpoint',
        type=str,
        default=None,
        help='Ruta al checkpoint para reanudar entrenamiento'
    )
    return parser.parse_args()


def load_config(config_path):
    """
    Carga la configuración desde el archivo YAML.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model_and_tokenizer(config):
    """
    Carga el modelo y tokenizer.
    """
    model_name = config['model_name']
    model_config = config.get('model_config', {})

    print(f"Cargando modelo: {model_name}")

    # Cargar tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Cargar modelo
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation=model_config.get('attn_implementation', 'sdpa'),
        use_cache=model_config.get('use_cache', False),
        torch_dtype=torch.bfloat16 if config['training'].get('bf16', True) else torch.float32
    )

    # Habilitar gradient checkpointing si está configurado
    if config.get('advanced', {}).get('gradient_checkpointing', False):
        model.gradient_checkpointing_enable()

    return model, tokenizer


def load_training_dataset(config):
    """
    Carga el dataset de entrenamiento.
    """
    dataset_name = config['TTS_dataset']

    print(f"Cargando dataset: {dataset_name}")

    # Determinar si es una ruta local o un dataset de HuggingFace
    if os.path.exists(dataset_name):
        # Cargar desde disco local
        dataset = load_from_disk(dataset_name)
    else:
        # Cargar desde HuggingFace Hub
        dataset = load_dataset(dataset_name)

    print(f"Dataset cargado:")
    print(f"  - Train: {len(dataset['train'])} ejemplos")
    if 'validation' in dataset:
        print(f"  - Validation: {len(dataset['validation'])} ejemplos")

    return dataset


def setup_wandb(config):
    """
    Configura WandB para logging.
    """
    wandb_config = config.get('wandb', {})

    if wandb_config.get('project_name'):
        wandb.init(
            project=wandb_config['project_name'],
            name=wandb_config.get('run_name', 'catalan-tts-training'),
            entity=wandb_config.get('entity'),
            config=config
        )


def create_training_arguments(config):
    """
    Crea los argumentos de entrenamiento.
    """
    training_config = config['training']
    paths_config = config['paths']

    # Crear directorios necesarios
    os.makedirs(paths_config['save_folder'], exist_ok=True)
    os.makedirs(paths_config['output_dir'], exist_ok=True)
    os.makedirs(paths_config['logging_dir'], exist_ok=True)

    args = TrainingArguments(
        output_dir=paths_config['output_dir'],
        overwrite_output_dir=True,

        # Epochs y batch size
        num_train_epochs=training_config['epochs'],
        per_device_train_batch_size=training_config['batch_size'],
        per_device_eval_batch_size=training_config['batch_size'],
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 1),

        # Optimización
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config.get('weight_decay', 0.01),
        max_grad_norm=training_config.get('max_grad_norm', 1.0),
        warmup_steps=training_config.get('warmup_steps', 500),

        # Scheduler
        lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),

        # Precisión
        fp16=training_config.get('fp16', False),
        bf16=training_config.get('bf16', True),

        # Logging
        logging_dir=paths_config['logging_dir'],
        logging_steps=training_config.get('logging_steps', 50),
        report_to=training_config.get('report_to', 'wandb'),

        # Guardado
        save_strategy=training_config.get('save_strategy', 'steps'),
        save_steps=training_config.get('save_steps', 500),
        save_total_limit=training_config.get('save_total_limit', 3),

        # Evaluación
        evaluation_strategy=training_config.get('evaluation_strategy', 'steps'),
        eval_steps=training_config.get('eval_steps', 500),

        # Otros
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        greater_is_better=False,
    )

    return args


def main():
    args = parse_args()

    # Cargar configuración
    print("="*60)
    print("ENTRENAMIENTO DE ORPHEUS TTS PARA CATALÁN")
    print("="*60)

    config = load_config(args.config)
    print(f"\nConfiguración cargada desde: {args.config}")

    # Setup WandB
    if config['training'].get('report_to') == 'wandb':
        setup_wandb(config)

    # Cargar modelo y tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Cargar dataset
    dataset = load_training_dataset(config)

    # Preparar data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, no masked LM
    )

    # Crear argumentos de entrenamiento
    training_args = create_training_arguments(config)

    # Crear trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset.get('validation'),
        data_collator=data_collator,
    )

    # Entrenar
    print("\nIniciando entrenamiento...")
    print(f"Dispositivo: {training_args.device}")
    print(f"Número de ejemplos (train): {len(dataset['train'])}")
    if 'validation' in dataset:
        print(f"Número de ejemplos (validation): {len(dataset['validation'])}")

    # Iniciar entrenamiento
    if args.resume_from_checkpoint:
        print(f"Reanudando desde checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    # Guardar modelo final
    final_model_path = os.path.join(config['paths']['save_folder'], 'final_model')
    print(f"\nGuardando modelo final en: {final_model_path}")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    # Evaluar modelo final
    if 'validation' in dataset:
        print("\nEvaluando modelo final...")
        eval_results = trainer.evaluate()
        print("Resultados de evaluación:")
        for key, value in eval_results.items():
            print(f"  {key}: {value}")

    print("\n¡Entrenamiento completado!")

    # Finalizar WandB
    if config['training'].get('report_to') == 'wandb':
        wandb.finish()


if __name__ == '__main__':
    main()
