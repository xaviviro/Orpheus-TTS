"""
Script para tokenizar el dataset catalán preparado según el formato de Orpheus TTS.

Este script toma el dataset procesado y aplica la tokenización usando el codec
de audio de Orpheus (Snac) para prepararlo para el entrenamiento.
"""

import os
import argparse
from datasets import load_from_disk, Dataset, Audio
from pathlib import Path
import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='Tokenizar dataset catalán para Orpheus TTS')
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directorio con el dataset procesado'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directorio de salida para el dataset tokenizado'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='canopylabs/orpheus-tts-0.1-pretrained',
        help='Modelo base para el tokenizer'
    )
    parser.add_argument(
        '--snac_model',
        type=str,
        default='hubertsiuzdak/snac_24khz',
        help='Modelo SNAC para tokenización de audio'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=8192,
        help='Longitud máxima de secuencia'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Dispositivo para procesamiento'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Tamaño del batch para procesamiento'
    )

    return parser.parse_args()


def load_snac_model(model_name, device):
    """
    Carga el modelo SNAC para tokenización de audio.
    """
    try:
        from snac import SNAC
        print(f"Cargando modelo SNAC: {model_name}")
        model = SNAC.from_pretrained(model_name).to(device)
        model.eval()
        return model
    except ImportError:
        print("ERROR: El paquete 'snac' no está instalado.")
        print("Instala con: pip install snac")
        raise
    except Exception as e:
        print(f"Error cargando modelo SNAC: {e}")
        raise


def tokenize_audio(audio_array, sample_rate, snac_model, device):
    """
    Tokeniza el audio usando SNAC.

    Args:
        audio_array: Array de audio numpy
        sample_rate: Frecuencia de muestreo
        snac_model: Modelo SNAC
        device: Dispositivo (cuda/cpu)

    Returns:
        tokens: Tokens de audio tokenizado
    """
    # Convertir a tensor
    audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0).unsqueeze(0).to(device)

    # Tokenizar con SNAC
    with torch.no_grad():
        codes = snac_model.encode(audio_tensor)

    # SNAC devuelve códigos en múltiples niveles jerárquicos
    # Orpheus usa los 3 niveles concatenados
    tokens = []
    for level_codes in codes:
        tokens.append(level_codes.cpu().numpy().flatten())

    # Concatenar todos los niveles
    all_tokens = np.concatenate(tokens)

    return all_tokens


def tokenize_example(example, tokenizer, snac_model, device, max_length):
    """
    Tokeniza un ejemplo completo (texto + audio).
    """
    try:
        # Tokenizar texto
        text = example['text']
        text_tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_length // 2,  # Reservar espacio para audio
            return_tensors='pt'
        )

        # Tokenizar audio
        audio_array = example['audio']['array']
        sample_rate = example['audio']['sampling_rate']
        audio_tokens = tokenize_audio(audio_array, sample_rate, snac_model, device)

        # Combinar tokens de texto y audio
        # Formato: [texto_tokens] [separador] [audio_tokens]
        input_ids = np.concatenate([
            text_tokens['input_ids'].numpy().flatten(),
            audio_tokens
        ])

        # Truncar si es necesario
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]

        # Crear labels (para causal LM, labels = input_ids desplazados)
        labels = input_ids.copy()

        return {
            'input_ids': input_ids.tolist(),
            'labels': labels.tolist(),
            'text': text,
            'accent': example['accent'],
            'speaker_id': example.get('speaker_id', 'unknown'),
            'duration': example['duration']
        }

    except Exception as e:
        print(f"Error tokenizando ejemplo: {e}")
        return None


def tokenize_dataset(args):
    """
    Función principal para tokenizar el dataset.
    """
    print("="*60)
    print("TOKENIZACIÓN DE DATASET CATALÁN PARA ORPHEUS TTS")
    print("="*60)

    # Cargar dataset procesado
    print(f"\nCargando dataset desde: {args.input_dir}")
    dataset = load_from_disk(args.input_dir)

    print(f"Train: {len(dataset['train'])} ejemplos")
    print(f"Validation: {len(dataset['validation'])} ejemplos")

    # Cargar tokenizer de texto
    print(f"\nCargando tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Cargar modelo SNAC
    print(f"\nCargando modelo SNAC: {args.snac_model}")
    snac_model = load_snac_model(args.snac_model, args.device)

    # Tokenizar dataset
    print("\nTokenizando ejemplos...")

    def tokenize_fn(example):
        return tokenize_example(
            example,
            tokenizer,
            snac_model,
            args.device,
            args.max_length
        )

    # Procesar train
    print("\nProcesando split de entrenamiento...")
    tokenized_train = []
    for example in tqdm(dataset['train']):
        result = tokenize_fn(example)
        if result is not None:
            tokenized_train.append(result)

    # Procesar validation
    print("\nProcesando split de validación...")
    tokenized_val = []
    for example in tqdm(dataset['validation']):
        result = tokenize_fn(example)
        if result is not None:
            tokenized_val.append(result)

    # Crear nuevo dataset
    print("\nCreando dataset tokenizado...")
    from datasets import DatasetDict
    tokenized_dataset = DatasetDict({
        'train': Dataset.from_list(tokenized_train),
        'validation': Dataset.from_list(tokenized_val)
    })

    # Guardar dataset tokenizado
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGuardando dataset tokenizado en: {output_dir}")
    tokenized_dataset.save_to_disk(str(output_dir))

    # Estadísticas
    print("\n" + "="*60)
    print("ESTADÍSTICAS DEL DATASET TOKENIZADO")
    print("="*60)
    print(f"Train: {len(tokenized_dataset['train'])} ejemplos")
    print(f"Validation: {len(tokenized_dataset['validation'])} ejemplos")

    # Longitudes promedio
    train_lengths = [len(ex['input_ids']) for ex in tokenized_dataset['train']]
    print(f"\nLongitud promedio de secuencia (train): {np.mean(train_lengths):.0f} tokens")
    print(f"Longitud máxima (train): {np.max(train_lengths)} tokens")
    print(f"Longitud mínima (train): {np.min(train_lengths)} tokens")

    print("\n¡Tokenización completada!")
    return tokenized_dataset


if __name__ == '__main__':
    args = parse_args()
    tokenize_dataset(args)
