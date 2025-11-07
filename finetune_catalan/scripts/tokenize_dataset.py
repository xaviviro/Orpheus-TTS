"""
Script para tokenizar el dataset catalán preparado según el formato de Orpheus TTS.

Este script toma el dataset procesado y aplica la tokenización usando el codec
de audio de Orpheus (Snac) para prepararlo para el entrenamiento.
"""

import os
import argparse
from datasets import load_from_disk, load_dataset, Dataset, Audio
from pathlib import Path
import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
import threading


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
        default=8,
        help='Tamaño del batch para procesamiento en GPU'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Número de workers para pre-procesamiento'
    )
    parser.add_argument(
        '--hf_repo',
        type=str,
        default=None,
        help='Repositorio de HuggingFace para subir el dataset tokenizado'
    )
    parser.add_argument(
        '--hf_dataset',
        type=str,
        default=None,
        help='Dataset de HuggingFace para cargar (alternativa a --input_dir)'
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


def tokenize_audio_batch(audio_arrays, snac_model, device):
    """
    Tokeniza un batch de audios usando SNAC.

    Args:
        audio_arrays: Lista de arrays de audio numpy
        snac_model: Modelo SNAC
        device: Dispositivo (cuda/cpu)

    Returns:
        Lista de tokens tokenizados
    """
    batch_tokens = []

    for audio_array in audio_arrays:
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
        batch_tokens.append(all_tokens)

    return batch_tokens


def tokenize_batch(examples, tokenizer, snac_model, device, max_length):
    """
    Tokeniza un batch de ejemplos (texto + audio).
    """
    results = []

    # Tokenizar textos en batch
    texts = [ex['text'] for ex in examples]
    text_tokens_batch = tokenizer(
        texts,
        truncation=True,
        max_length=max_length // 2,  # Reservar espacio para audio
        padding=False
    )

    # Tokenizar audios en batch
    audio_arrays = [ex['audio']['array'] for ex in examples]
    audio_tokens_batch = tokenize_audio_batch(audio_arrays, snac_model, device)

    # Combinar cada ejemplo
    for i, example in enumerate(examples):
        try:
            # Combinar tokens de texto y audio
            # Formato: [texto_tokens] [separador] [audio_tokens]
            input_ids = np.concatenate([
                np.array(text_tokens_batch['input_ids'][i]),
                audio_tokens_batch[i]
            ])

            # Truncar si es necesario
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]

            # Crear labels (para causal LM, labels = input_ids)
            labels = input_ids.copy()

            results.append({
                'input_ids': input_ids.tolist(),
                'labels': labels.tolist(),
                'text': example['text'],
                'accent': example.get('accent', 'unknown'),
                'speaker_id': example.get('speaker_id', 'unknown'),
                'duration': example['duration']
            })

        except Exception as e:
            print(f"Error tokenizando ejemplo {i}: {e}")
            continue

    return results


def tokenize_dataset(args):
    """
    Función principal para tokenizar el dataset con batching.
    """
    print("="*60)
    print("TOKENIZACIÓN DE DATASET CATALÁN PARA ORPHEUS TTS")
    print("="*60)

    # Cargar dataset
    if args.hf_dataset:
        print(f"\nCargando dataset desde HuggingFace: {args.hf_dataset}")
        dataset = load_dataset(args.hf_dataset)
    else:
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

    # Función para procesar un split con batching
    def process_split(split_data, split_name):
        print(f"\n{'='*60}")
        print(f"Procesando split: {split_name}")
        print(f"{'='*60}")

        tokenized = []
        batch = []

        # Procesar en batches
        pbar = tqdm(total=len(split_data), desc=f"Tokenizando {split_name}")

        for example in split_data:
            batch.append(example)

            if len(batch) >= args.batch_size:
                # Tokenizar batch
                results = tokenize_batch(batch, tokenizer, snac_model, args.device, args.max_length)
                tokenized.extend(results)
                batch = []
                pbar.update(args.batch_size)

        # Procesar batch restante
        if batch:
            results = tokenize_batch(batch, tokenizer, snac_model, args.device, args.max_length)
            tokenized.extend(results)
            pbar.update(len(batch))

        pbar.close()

        print(f"✓ {split_name}: {len(tokenized)}/{len(split_data)} ejemplos tokenizados")
        return tokenized

    # Procesar splits
    tokenized_train = process_split(dataset['train'], 'train')
    tokenized_val = process_split(dataset['validation'], 'validation')

    # Crear nuevo dataset
    print("\n🔨 Creando dataset tokenizado...")
    from datasets import DatasetDict
    tokenized_dataset = DatasetDict({
        'train': Dataset.from_list(tokenized_train),
        'validation': Dataset.from_list(tokenized_val)
    })

    # Guardar localmente
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n💾 Guardando dataset tokenizado en: {output_dir}")
        tokenized_dataset.save_to_disk(str(output_dir))
        print("✓ Dataset guardado localmente")

    # Subir a HuggingFace
    if args.hf_repo:
        print(f"\n☁️  Subiendo a HuggingFace: {args.hf_repo}")
        try:
            tokenized_dataset.push_to_hub(
                args.hf_repo,
                commit_message="Add tokenized Catalan TTS dataset"
            )
            print(f"✅ Dataset subido exitosamente")
            print(f"   URL: https://huggingface.co/datasets/{args.hf_repo}")
        except Exception as e:
            print(f"⚠️  Error subiendo a HuggingFace: {e}")

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

    print("\n✅ ¡Tokenización completada!")
    return tokenized_dataset


if __name__ == '__main__':
    args = parse_args()
    tokenize_dataset(args)
