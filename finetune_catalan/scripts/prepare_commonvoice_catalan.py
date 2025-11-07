"""
Script para preparar el dataset Common Voice Catalán con variantes dialectales
para fine-tuning de Orpheus TTS.

Este script:
1. Descarga los datasets de Common Voice en catalán
2. Filtra y balancea los datos por acento/dialecto
3. Procesa el audio y texto según el formato requerido por Orpheus
4. Genera un dataset en formato Hugging Face listo para entrenamiento
"""

import os
import argparse
from datasets import load_dataset, Dataset, Audio, DatasetDict
from pathlib import Path
import pandas as pd
import soundfile as sf
import librosa
import numpy as np
from tqdm import tqdm
import json

# Mapeo de acentos/dialectos catalanes
CATALAN_ACCENTS = {
    'balearic': 'balear',
    'central': 'central',
    'northern': 'nord',
    'northwestern': 'nord-occidental',
    'valencian': 'valencià'
}

# Voces asignadas por dialecto (puedes personalizar estos nombres)
VOICE_NAMES_BY_ACCENT = {
    'balearic': 'maria',
    'central': 'pau',
    'northern': 'montse',
    'northwestern': 'jordi',
    'valencian': 'carla'
}


def parse_args():
    parser = argparse.ArgumentParser(description='Preparar dataset Common Voice Catalán para Orpheus TTS')
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='projecte-aina/commonvoice_benchmark_catalan_accents',
        help='Nombre del dataset en HuggingFace'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../data/processed',
        help='Directorio de salida para el dataset procesado'
    )
    parser.add_argument(
        '--target_sample_rate',
        type=int,
        default=24000,
        help='Frecuencia de muestreo objetivo (Orpheus usa 24kHz)'
    )
    parser.add_argument(
        '--min_duration',
        type=float,
        default=1.0,
        help='Duración mínima del audio en segundos'
    )
    parser.add_argument(
        '--max_duration',
        type=float,
        default=30.0,
        help='Duración máxima del audio en segundos'
    )
    parser.add_argument(
        '--accents',
        type=str,
        nargs='+',
        default=['balearic', 'central', 'northern', 'northwestern', 'valencian'],
        help='Acentos a incluir en el dataset'
    )
    parser.add_argument(
        '--samples_per_accent',
        type=int,
        default=500,
        help='Número de muestras por acento (None para todas)'
    )
    parser.add_argument(
        '--use_alternative_dataset',
        action='store_true',
        help='Usar mozilla-foundation/common_voice_13_0 con filtro de idioma ca'
    )
    parser.add_argument(
        '--push_to_hub',
        action='store_true',
        help='Subir el dataset procesado a Hugging Face Hub'
    )
    parser.add_argument(
        '--hub_dataset_name',
        type=str,
        default='orpheus-catalan-tts',
        help='Nombre del dataset en HuggingFace Hub'
    )

    return parser.parse_args()


def load_common_voice_dataset(dataset_name, use_alternative=False):
    """
    Carga el dataset Common Voice en catalán.
    """
    print(f"Cargando dataset: {dataset_name}")

    if use_alternative:
        # Usar Common Voice 13.0 con filtro de idioma catalán
        ds = load_dataset(dataset_name, 'ca', split='train')
    else:
        # Usar el dataset especializado de variantes dialectales
        try:
            ds = load_dataset(dataset_name, split='train')
        except Exception as e:
            print(f"Error cargando dataset: {e}")
            print("Intentando cargar desde cache o descarga manual...")
            raise

    print(f"Dataset cargado: {len(ds)} muestras")
    return ds


def filter_by_quality(example):
    """
    Filtra ejemplos de baja calidad basándose en metadatos.
    """
    # Si tiene anotaciones de calidad, usar ese criterio
    if 'mean quality' in example and example['mean quality'] is not None:
        return example['mean quality'] > 2.5  # Umbral de calidad

    # Si no, usar votos
    if 'up_votes' in example and 'down_votes' in example:
        up = example.get('up_votes', 0) or 0
        down = example.get('down_votes', 0) or 0
        if (up + down) == 0:
            return True
        return up / (up + down) > 0.7

    return True


def filter_by_duration(example, min_dur, max_dur, sample_rate=24000):
    """
    Filtra por duración del audio.
    """
    if 'audio' not in example or example['audio'] is None:
        return False

    # Calcular duración
    audio_array = example['audio']['array']
    duration = len(audio_array) / sample_rate

    return min_dur <= duration <= max_dur


def resample_audio(audio_array, orig_sr, target_sr):
    """
    Resamplea el audio a la frecuencia objetivo.
    """
    if orig_sr == target_sr:
        return audio_array

    return librosa.resample(audio_array, orig_sr=orig_sr, target_sr=target_sr)


def process_example(example, target_sr, accent_name):
    """
    Procesa un ejemplo individual para el formato de Orpheus.
    """
    # Obtener audio y texto
    audio_data = example['audio']
    text = example['sentence']

    # Resamplear si es necesario
    audio_array = audio_data['array']
    orig_sr = audio_data['sampling_rate']

    if orig_sr != target_sr:
        audio_array = resample_audio(audio_array, orig_sr, target_sr)

    # Asignar nombre de voz basado en el acento
    voice_name = VOICE_NAMES_BY_ACCENT.get(accent_name, 'speaker')

    # Formatear el texto según el formato de Orpheus
    formatted_text = f"{voice_name}: {text}"

    # Extraer metadatos adicionales
    accent = example.get('annotated_accent', example.get('accent', accent_name))
    gender = example.get('annotated_gender', example.get('gender', 'unknown'))

    return {
        'audio': {
            'array': audio_array,
            'sampling_rate': target_sr
        },
        'text': formatted_text,
        'original_text': text,
        'voice_name': voice_name,
        'accent': accent,
        'gender': gender,
        'duration': len(audio_array) / target_sr
    }


def balance_dataset_by_accent(dataset, accents, samples_per_accent):
    """
    Balancea el dataset para tener una representación equitativa de cada acento.
    """
    balanced_examples = []

    for accent in accents:
        print(f"\nProcesando acento: {accent}")

        # Filtrar por acento
        accent_examples = dataset.filter(
            lambda x: x.get('annotated_accent', x.get('accent', '')) == accent
        )

        print(f"  - Muestras encontradas: {len(accent_examples)}")

        if samples_per_accent and len(accent_examples) > samples_per_accent:
            # Tomar una muestra aleatoria
            accent_examples = accent_examples.shuffle(seed=42).select(range(samples_per_accent))
            print(f"  - Muestras seleccionadas: {samples_per_accent}")

        balanced_examples.extend(accent_examples)

    return balanced_examples


def prepare_dataset(args):
    """
    Función principal para preparar el dataset.
    """
    print("="*60)
    print("PREPARACIÓN DE DATASET COMMON VOICE CATALÁN PARA ORPHEUS TTS")
    print("="*60)

    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar dataset
    dataset = load_common_voice_dataset(
        args.dataset_name,
        use_alternative=args.use_alternative_dataset
    )

    # Aplicar filtros de calidad
    print("\nFiltrando por calidad...")
    dataset = dataset.filter(filter_by_quality)
    print(f"Después del filtro de calidad: {len(dataset)} muestras")

    # Aplicar filtros de duración
    print("\nFiltrando por duración...")
    dataset = dataset.filter(
        lambda x: filter_by_duration(x, args.min_duration, args.max_duration, args.target_sample_rate)
    )
    print(f"Después del filtro de duración: {len(dataset)} muestras")

    # Balancear por acento
    if args.samples_per_accent:
        print("\nBalanceando dataset por acentos...")
        balanced_data = balance_dataset_by_accent(
            dataset,
            args.accents,
            args.samples_per_accent
        )
    else:
        balanced_data = dataset

    # Procesar ejemplos
    print("\nProcesando ejemplos al formato de Orpheus...")
    processed_examples = []

    for example in tqdm(balanced_data):
        try:
            # Determinar el acento del ejemplo
            accent = example.get('annotated_accent', example.get('accent', 'central'))

            processed = process_example(example, args.target_sample_rate, accent)
            processed_examples.append(processed)
        except Exception as e:
            print(f"Error procesando ejemplo: {e}")
            continue

    # Crear dataset final
    print("\nCreando dataset final...")
    final_dataset = Dataset.from_list(processed_examples)

    # Cast de columna de audio
    final_dataset = final_dataset.cast_column('audio', Audio(sampling_rate=args.target_sample_rate))

    # Dividir en train/validation
    print("\nDividiendo en train/validation (90/10)...")
    dataset_dict = final_dataset.train_test_split(test_size=0.1, seed=42)
    dataset_dict = DatasetDict({
        'train': dataset_dict['train'],
        'validation': dataset_dict['test']
    })

    # Guardar dataset
    print(f"\nGuardando dataset en: {output_dir}")
    dataset_dict.save_to_disk(str(output_dir))

    # Estadísticas finales
    print("\n" + "="*60)
    print("ESTADÍSTICAS DEL DATASET FINAL")
    print("="*60)
    print(f"Total de ejemplos (train): {len(dataset_dict['train'])}")
    print(f"Total de ejemplos (validation): {len(dataset_dict['validation'])}")

    # Estadísticas por acento
    accent_counts = {}
    for example in dataset_dict['train']:
        accent = example['accent']
        accent_counts[accent] = accent_counts.get(accent, 0) + 1

    print("\nDistribución por acento:")
    for accent, count in sorted(accent_counts.items()):
        print(f"  - {accent}: {count} muestras")

    # Estadísticas de duración
    durations = [ex['duration'] for ex in dataset_dict['train']]
    print(f"\nDuración promedio: {np.mean(durations):.2f}s")
    print(f"Duración total: {np.sum(durations)/3600:.2f} horas")

    # Subir a HuggingFace Hub si se especifica
    if args.push_to_hub:
        print(f"\nSubiendo dataset a HuggingFace Hub: {args.hub_dataset_name}")
        dataset_dict.push_to_hub(args.hub_dataset_name)

    print("\n¡Preparación completada!")
    return dataset_dict


if __name__ == '__main__':
    args = parse_args()
    prepare_dataset(args)
