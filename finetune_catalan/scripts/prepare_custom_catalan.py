"""
Script para preparar datasets personalizados de catalán (formato xaviviro/cv_23_ca_*)
para fine-tuning de Orpheus TTS.

Este script está optimizado para trabajar con los datasets de Common Voice 23
organizados por variante dialectal.
"""

import os
import argparse
from datasets import load_dataset, Dataset, Audio, DatasetDict, concatenate_datasets
from pathlib import Path
import librosa
import numpy as np
from tqdm import tqdm
import json

# Mapeo de variantes dialectales a nombres de voz
VOICE_NAMES_BY_VARIANT = {
    'central': 'pau',
    'balearic': 'maria',
    'northern': 'montse',
    'northwestern': 'jordi',
    'valencian': 'carla'
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preparar datasets personalizados de catalán para Orpheus TTS'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        required=True,
        help='Lista de datasets a procesar (ej: xaviviro/cv_23_ca_central xaviviro/cv_23_ca_balearic)'
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
        '--samples_per_variant',
        type=int,
        default=None,
        help='Número de muestras por variante (None para todas)'
    )
    parser.add_argument(
        '--filter_by_age',
        type=str,
        nargs='+',
        default=None,
        help='Filtrar por edades específicas (ej: twenties thirties)'
    )
    parser.add_argument(
        '--filter_by_gender',
        type=str,
        nargs='+',
        default=None,
        help='Filtrar por género (ej: male female)'
    )
    parser.add_argument(
        '--push_to_hub',
        action='store_true',
        help='Subir el dataset procesado a Hugging Face Hub'
    )
    parser.add_argument(
        '--hub_dataset_name',
        type=str,
        default='orpheus-catalan-multidialect',
        help='Nombre del dataset en HuggingFace Hub'
    )
    parser.add_argument(
        '--train_split_ratio',
        type=float,
        default=0.9,
        help='Ratio de datos para entrenamiento (0.9 = 90% train, 10% validation)'
    )

    return parser.parse_args()


def get_variant_from_dataset_name(dataset_name):
    """
    Extrae la variante dialectal del nombre del dataset.
    Ej: 'xaviviro/cv_23_ca_central' -> 'central'
    """
    parts = dataset_name.split('_')
    if 'central' in dataset_name.lower():
        return 'central'
    elif 'balearic' in dataset_name.lower() or 'balear' in dataset_name.lower():
        return 'balearic'
    elif 'northern' in dataset_name.lower() or 'nord' in dataset_name.lower():
        return 'northern'
    elif 'northwestern' in dataset_name.lower() or 'occidental' in dataset_name.lower():
        return 'northwestern'
    elif 'valencian' in dataset_name.lower() or 'valencia' in dataset_name.lower():
        return 'valencian'
    else:
        # Por defecto, usar el último componente del nombre
        return parts[-1] if parts else 'unknown'


def filter_by_duration(example, min_dur, max_dur, target_sr=24000):
    """
    Filtra por duración del audio.
    """
    if 'audio' not in example or example['audio'] is None:
        return False

    audio_array = example['audio']['array']
    sample_rate = example['audio']['sampling_rate']

    duration = len(audio_array) / sample_rate
    return min_dur <= duration <= max_dur


def filter_by_metadata(example, age_filter=None, gender_filter=None):
    """
    Filtra por metadatos de edad y género.
    """
    if age_filter:
        age = example.get('age', '')
        if age and age not in age_filter:
            return False

    if gender_filter:
        gender = example.get('gender', '')
        if gender and gender not in gender_filter:
            return False

    return True


def resample_audio(audio_array, orig_sr, target_sr):
    """
    Resamplea el audio a la frecuencia objetivo.
    """
    if orig_sr == target_sr:
        return audio_array

    return librosa.resample(audio_array, orig_sr=orig_sr, target_sr=target_sr)


def process_example(example, target_sr, variant):
    """
    Procesa un ejemplo individual para el formato de Orpheus.
    """
    try:
        # Obtener audio y texto
        audio_data = example['audio']
        text = example['sentence']

        # Resamplear si es necesario
        audio_array = audio_data['array']
        orig_sr = audio_data['sampling_rate']

        if orig_sr != target_sr:
            audio_array = resample_audio(audio_array, orig_sr, target_sr)

        # Asignar nombre de voz basado en la variante
        voice_name = VOICE_NAMES_BY_VARIANT.get(variant, 'speaker')

        # Formatear el texto según el formato de Orpheus
        formatted_text = f"{voice_name}: {text}"

        # Extraer metadatos
        duration = len(audio_array) / target_sr

        return {
            'audio': {
                'array': audio_array,
                'sampling_rate': target_sr
            },
            'text': formatted_text,
            'original_text': text,
            'voice_name': voice_name,
            'variant': variant,
            'gender': example.get('gender', 'unknown'),
            'age': example.get('age', 'unknown'),
            'accents': example.get('accents', ''),
            'locale': example.get('locale', 'ca'),
            'duration': duration,
            'client_id': example.get('client_id', ''),
        }

    except Exception as e:
        print(f"Error procesando ejemplo: {e}")
        return None


def load_and_process_dataset(dataset_name, args):
    """
    Carga y procesa un dataset individual.
    """
    print(f"\n{'='*60}")
    print(f"Procesando dataset: {dataset_name}")
    print(f"{'='*60}")

    # Determinar variante dialectal
    variant = get_variant_from_dataset_name(dataset_name)
    print(f"Variante detectada: {variant}")

    # Cargar dataset
    print("Cargando dataset...")
    try:
        dataset = load_dataset(dataset_name, split='train')
    except Exception as e:
        print(f"Error cargando dataset: {e}")
        return None

    print(f"Dataset cargado: {len(dataset)} muestras")

    # Aplicar filtros de duración
    print("\nFiltrando por duración...")
    dataset = dataset.filter(
        lambda x: filter_by_duration(x, args.min_duration, args.max_duration, args.target_sample_rate),
        desc="Filtrando duración"
    )
    print(f"Después del filtro de duración: {len(dataset)} muestras")

    # Aplicar filtros de metadatos
    if args.filter_by_age or args.filter_by_gender:
        print("\nFiltrando por metadatos...")
        dataset = dataset.filter(
            lambda x: filter_by_metadata(x, args.filter_by_age, args.filter_by_gender),
            desc="Filtrando metadatos"
        )
        print(f"Después del filtro de metadatos: {len(dataset)} muestras")

    # Limitar número de muestras si se especifica
    if args.samples_per_variant and len(dataset) > args.samples_per_variant:
        print(f"\nSeleccionando {args.samples_per_variant} muestras aleatorias...")
        dataset = dataset.shuffle(seed=42).select(range(args.samples_per_variant))

    # Procesar ejemplos
    print("\nProcesando ejemplos al formato de Orpheus...")
    processed_examples = []

    for example in tqdm(dataset, desc=f"Procesando {variant}"):
        processed = process_example(example, args.target_sample_rate, variant)
        if processed is not None:
            processed_examples.append(processed)

    if not processed_examples:
        print(f"ADVERTENCIA: No se procesaron ejemplos para {dataset_name}")
        return None

    # Crear dataset procesado
    processed_dataset = Dataset.from_list(processed_examples)
    processed_dataset = processed_dataset.cast_column(
        'audio',
        Audio(sampling_rate=args.target_sample_rate)
    )

    print(f"✓ Dataset procesado: {len(processed_dataset)} ejemplos")

    return processed_dataset


def prepare_datasets(args):
    """
    Función principal para preparar múltiples datasets.
    """
    print("="*60)
    print("PREPARACIÓN DE DATASETS CATALANES PERSONALIZADOS")
    print("="*60)

    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Procesar cada dataset
    processed_datasets = []
    variant_stats = {}

    for dataset_name in args.datasets:
        processed = load_and_process_dataset(dataset_name, args)
        if processed is not None:
            processed_datasets.append(processed)

            # Recopilar estadísticas
            variant = get_variant_from_dataset_name(dataset_name)
            variant_stats[variant] = {
                'samples': len(processed),
                'total_duration': sum(ex['duration'] for ex in processed)
            }

    if not processed_datasets:
        print("\nERROR: No se pudo procesar ningún dataset")
        return None

    # Combinar todos los datasets
    print("\n" + "="*60)
    print("COMBINANDO DATASETS")
    print("="*60)

    combined_dataset = concatenate_datasets(processed_datasets)
    print(f"Total de ejemplos combinados: {len(combined_dataset)}")

    # Mezclar dataset
    print("\nMezclando ejemplos...")
    combined_dataset = combined_dataset.shuffle(seed=42)

    # Dividir en train/validation
    print(f"\nDividiendo en train/validation ({args.train_split_ratio*100:.0f}/{(1-args.train_split_ratio)*100:.0f})...")
    split = combined_dataset.train_test_split(
        test_size=(1 - args.train_split_ratio),
        seed=42
    )

    dataset_dict = DatasetDict({
        'train': split['train'],
        'validation': split['test']
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

    # Estadísticas por variante
    print("\nDistribución por variante:")
    for variant, stats in sorted(variant_stats.items()):
        duration_hours = stats['total_duration'] / 3600
        print(f"  - {variant}:")
        print(f"      Muestras: {stats['samples']}")
        print(f"      Duración: {duration_hours:.2f} horas")

    # Estadísticas de duración total
    total_duration = sum(ex['duration'] for ex in dataset_dict['train'])
    avg_duration = total_duration / len(dataset_dict['train'])

    print(f"\nDuración promedio: {avg_duration:.2f}s")
    print(f"Duración total (train): {total_duration/3600:.2f} horas")

    # Estadísticas de género (si disponible)
    gender_counts = {}
    for example in dataset_dict['train']:
        gender = example.get('gender', 'unknown')
        gender_counts[gender] = gender_counts.get(gender, 0) + 1

    if gender_counts:
        print("\nDistribución por género:")
        for gender, count in sorted(gender_counts.items()):
            print(f"  - {gender}: {count} muestras ({count/len(dataset_dict['train'])*100:.1f}%)")

    # Guardar metadatos
    metadata = {
        'datasets': args.datasets,
        'variant_stats': variant_stats,
        'total_samples': len(combined_dataset),
        'train_samples': len(dataset_dict['train']),
        'validation_samples': len(dataset_dict['validation']),
        'target_sample_rate': args.target_sample_rate,
        'min_duration': args.min_duration,
        'max_duration': args.max_duration,
    }

    metadata_path = output_dir / 'dataset_info.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nMetadatos guardados en: {metadata_path}")

    # Subir a HuggingFace Hub si se especifica
    if args.push_to_hub:
        print(f"\nSubiendo dataset a HuggingFace Hub: {args.hub_dataset_name}")
        try:
            dataset_dict.push_to_hub(args.hub_dataset_name)
            print("✓ Dataset subido exitosamente")
        except Exception as e:
            print(f"Error subiendo dataset: {e}")

    print("\n¡Preparación completada!")
    return dataset_dict


if __name__ == '__main__':
    args = parse_args()
    prepare_datasets(args)
