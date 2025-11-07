"""
Script para preparar datasets agrupando por VARIANTE DIALECTAL (no por hablante).

Estrategia: Entrenar voces representativas de cada dialecto, luego usar
voice cloning para adaptar a hablantes específicos.

Esta estrategia es ideal cuando tienes muchos hablantes con pocos audios cada uno.
"""

import os
import argparse
from datasets import load_dataset, Dataset, Audio, DatasetDict, concatenate_datasets
from pathlib import Path
import librosa
import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial


# Mapeo de variantes a nombres de voz "genéricos" por dialecto
DIALECT_VOICE_NAMES = {
    'central': 'central',       # Voz genérica del dialecto central
    'balearic': 'balear',       # Voz genérica del dialecto balear
    'northern': 'nord',         # Voz genérica del dialecto norte
    'northwestern': 'occidental',  # Voz genérica del dialecto noroccidental
    'valencian': 'valencia'     # Voz genérica del dialecto valenciano
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preparar datasets por variante dialectal (estrategia voice cloning)'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        required=True,
        help='Lista de datasets a procesar'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../data/processed_by_dialect',
        help='Directorio de salida'
    )
    parser.add_argument(
        '--target_sample_rate',
        type=int,
        default=24000,
        help='Frecuencia de muestreo objetivo'
    )
    parser.add_argument(
        '--min_duration',
        type=float,
        default=1.0,
        help='Duración mínima del audio'
    )
    parser.add_argument(
        '--max_duration',
        type=float,
        default=20.0,
        help='Duración máxima del audio'
    )
    parser.add_argument(
        '--samples_per_dialect',
        type=int,
        default=None,
        help='Número máximo de muestras por dialecto (None para todas)'
    )
    parser.add_argument(
        '--min_samples_per_speaker',
        type=int,
        default=3,
        help='Mínimo de muestras por hablante para incluirlo'
    )
    parser.add_argument(
        '--balance_speakers',
        action='store_true',
        help='Balancear muestras entre hablantes del mismo dialecto'
    )
    parser.add_argument(
        '--max_samples_per_speaker',
        type=int,
        default=50,
        help='Máximo de muestras por hablante (para balancear)'
    )
    parser.add_argument(
        '--save_speaker_metadata',
        action='store_true',
        help='Guardar metadata de hablantes para posterior voice cloning'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Número de workers para multiprocessing (None = auto-detect)'
    )

    return parser.parse_args()


def get_variant_from_dataset_name(dataset_name):
    """Extrae la variante del nombre del dataset."""
    if 'central' in dataset_name.lower():
        return 'central'
    elif 'balearic' in dataset_name.lower() or 'balear' in dataset_name.lower():
        return 'balearic'
    elif 'valencian' in dataset_name.lower() or 'valencia' in dataset_name.lower():
        return 'valencian'
    elif 'northern' in dataset_name.lower() or 'nord' in dataset_name.lower():
        return 'northern'
    elif 'northwestern' in dataset_name.lower():
        return 'northwestern'
    return 'unknown'


def filter_by_duration(example, min_dur, max_dur):
    """Filtra por duración."""
    if 'audio' not in example or example['audio'] is None:
        return False
    audio_array = example['audio']['array']
    sample_rate = example['audio']['sampling_rate']
    duration = len(audio_array) / sample_rate
    return min_dur <= duration <= max_dur


def resample_audio(audio_array, orig_sr, target_sr):
    """Resamplea audio."""
    if orig_sr == target_sr:
        return audio_array
    return librosa.resample(audio_array, orig_sr=orig_sr, target_sr=target_sr)


def process_example(example, target_sr, dialect_name, speaker_id=None):
    """
    Procesa ejemplo con formato de dialecto.

    El texto se formatea como: {dialect_voice}: {texto}
    Por ejemplo: "central: Bon dia"

    Guardamos también el speaker_id para posterior voice cloning.
    """
    try:
        audio_data = example['audio']
        text = example['sentence']

        # Si speaker_id no viene como argumento, extraerlo del ejemplo
        if speaker_id is None:
            speaker_id = example.get('client_id', 'unknown')

        # Resamplear
        audio_array = audio_data['array']
        orig_sr = audio_data['sampling_rate']

        if orig_sr != target_sr:
            audio_array = resample_audio(audio_array, orig_sr, target_sr)

        # Nombre de voz basado en DIALECTO (no en hablante)
        voice_name = DIALECT_VOICE_NAMES.get(dialect_name, dialect_name)

        # Formato para entrenamiento: "{dialect_voice}: {text}"
        formatted_text = f"{voice_name}: {text}"

        duration = len(audio_array) / target_sr

        return {
            'audio': {
                'array': audio_array.astype(np.float32),  # Asegurar tipo correcto
                'sampling_rate': target_sr
            },
            'text': formatted_text,              # Para entrenamiento
            'original_text': text,               # Texto original
            'voice_name': voice_name,            # Nombre de voz (dialecto)
            'dialect': dialect_name,             # Dialecto
            'speaker_id': speaker_id,            # ID del hablante (para voice cloning)
            'gender': example.get('gender', 'unknown'),
            'age': example.get('age', 'unknown'),
            'duration': duration,
        }

    except Exception as e:
        print(f"Error procesando ejemplo: {e}")
        return None


def process_batch(batch, target_sr, dialect_name):
    """
    Procesa un batch de ejemplos en paralelo.

    Esta función se ejecuta en múltiples procesos workers.
    """
    results = []
    for example in batch:
        processed = process_example(example, target_sr, dialect_name)
        if processed is not None:
            results.append(processed)
    return results


def balance_by_speakers(dataset, max_per_speaker):
    """
    Balancea el dataset limitando muestras por hablante.

    Esto evita que hablantes con muchos audios dominen el entrenamiento.
    """
    speaker_samples = defaultdict(list)

    # Agrupar por hablante
    for i, example in enumerate(dataset):
        speaker_id = example.get('client_id', 'unknown')
        speaker_samples[speaker_id].append(i)

    # Seleccionar hasta max_per_speaker de cada hablante
    selected_indices = []
    for speaker, indices in speaker_samples.items():
        if len(indices) <= max_per_speaker:
            selected_indices.extend(indices)
        else:
            # Tomar muestra aleatoria
            import random
            random.seed(42)
            selected_indices.extend(random.sample(indices, max_per_speaker))

    return dataset.select(sorted(selected_indices))


def load_and_process_dataset(dataset_name, args):
    """Carga y procesa un dataset por dialecto."""
    print(f"\n{'='*70}")
    print(f"Procesando: {dataset_name}")
    print(f"{'='*70}")

    # Detectar dialecto
    dialect = get_variant_from_dataset_name(dataset_name)
    print(f"Dialecto detectado: {dialect}")
    print(f"Voz de entrenamiento: {DIALECT_VOICE_NAMES.get(dialect, dialect)}")

    # Cargar dataset
    print("Cargando dataset...")
    try:
        dataset = load_dataset(dataset_name, split='train')
    except Exception as e:
        print(f"Error: {e}")
        return None, None

    print(f"Muestras cargadas: {len(dataset)}")

    # Filtrar por duración
    print("\nFiltrando por duración...")
    dataset = dataset.filter(
        lambda x: filter_by_duration(x, args.min_duration, args.max_duration),
        desc="Filtrando"
    )
    print(f"Después del filtro: {len(dataset)} muestras")

    # Analizar hablantes
    print("\nAnalizando hablantes...")
    from collections import Counter
    speaker_counts = Counter(ex.get('client_id', 'unknown') for ex in dataset)

    # Filtrar hablantes con muy pocas muestras
    valid_speakers = {
        speaker for speaker, count in speaker_counts.items()
        if count >= args.min_samples_per_speaker
    }

    print(f"Hablantes totales: {len(speaker_counts)}")
    print(f"Hablantes con >={args.min_samples_per_speaker} muestras: {len(valid_speakers)}")

    dataset = dataset.filter(
        lambda x: x.get('client_id', 'unknown') in valid_speakers,
        desc="Filtrando hablantes"
    )
    print(f"Muestras después de filtrar hablantes: {len(dataset)}")

    # Balancear por hablantes si se solicita
    if args.balance_speakers:
        print(f"\nBalanceando (max {args.max_samples_per_speaker} por hablante)...")
        dataset = balance_by_speakers(dataset, args.max_samples_per_speaker)
        print(f"Muestras después de balancear: {len(dataset)}")

    # Limitar número total si se especifica
    if args.samples_per_dialect and len(dataset) > args.samples_per_dialect:
        print(f"\nLimitando a {args.samples_per_dialect} muestras...")
        dataset = dataset.shuffle(seed=42).select(range(args.samples_per_dialect))

    # Guardar metadata de hablantes (para voice cloning posterior)
    speaker_metadata = None
    if args.save_speaker_metadata:
        print("\nGuardando metadata de hablantes...")
        speaker_metadata = {}

        for example in dataset:
            speaker_id = example.get('client_id', 'unknown')
            if speaker_id not in speaker_metadata:
                speaker_metadata[speaker_id] = {
                    'dialect': dialect,
                    'samples': [],
                    'gender': example.get('gender', 'unknown'),
                    'age': example.get('age', 'unknown'),
                }

        # Agregar ejemplos representativos (primeros 3 de cada hablante)
        speaker_samples = defaultdict(list)
        for i, example in enumerate(dataset):
            speaker_id = example.get('client_id', 'unknown')
            if len(speaker_samples[speaker_id]) < 3:
                speaker_samples[speaker_id].append(i)

        for speaker_id, indices in speaker_samples.items():
            speaker_metadata[speaker_id]['representative_indices'] = indices

    # Procesar ejemplos con multiprocessing
    print("\nProcesando ejemplos...")

    # Determinar número de workers
    num_workers = args.num_workers if args.num_workers is not None else cpu_count()
    print(f"Usando {num_workers} workers para procesamiento paralelo")

    # Convertir dataset a lista para dividir en batches
    dataset_list = list(dataset)
    batch_size = max(1, len(dataset_list) // (num_workers * 4))  # 4 batches por worker
    print(f"Batch size: {batch_size}")

    # Dividir en batches
    batches = []
    for i in range(0, len(dataset_list), batch_size):
        batches.append(dataset_list[i:i + batch_size])

    print(f"Total batches: {len(batches)}")

    # Procesar en paralelo
    processed_examples = []
    process_func = partial(process_batch, target_sr=args.target_sample_rate, dialect_name=dialect)

    with Pool(processes=num_workers) as pool:
        # Usar imap_unordered para ver progreso
        for batch_results in tqdm(
            pool.imap_unordered(process_func, batches),
            total=len(batches),
            desc=f"Procesando {dialect}"
        ):
            processed_examples.extend(batch_results)

    if not processed_examples:
        print(f"⚠️  No se procesaron ejemplos para {dataset_name}")
        return None, None

    print(f"Ejemplos procesados exitosamente: {len(processed_examples)}")

    # Crear dataset procesado
    # En lugar de cast_column (que causa el error), crear directamente con Audio feature
    processed_dataset = Dataset.from_dict({
        'audio': [ex['audio']['array'] for ex in processed_examples],
        'text': [ex['text'] for ex in processed_examples],
        'original_text': [ex['original_text'] for ex in processed_examples],
        'voice_name': [ex['voice_name'] for ex in processed_examples],
        'dialect': [ex['dialect'] for ex in processed_examples],
        'speaker_id': [ex['speaker_id'] for ex in processed_examples],
        'gender': [ex['gender'] for ex in processed_examples],
        'age': [ex['age'] for ex in processed_examples],
        'duration': [ex['duration'] for ex in processed_examples],
    }).cast_column('audio', Audio(sampling_rate=args.target_sample_rate))

    print(f"✅ Dataset procesado: {len(processed_dataset)} ejemplos")

    # Estadísticas
    final_speaker_counts = Counter(ex['speaker_id'] for ex in processed_dataset)
    print(f"\nEstadísticas finales:")
    print(f"  Hablantes únicos: {len(final_speaker_counts)}")
    print(f"  Promedio muestras/hablante: {len(processed_dataset)/len(final_speaker_counts):.1f}")
    print(f"  Duración total: {sum(ex['duration'] for ex in processed_dataset)/3600:.2f} horas")

    return processed_dataset, speaker_metadata


def main():
    args = parse_args()

    print("="*70)
    print("PREPARACIÓN POR DIALECTO (Estrategia Voice Cloning)")
    print("="*70)
    print("\nEstrategia:")
    print("1. Entrenar voces por DIALECTO (no por hablante)")
    print("2. Usar todos los hablantes para cada dialecto")
    print("3. Después: Voice cloning para voces específicas")
    print("="*70)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_datasets = []
    all_speaker_metadata = {}
    dialect_stats = {}

    # Procesar cada dataset
    for dataset_name in args.datasets:
        processed_ds, speaker_meta = load_and_process_dataset(dataset_name, args)

        if processed_ds is not None:
            all_datasets.append(processed_ds)

            dialect = get_variant_from_dataset_name(dataset_name)
            dialect_stats[dialect] = {
                'samples': len(processed_ds),
                'speakers': len(set(ex['speaker_id'] for ex in processed_ds)),
                'duration_hours': sum(ex['duration'] for ex in processed_ds) / 3600
            }

            if speaker_meta:
                all_speaker_metadata[dialect] = speaker_meta

    if not all_datasets:
        print("\n❌ No se pudo procesar ningún dataset")
        return

    # Combinar datasets
    print("\n" + "="*70)
    print("COMBINANDO DIALECTOS")
    print("="*70)

    combined_dataset = concatenate_datasets(all_datasets)
    print(f"Total combinado: {len(combined_dataset)} muestras")

    # Mezclar
    print("Mezclando...")
    combined_dataset = combined_dataset.shuffle(seed=42)

    # Dividir train/validation
    print("Dividiendo train/validation (90/10)...")
    split = combined_dataset.train_test_split(test_size=0.1, seed=42)

    dataset_dict = DatasetDict({
        'train': split['train'],
        'validation': split['test']
    })

    # Guardar
    print(f"\nGuardando en: {output_dir}")
    dataset_dict.save_to_disk(str(output_dir))

    # Guardar metadata de hablantes
    if all_speaker_metadata:
        metadata_path = output_dir / 'speaker_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(all_speaker_metadata, f, indent=2, ensure_ascii=False)
        print(f"Metadata de hablantes guardada: {metadata_path}")

    # Guardar estadísticas
    stats_path = output_dir / 'dialect_statistics.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(dialect_stats, f, indent=2, ensure_ascii=False)

    # Reporte final
    print("\n" + "="*70)
    print("ESTADÍSTICAS FINALES")
    print("="*70)
    print(f"Train: {len(dataset_dict['train'])} muestras")
    print(f"Validation: {len(dataset_dict['validation'])} muestras")

    print("\nPor dialecto:")
    for dialect, stats in dialect_stats.items():
        voice_name = DIALECT_VOICE_NAMES.get(dialect, dialect)
        print(f"  {dialect} (voz: '{voice_name}'):")
        print(f"    • Muestras: {stats['samples']}")
        print(f"    • Hablantes: {stats['speakers']}")
        print(f"    • Duración: {stats['duration_hours']:.2f} horas")

    print("\n" + "="*70)
    print("✅ PREPARACIÓN COMPLETADA")
    print("="*70)
    print("\nPróximos pasos:")
    print("1. Tokenizar dataset:")
    print(f"   python scripts/tokenize_dataset.py --input_dir {output_dir} --output_dir {output_dir}_tokenized")
    print("\n2. Entrenar modelo:")
    print("   accelerate launch scripts/train_catalan.py --config configs/config_catalan.yaml")
    print("\n3. Después del entrenamiento, usar voice cloning:")
    print("   Ver: notebooks/voice_cloning_inference.ipynb")


if __name__ == '__main__':
    main()
