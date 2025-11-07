"""
Script para preparar datasets agrupando por VARIANTE DIALECTAL (no por hablante).

Estrategia: Entrenar voces representativas de cada dialecto, luego usar
voice cloning para adaptar a hablantes específicos.

Esta estrategia es ideal cuando tienes muchos hablantes con pocos audios cada uno.

NOTA: Este script NO procesa/resamplea audio - lo deja tal cual está en el dataset original.
Solo formatea el texto y organiza los datos por dialecto.
"""

import argparse
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets, Audio
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict, Counter


# Mapeo de variantes a nombres de voz "genéricos" por dialecto
DIALECT_VOICE_NAMES = {
    'central': 'central',           # Dialecto central
    'balearic': 'balear',           # Dialecto balear
    'valencian': 'valencia',        # Dialecto valenciano
    'alacanti': 'alacanti',         # Alacantí
    'tortosi': 'tortosi',           # Tortosí
    'septentrional': 'septentrional',  # Septentrional
    'northwestern': 'occidental',   # Nord-occidental
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
        help='Número de workers para filtrado con multiprocessing'
    )
    parser.add_argument(
        '--hf_repo',
        type=str,
        default=None,
        help='Repositorio de HuggingFace para subir cada dialecto (ej: xaviviro/cv_23_ca_distilled)'
    )
    return parser.parse_args()


def get_variant_from_dataset_name(dataset_name):
    """Extrae la variante del nombre del dataset."""
    dataset_lower = dataset_name.lower()

    # Mapeo de patrones a variantes
    if 'central' in dataset_lower:
        return 'central'
    elif 'balear' in dataset_lower:
        return 'balearic'
    elif 'valencia' in dataset_lower and 'nord_occidental' not in dataset_lower:
        return 'valencian'
    elif 'alacant' in dataset_lower:
        return 'alacanti'
    elif 'tortosi' in dataset_lower:
        return 'tortosi'
    elif 'septentrional' in dataset_lower:
        return 'septentrional'
    elif 'nord_occidental' in dataset_lower or 'northwestern' in dataset_lower:
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




def balance_by_speakers(dataset, max_per_speaker):
    """
    Balancea el dataset limitando muestras por hablante.

    Esto evita que hablantes con muchos audios dominen el entrenamiento.
    """
    speaker_samples = defaultdict(list)

    # Agrupar por hablante
    for i, example in enumerate(tqdm(dataset, desc="📊 Agrupando por hablante")):
        speaker_id = example.get('client_id', 'unknown')
        speaker_samples[speaker_id].append(i)

    # Seleccionar hasta max_per_speaker de cada hablante
    selected_indices = []
    for speaker, indices in tqdm(speaker_samples.items(), desc="⚖️  Balanceando hablantes"):
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
    print("\n📥 Cargando dataset desde HuggingFace Hub...")
    try:
        dataset = load_dataset(dataset_name, split='train')
        print("✓ Dataset cargado exitosamente")
    except Exception as e:
        print(f"❌ Error cargando dataset: {e}")
        return None, None

    print(f"📊 Muestras iniciales: {len(dataset):,}")

    # Determinar número de workers para filtrado
    num_proc = args.num_workers if args.num_workers else None

    # Filtrar por duración
    print(f"\n🔍 Filtrando por duración ({args.min_duration}s - {args.max_duration}s)...")
    if num_proc:
        print(f"  Usando {num_proc} procesos")
    original_len = len(dataset)
    dataset = dataset.filter(
        lambda x: filter_by_duration(x, args.min_duration, args.max_duration),
        num_proc=num_proc,
        desc="Filtrando por duración"
    )
    filtered_len = len(dataset)
    print(f"✓ Filtrado completado: {filtered_len}/{original_len} muestras ({filtered_len/original_len*100:.1f}% conservadas)")

    # Analizar hablantes
    print("\nAnalizando hablantes...")
    speaker_counts = Counter(
        ex.get('client_id', 'unknown')
        for ex in tqdm(dataset, desc="🔍 Analizando hablantes")
    )

    # Filtrar hablantes con muy pocas muestras
    valid_speakers = {
        speaker for speaker, count in speaker_counts.items()
        if count >= args.min_samples_per_speaker
    }

    print(f"Hablantes totales: {len(speaker_counts)}")
    print(f"Hablantes con >={args.min_samples_per_speaker} muestras: {len(valid_speakers)}")

    print("\n🔍 Filtrando por hablantes válidos...")
    if num_proc:
        print(f"  Usando {num_proc} procesos")
    original_filtered = len(dataset)
    dataset = dataset.filter(
        lambda x: x.get('client_id', 'unknown') in valid_speakers,
        num_proc=num_proc,
        desc="Filtrando hablantes"
    )
    after_speaker_filter = len(dataset)
    print(f"✓ Muestras después de filtrar hablantes: {after_speaker_filter}/{original_filtered} ({after_speaker_filter/original_filtered*100:.1f}% conservadas)")

    # Balancear por hablantes si se solicita
    if args.balance_speakers:
        print(f"\n⚖️  Balanceando (max {args.max_samples_per_speaker} por hablante)...")
        before_balance = len(dataset)
        dataset = balance_by_speakers(dataset, args.max_samples_per_speaker)
        after_balance = len(dataset)
        print(f"✓ Muestras después de balancear: {after_balance}/{before_balance} ({after_balance/before_balance*100:.1f}% conservadas)")

    # Limitar número total si se especifica
    if args.samples_per_dialect and len(dataset) > args.samples_per_dialect:
        print(f"\nLimitando a {args.samples_per_dialect} muestras...")
        dataset = dataset.shuffle(seed=42).select(range(args.samples_per_dialect))

    # Guardar metadata de hablantes (para voice cloning posterior)
    speaker_metadata = None
    if args.save_speaker_metadata:
        print("\n📋 Guardando metadata de hablantes...")
        speaker_metadata = {}

        for example in tqdm(dataset, desc="📝 Extrayendo metadata"):
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
        for i, example in enumerate(tqdm(dataset, desc="🎯 Seleccionando ejemplos representativos")):
            speaker_id = example.get('client_id', 'unknown')
            if len(speaker_samples[speaker_id]) < 3:
                speaker_samples[speaker_id].append(i)

        for speaker_id, indices in speaker_samples.items():
            speaker_metadata[speaker_id]['representative_indices'] = indices

        print(f"✓ Metadata guardada para {len(speaker_metadata)} hablantes")

    # Procesar ejemplos - extraer solo los numpy arrays para evitar errores de stream
    print("\n🔄 Extrayendo ejemplos...")

    # Extraer datos (sin formatear el texto con dialecto)
    processed_examples = []
    for example in tqdm(dataset, desc="📦 Extrayendo datos"):
        try:
            # Extraer datos básicos del audio
            audio_array = example['audio']['array']
            sample_rate = example['audio']['sampling_rate']
            text = example['sentence']  # Sin prefijo de dialecto
            speaker_id = example.get('client_id', 'unknown')

            duration = len(audio_array) / sample_rate

            processed_examples.append({
                'audio': {
                    'array': audio_array.astype(np.float32),
                    'sampling_rate': sample_rate
                },
                'text': text,  # Texto sin modificar
                'duration': duration,
                'speaker_id': speaker_id,
            })
        except Exception as e:
            print(f"Error procesando ejemplo: {e}")
            continue

    if not processed_examples:
        print(f"⚠️  No se procesaron ejemplos para {dataset_name}")
        return None, None

    print(f"✅ Ejemplos procesados exitosamente: {len(processed_examples)}/{len(dataset)} ({len(processed_examples)/len(dataset)*100:.1f}%)")

    # Crear dataset directamente desde la lista de dicts
    print("\n🔨 Creando HuggingFace Dataset...")
    processed_dataset = Dataset.from_list(processed_examples)
    print(f"✓ Dataset creado con {len(processed_dataset)} ejemplos")

    # Aplicar Audio feature para que se suba correctamente a HuggingFace
    print("\n🎵 Aplicando Audio feature...")
    processed_dataset = processed_dataset.cast_column('audio', Audio())
    print("✅ Audio feature aplicada correctamente")

    print(f"\n✅ Dataset procesado: {len(processed_dataset):,} ejemplos")

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
    print(f"\n🔄 Procesando {len(args.datasets)} dataset(s)...")
    for idx, dataset_name in enumerate(args.datasets, 1):
        print(f"\n{'='*70}")
        print(f"📦 Dataset {idx}/{len(args.datasets)}: {dataset_name}")
        print(f"{'='*70}")

        processed_ds, speaker_meta = load_and_process_dataset(dataset_name, args)

        if processed_ds is not None:
            all_datasets.append(processed_ds)

            dialect = get_variant_from_dataset_name(dataset_name)

            # Guardar cada dialecto por separado
            dialect_dir = output_dir / dialect
            print(f"\n💾 Guardando dialecto '{dialect}' en: {dialect_dir}")
            processed_ds.save_to_disk(str(dialect_dir))
            print("✓ Dialecto guardado localmente")

            # Subir a HuggingFace si se especificó repo
            if args.hf_repo:
                try:
                    print(f"\n☁️  Subiendo '{dialect}' a HuggingFace Hub...")
                    print(f"   Repo: {args.hf_repo}")
                    print(f"   Config: {dialect}")
                    processed_ds.push_to_hub(
                        args.hf_repo,
                        config_name=dialect,
                        commit_message=f"Add {dialect} dialect (max {args.max_samples_per_speaker} samples/speaker)"
                    )
                    print(f"✅ Dialecto '{dialect}' subido exitosamente")
                    print(f"   URL: https://huggingface.co/datasets/{args.hf_repo}/tree/main/{dialect}")
                except Exception as e:
                    print(f"⚠️  Error subiendo '{dialect}' a HuggingFace: {e}")
                    print("   Continuando con el siguiente dialecto...")

            # Guardar estadísticas básicas (sin iterar sobre el dataset)
            dialect_stats[dialect] = {
                'samples': len(processed_ds),
            }

            if speaker_meta:
                all_speaker_metadata[dialect] = speaker_meta

    if not all_datasets:
        print("\n❌ No se pudo procesar ningún dataset")
        return

    # Combinar datasets
    print("\n" + "="*70)
    print("🔗 COMBINANDO DIALECTOS")
    print("="*70)

    print(f"\n📦 Concatenando {len(all_datasets)} datasets...")
    combined_dataset = concatenate_datasets(all_datasets)
    print(f"✓ Total combinado: {len(combined_dataset):,} muestras")

    # Mezclar
    print("\n🔀 Mezclando dataset...")
    combined_dataset = combined_dataset.shuffle(seed=42)
    print("✓ Dataset mezclado")

    # Dividir train/validation
    print("\n✂️  Dividiendo train/validation (90/10)...")
    split = combined_dataset.train_test_split(test_size=0.1, seed=42)
    print(f"✓ Train: {len(split['train']):,} muestras")
    print(f"✓ Validation: {len(split['test']):,} muestras")

    dataset_dict = DatasetDict({
        'train': split['train'],
        'validation': split['test']
    })

    # Guardar
    print(f"\n💾 Guardando dataset en: {output_dir}")
    dataset_dict.save_to_disk(str(output_dir))
    print("✓ Dataset guardado en disco")

    # Guardar metadata de hablantes
    if all_speaker_metadata:
        print("\n📝 Guardando metadata de hablantes...")
        metadata_path = output_dir / 'speaker_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(all_speaker_metadata, f, indent=2, ensure_ascii=False)
        print(f"✓ Metadata guardada: {metadata_path}")

    # Guardar estadísticas
    print("\n📊 Guardando estadísticas...")
    stats_path = output_dir / 'dialect_statistics.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(dialect_stats, f, indent=2, ensure_ascii=False)
    print(f"✓ Estadísticas guardadas: {stats_path}")

    # Reporte final
    print("\n" + "="*70)
    print("📊 ESTADÍSTICAS FINALES")
    print("="*70)
    print("\n📚 Dataset completo:")
    print(f"  • Train: {len(dataset_dict['train']):,} muestras")
    print(f"  • Validation: {len(dataset_dict['validation']):,} muestras")
    print(f"  • Total: {len(dataset_dict['train']) + len(dataset_dict['validation']):,} muestras")

    print("\n🗣️  Por dialecto:")
    total_samples = 0
    for dialect, stats in sorted(dialect_stats.items()):
        voice_name = DIALECT_VOICE_NAMES.get(dialect, dialect)
        print(f"  📍 {dialect.upper()} (voz: '{voice_name}'): {stats['samples']:,} muestras")
        total_samples += stats['samples']

    print(f"\n  🎯 TOTAL: {len(dialect_stats)} dialectos, {total_samples:,} muestras")

    print("\n" + "="*70)
    print("✅ PREPARACIÓN COMPLETADA CON ÉXITO")
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
