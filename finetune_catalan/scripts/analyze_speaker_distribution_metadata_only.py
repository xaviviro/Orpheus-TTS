"""
Script para analizar la distribución de hablantes usando solo metadatos (sin cargar audio).

Este script evita cargar audio completamente para manejar datasets muy grandes.
Lee directamente los archivos Parquet de HuggingFace sin usar la librería datasets.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analizar distribución de hablantes (solo metadata)'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        required=True,
        help='Lista de datasets a analizar (formato: user/dataset)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./analysis',
        help='Directorio para guardar análisis'
    )
    parser.add_argument(
        '--min_samples_fixed',
        type=int,
        default=100,
        help='Mínimo de muestras para considerar como voz fija'
    )
    parser.add_argument(
        '--min_samples_multispeaker',
        type=int,
        default=5,
        help='Mínimo de muestras para entrenamiento multi-speaker'
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


def analyze_dataset_from_parquet(dataset_name, args):
    """
    Analiza dataset leyendo directamente los archivos Parquet.

    Esto evita cargar audio y es mucho más memory-efficient.
    """
    print(f"\n{'='*70}")
    print(f"Analizando: {dataset_name}")
    print(f"{'='*70}")

    try:
        from huggingface_hub import HfFileSystem

        # Conectar al filesystem de HuggingFace
        fs = HfFileSystem()

        # Listar archivos parquet del dataset
        print("Buscando archivos de datos...")
        dataset_path = f"datasets/{dataset_name}"

        # Buscar archivos parquet en el repositorio
        files = fs.ls(dataset_path, detail=False)
        parquet_files = [
            f for f in files
            if f.endswith('.parquet') and 'train' in f
        ]

        if not parquet_files:
            # Intentar en subdirectorio data/
            try:
                files = fs.ls(f"{dataset_path}/data", detail=False)
                parquet_files = [
                    f for f in files
                    if f.endswith('.parquet') and 'train' in f
                ]
            except:
                pass

        if not parquet_files:
            print(f"⚠️  No se encontraron archivos parquet en {dataset_name}")
            print("   Intentando con datasets.load_dataset (modo lectura limitada)...")
            return analyze_dataset_limited(dataset_name, args)

        print(f"Encontrados {len(parquet_files)} archivos parquet")

        # Leer y procesar cada archivo parquet
        speaker_counts = Counter()
        speaker_genders = defaultdict(set)
        speaker_ages = defaultdict(set)
        total_samples = 0

        print("\nProcesando archivos...")
        for pq_file in tqdm(parquet_files, desc="Archivos"):
            # Leer archivo parquet directamente
            with fs.open(pq_file, 'rb') as f:
                # Leer solo las columnas que necesitamos (sin audio)
                df = pd.read_parquet(
                    f,
                    columns=['client_id', 'gender', 'age', 'sentence']
                )

                total_samples += len(df)

                # Contar speakers
                for _, row in df.iterrows():
                    speaker_id = row['client_id'] if pd.notna(row['client_id']) else 'unknown'
                    speaker_counts[speaker_id] += 1

                    if pd.notna(row.get('gender')):
                        speaker_genders[speaker_id].add(row['gender'])
                    if pd.notna(row.get('age')):
                        speaker_ages[speaker_id].add(row['age'])

        print(f"\n✅ Procesadas {total_samples} muestras totales")

    except ImportError:
        print("⚠️  huggingface_hub no está instalado")
        print("   Instalando: pip install huggingface_hub")
        import subprocess
        subprocess.run(['pip', 'install', '-q', 'huggingface_hub[hf_transfer]'])
        return analyze_dataset_from_parquet(dataset_name, args)

    except Exception as e:
        print(f"❌ Error al leer archivos parquet: {e}")
        print("   Intentando método alternativo...")
        return analyze_dataset_limited(dataset_name, args)

    # Calcular estadísticas
    return compute_statistics(
        dataset_name,
        speaker_counts,
        speaker_genders,
        speaker_ages,
        total_samples,
        args
    )


def analyze_dataset_limited(dataset_name, args):
    """
    Método alternativo usando datasets pero con límite de muestras.
    """
    from datasets import load_dataset

    print("Cargando subset del dataset (primeras 100k muestras)...")

    try:
        # Cargar solo un subset
        ds = load_dataset(dataset_name, split='train', streaming=True)

        speaker_counts = Counter()
        speaker_genders = defaultdict(set)
        speaker_ages = defaultdict(set)
        total_samples = 0

        # Procesar solo primeras 100k muestras
        MAX_SAMPLES = 100000

        for i, example in enumerate(ds):
            if i >= MAX_SAMPLES:
                break

            total_samples += 1
            speaker_id = example.get('client_id', 'unknown')
            speaker_counts[speaker_id] += 1

            if example.get('gender'):
                speaker_genders[speaker_id].add(example['gender'])
            if example.get('age'):
                speaker_ages[speaker_id].add(example['age'])

            if (i + 1) % 10000 == 0:
                print(f"  Procesadas {i + 1} muestras...", end='\r')

        print(f"\n⚠️  Análisis limitado a {total_samples} muestras (subset)")

        return compute_statistics(
            dataset_name,
            speaker_counts,
            speaker_genders,
            speaker_ages,
            total_samples,
            args
        )

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def compute_statistics(dataset_name, speaker_counts, speaker_genders, speaker_ages, total_samples, args):
    """Calcula estadísticas del análisis."""

    total_speakers = len(speaker_counts)
    variant = get_variant_from_dataset_name(dataset_name)

    # Top speakers
    top_speakers = speaker_counts.most_common(1000)

    # Estadísticas
    counts_values = speaker_counts.values()
    min_samples = min(counts_values)
    max_samples = max(counts_values)
    total_count = sum(counts_values)
    mean_samples = total_count / total_speakers

    sorted_samples = sorted(counts_values)
    median_samples = sorted_samples[len(sorted_samples)//2]

    # Distribución
    distribution_ranges = {
        '1-9': sum(1 for c in counts_values if c < 10),
        '10-49': sum(1 for c in counts_values if 10 <= c < 50),
        '50-99': sum(1 for c in counts_values if 50 <= c < 100),
        '100-199': sum(1 for c in counts_values if 100 <= c < 200),
        '200+': sum(1 for c in counts_values if c >= 200)
    }

    # Análisis de estrategias
    speakers_for_fixed = [
        (speaker, count) for speaker, count in speaker_counts.items()
        if count >= args.min_samples_fixed
    ]

    speakers_for_multispeaker = [
        (speaker, count) for speaker, count in speaker_counts.items()
        if count >= args.min_samples_multispeaker
    ]

    stats = {
        'dataset_name': dataset_name,
        'variant': variant,
        'total_samples': total_samples,
        'total_speakers': total_speakers,
        'samples_per_speaker': {
            'min': min_samples,
            'max': max_samples,
            'mean': mean_samples,
            'median': median_samples
        },
        'distribution_ranges': distribution_ranges,
        'speaker_counts': dict(top_speakers),
        'strategy_analysis': {
            'fixed_voices': {
                'count': len(speakers_for_fixed),
                'speakers': sorted(speakers_for_fixed, key=lambda x: x[1], reverse=True),
                'total_samples': sum(count for _, count in speakers_for_fixed)
            },
            'multispeaker': {
                'count': len(speakers_for_multispeaker),
                'total_samples': sum(count for _, count in speakers_for_multispeaker)
            }
        }
    }

    # Imprimir resumen
    print(f"\n📊 ESTADÍSTICAS:")
    print(f"  Total hablantes: {total_speakers}")
    print(f"  Muestras por hablante (promedio): {mean_samples:.1f}")
    print(f"  Muestras por hablante (mediana): {median_samples}")
    print(f"  Rango: {min_samples} - {max_samples}")

    print(f"\n🎯 ANÁLISIS DE ESTRATEGIAS:")
    print(f"  Voces fijas candidatas (>={args.min_samples_fixed} muestras):")
    print(f"    - {len(speakers_for_fixed)} hablantes")
    print(f"    - {stats['strategy_analysis']['fixed_voices']['total_samples']} muestras totales")

    if speakers_for_fixed:
        print(f"\n  Top 5 candidatos para voces fijas:")
        for i, (speaker, count) in enumerate(speakers_for_fixed[:5], 1):
            gender = list(speaker_genders[speaker])[0] if speaker_genders[speaker] else 'unknown'
            print(f"    {i}. {speaker[:10]}... : {count} muestras, {gender}")
    else:
        print(f"    ⚠️  No hay hablantes con suficientes muestras para voces fijas")

    print(f"\n  Multi-speaker candidatos (>={args.min_samples_multispeaker} muestras):")
    print(f"    - {len(speakers_for_multispeaker)} hablantes")
    print(f"    - {stats['strategy_analysis']['multispeaker']['total_samples']} muestras totales")

    print(f"\n  Distribución de hablantes:")
    for range_name in ['1-9', '10-49', '50-99', '100-199', '200+']:
        count = distribution_ranges[range_name]
        pct = count / total_speakers * 100
        print(f"    {range_name} muestras: {count} hablantes ({pct:.1f}%)")

    return stats


def main():
    args = parse_args()

    print("="*70)
    print("ANÁLISIS DE DISTRIBUCIÓN DE HABLANTES (METADATA ONLY)")
    print("="*70)

    all_stats = []

    for dataset_name in args.datasets:
        stats = analyze_dataset_from_parquet(dataset_name, args)
        if stats:
            all_stats.append(stats)

    if not all_stats:
        print("\n❌ No se pudo analizar ningún dataset")
        return

    # Guardar estadísticas
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = output_dir / 'speaker_statistics.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Estadísticas guardadas: {stats_path}")

    print("\n" + "="*70)
    print("ANÁLISIS COMPLETADO")
    print("="*70)


if __name__ == '__main__':
    import sys
    import traceback

    try:
        main()
    except Exception as e:
        print("\n" + "="*70)
        print("❌ ERROR DURANTE LA EJECUCIÓN")
        print("="*70)
        print(f"\nTipo de error: {type(e).__name__}")
        print(f"Mensaje: {str(e)}")
        print("\nTraceback completo:")
        traceback.print_exc()
        sys.exit(1)
