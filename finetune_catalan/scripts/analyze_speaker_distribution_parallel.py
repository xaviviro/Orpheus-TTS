"""
Script para analizar la distribución de hablantes con procesamiento paralelo.

Usa multiprocesamiento para aprovechar todos los cores de la CPU y acelerar el análisis
de datasets muy grandes (1M+ muestras).
"""

import argparse
from datasets import load_dataset
from collections import Counter, defaultdict
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analizar distribución de hablantes (multiproceso)'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        required=True,
        help='Lista de datasets a analizar'
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
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Número de workers (default: número de CPUs)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help='Tamaño de batch para procesamiento paralelo'
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


def process_batch(batch):
    """
    Procesa un batch de ejemplos y retorna contadores locales.

    Esta función se ejecuta en paralelo en múltiples procesos.
    Solo procesa client_id para máxima velocidad.
    """
    local_speaker_counts = Counter()

    for example in batch:
        speaker_id = example.get('client_id', 'unknown')
        local_speaker_counts[speaker_id] += 1

    return local_speaker_counts


def merge_results(results):
    """Combina resultados de múltiples procesos."""
    combined_counts = Counter()

    for counts in results:
        combined_counts.update(counts)

    return combined_counts


def analyze_dataset_parallel(dataset_name, args):
    """Analiza un dataset usando procesamiento paralelo."""
    print(f"\n{'='*70}")
    print(f"Analizando: {dataset_name}")
    print(f"{'='*70}")

    # Determinar número de workers
    num_workers = args.num_workers or cpu_count()
    print(f"\nUsando {num_workers} workers para procesamiento paralelo")

    # Cargar dataset en modo streaming - SOLO client_id para máxima velocidad
    print("Cargando dataset en modo streaming (solo client_id)...")
    try:
        ds = load_dataset(dataset_name, split='train', streaming=True)

        # Obtener nombres de todas las columnas
        column_names = ds.column_names

        # Remover TODAS las columnas excepto client_id
        # Esto es MUCHO más rápido que cargar todo el dataset
        columns_to_remove = [col for col in column_names if col != 'client_id']
        ds = ds.remove_columns(columns_to_remove)

        print(f"✓ Solo cargando columna 'client_id' ({len(columns_to_remove)} columnas omitidas)")

    except Exception as e:
        print(f"Error cargando dataset: {e}")
        return None

    # Recolectar ejemplos en batches
    print(f"\nProcesando en batches de {args.batch_size} muestras...")

    batches = []
    current_batch = []
    total_samples = 0

    # Primero, recolectamos todos los ejemplos en batches
    # (Nota: para datasets MUY grandes, esto podría optimizarse más)
    print("Leyendo dataset...")
    for i, example in enumerate(tqdm(ds, desc="Cargando datos")):
        current_batch.append(example)
        total_samples += 1

        if len(current_batch) >= args.batch_size:
            batches.append(current_batch)
            current_batch = []

        # Para debugging, limitar a 100k samples si quieres
        # if total_samples >= 100000:
        #     break

    # Agregar último batch si quedó algo
    if current_batch:
        batches.append(current_batch)

    print(f"\n✓ Leídas {total_samples} muestras en {len(batches)} batches")

    # Procesar batches en paralelo
    print(f"\nProcesando {len(batches)} batches en paralelo con {num_workers} workers...")

    start_time = time.time()

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_batch, batches),
            total=len(batches),
            desc="Procesando batches"
        ))

    elapsed = time.time() - start_time
    print(f"\n✓ Procesamiento completado en {elapsed:.1f} segundos")
    print(f"  Velocidad: {total_samples/elapsed:.0f} muestras/segundo")

    # Combinar resultados
    print("\nCombinando resultados de todos los workers...")
    speaker_counts = merge_results(results)

    print(f"✓ Procesadas {total_samples} muestras totales")
    print(f"  {len(speaker_counts)} hablantes únicos encontrados")

    # Calcular estadísticas
    return compute_statistics(
        dataset_name,
        speaker_counts,
        total_samples,
        args
    )


def compute_statistics(dataset_name, speaker_counts, total_samples, args):
    """
    Calcula estadísticas del análisis.

    Solo usa speaker_counts para máxima velocidad (sin gender/age).
    """

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
            print(f"    {i}. {speaker[:10]}... : {count} muestras")
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


def create_visualizations(all_stats, output_dir):
    """Crea visualizaciones del análisis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use('Agg')

        for stats in all_stats:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f"Análisis: {stats['variant']}", fontsize=16)

            samples = list(stats['speaker_counts'].values())

            # Histograma
            axes[0, 0].hist(samples, bins=50, edgecolor='black')
            axes[0, 0].set_xlabel('Muestras por hablante')
            axes[0, 0].set_ylabel('Número de hablantes')
            axes[0, 0].set_title('Distribución de muestras')
            axes[0, 0].axvline(100, color='r', linestyle='--', label='Umbral voces fijas')
            axes[0, 0].legend()

            # Top hablantes
            top_20 = sorted(stats['speaker_counts'].items(), key=lambda x: x[1], reverse=True)[:20]
            speakers, counts = zip(*top_20)
            speakers_short = [s[:10] + '...' for s in speakers]
            axes[0, 1].barh(range(len(speakers_short)), counts)
            axes[0, 1].set_yticks(range(len(speakers_short)))
            axes[0, 1].set_yticklabels(speakers_short, fontsize=8)
            axes[0, 1].set_xlabel('Muestras')
            axes[0, 1].set_title('Top 20 hablantes')
            axes[0, 1].invert_yaxis()

            # Distribución acumulativa
            sorted_samples = sorted(samples, reverse=True)[:1000]
            cumsum = [sum(sorted_samples[:i+1]) for i in range(len(sorted_samples))]
            axes[1, 0].plot(cumsum)
            axes[1, 0].set_xlabel('Número de hablantes (top 1000)')
            axes[1, 0].set_ylabel('Muestras acumuladas')
            axes[1, 0].set_title('Muestras acumuladas por top hablantes')
            axes[1, 0].grid(True)

            # Resumen de estadísticas
            axes[1, 1].axis('off')
            summary_text = f"""
Resumen Estadístico:

Total hablantes: {stats['total_speakers']}
Total muestras: {stats['total_samples']}

Muestras por hablante:
  • Promedio: {stats['samples_per_speaker']['mean']:.1f}
  • Mediana: {stats['samples_per_speaker']['median']}
  • Min-Max: {stats['samples_per_speaker']['min']}-{stats['samples_per_speaker']['max']}

Candidatos para voces fijas: {stats['strategy_analysis']['fixed_voices']['count']}
Candidatos para multi-speaker: {stats['strategy_analysis']['multispeaker']['count']}
            """
            axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, family='monospace')

            plt.tight_layout()
            output_path = output_dir / f"analysis_{stats['variant']}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Visualización guardada: {output_path}")

    except ImportError:
        print("matplotlib no disponible, saltando visualizaciones")


def generate_report(all_stats, args, output_dir):
    """Genera reporte en markdown."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / 'speaker_analysis_report.md'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Análisis de Distribución de Hablantes (Procesamiento Paralelo)\n\n")
        f.write(f"Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        # Resumen general
        f.write("## Resumen General\n\n")
        total_samples = sum(s['total_samples'] for s in all_stats)
        total_speakers = sum(s['total_speakers'] for s in all_stats)
        f.write(f"- **Total de datasets analizados**: {len(all_stats)}\n")
        f.write(f"- **Total de muestras**: {total_samples:,}\n")
        f.write(f"- **Total de hablantes**: {total_speakers:,}\n\n")

        # Por dataset
        f.write("## Análisis por Dataset\n\n")
        for stats in all_stats:
            f.write(f"### {stats['variant'].capitalize()}\n\n")
            f.write(f"**Dataset**: `{stats['dataset_name']}`\n\n")

            f.write("#### Estadísticas Básicas\n\n")
            f.write(f"- Total hablantes: {stats['total_speakers']}\n")
            f.write(f"- Total muestras: {stats['total_samples']}\n")
            f.write(f"- Promedio muestras/hablante: {stats['samples_per_speaker']['mean']:.1f}\n")
            f.write(f"- Mediana muestras/hablante: {stats['samples_per_speaker']['median']}\n\n")

            f.write("#### Recomendación de Estrategia\n\n")
            fixed_count = stats['strategy_analysis']['fixed_voices']['count']
            multi_count = stats['strategy_analysis']['multispeaker']['count']

            if fixed_count >= 3:
                f.write(f"✅ **Estrategia Híbrida recomendada**\n")
                f.write(f"- {fixed_count} hablantes con suficientes datos para voces fijas\n")
                f.write(f"- {multi_count} hablantes para entrenamiento multi-speaker\n\n")
            elif multi_count >= 50:
                f.write(f"✅ **Estrategia Multi-Speaker recomendada**\n")
                f.write(f"- {multi_count} hablantes con datos suficientes\n")
                f.write(f"- Ideal para zero-shot voice cloning\n\n")
            else:
                f.write(f"⚠️  **Dataset pequeño**\n")
                f.write(f"- Considerar usar modelo preentrenado con zero-shot\n\n")

    print(f"\n✅ Reporte generado: {report_path}")


def main():
    args = parse_args()

    print("="*70)
    print("ANÁLISIS DE DISTRIBUCIÓN DE HABLANTES (MULTIPROCESO)")
    print("="*70)
    print(f"\nCPUs disponibles: {cpu_count()}")
    print(f"Workers a usar: {args.num_workers or cpu_count()}")
    print(f"Batch size: {args.batch_size}")

    all_stats = []

    for dataset_name in args.datasets:
        stats = analyze_dataset_parallel(dataset_name, args)
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

    # Crear visualizaciones
    create_visualizations(all_stats, output_dir)

    # Generar reporte
    generate_report(all_stats, args, output_dir)

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
