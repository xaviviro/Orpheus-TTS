"""
Script para analizar la distribución de hablantes en datasets de Common Voice.

Ayuda a decidir la mejor estrategia: voces fijas, multi-speaker, o híbrido.

NOTA: Este script usa streaming mode para manejar datasets muy grandes (1M+ muestras)
sin consumir toda la RAM. El análisis puede tardar unos minutos pero es memory-efficient.
"""

import argparse
from datasets import load_dataset
from collections import Counter, defaultdict
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analizar distribución de hablantes en datasets de catalán'
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


def analyze_dataset(dataset_name, args):
    """Analiza un dataset individual."""
    print(f"\n{'='*70}")
    print(f"Analizando: {dataset_name}")
    print(f"{'='*70}")

    # Cargar dataset en modo streaming para no consumir toda la RAM
    print("Cargando dataset en modo streaming...")
    try:
        ds = load_dataset(dataset_name, split='train', streaming=True)
    except Exception as e:
        print(f"Error cargando dataset: {e}")
        return None

    # Análisis por hablante - procesar en streaming
    print("\nAnalizando hablantes (esto puede tomar unos minutos)...")
    speaker_counts = Counter()
    speaker_durations = defaultdict(float)
    speaker_genders = defaultdict(set)
    speaker_ages = defaultdict(set)

    total_samples = 0

    # Procesar en streaming para no cargar todo en memoria
    for i, example in enumerate(ds):
        total_samples += 1

        speaker_id = example.get('client_id', 'unknown')
        speaker_counts[speaker_id] += 1

        # Duración (aproximada) - solo calculamos de una muestra cada 10 para ahorrar memoria
        if i % 10 == 0 and 'audio' in example and example['audio']:
            try:
                duration = len(example['audio']['array']) / example['audio']['sampling_rate']
                # Estimamos duración multiplicando por 10 (ya que solo procesamos 1 de cada 10)
                speaker_durations[speaker_id] += duration * 10
            except:
                pass  # Saltamos si hay error en el audio

        # Metadata
        if 'gender' in example and example['gender']:
            speaker_genders[speaker_id].add(example['gender'])
        if 'age' in example and example['age']:
            speaker_ages[speaker_id].add(example['age'])

        # Mostrar progreso cada 10000 muestras
        if (i + 1) % 10000 == 0:
            print(f"  Procesadas {i + 1} muestras, {len(speaker_counts)} hablantes únicos...", end='\r')

    print(f"\n  ✅ Procesadas {total_samples} muestras totales")
    print(f"Total de muestras: {total_samples}")

    # Estadísticas
    total_speakers = len(speaker_counts)
    samples_per_speaker = list(speaker_counts.values())

    variant = get_variant_from_dataset_name(dataset_name)

    stats = {
        'dataset_name': dataset_name,
        'variant': variant,
        'total_samples': total_samples,
        'total_speakers': total_speakers,
        'samples_per_speaker': {
            'min': min(samples_per_speaker),
            'max': max(samples_per_speaker),
            'mean': sum(samples_per_speaker) / len(samples_per_speaker),
            'median': sorted(samples_per_speaker)[len(samples_per_speaker)//2]
        },
        'speaker_counts': dict(speaker_counts.most_common()),
        'speaker_durations': dict(speaker_durations)
    }

    # Análisis para estrategias
    speakers_for_fixed = [
        (speaker, count) for speaker, count in speaker_counts.items()
        if count >= args.min_samples_fixed
    ]

    speakers_for_multispeaker = [
        (speaker, count) for speaker, count in speaker_counts.items()
        if count >= args.min_samples_multispeaker
    ]

    stats['strategy_analysis'] = {
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

    # Imprimir resumen
    print(f"\n📊 ESTADÍSTICAS:")
    print(f"  Total hablantes: {total_speakers}")
    print(f"  Muestras por hablante (promedio): {stats['samples_per_speaker']['mean']:.1f}")
    print(f"  Muestras por hablante (mediana): {stats['samples_per_speaker']['median']}")
    print(f"  Rango: {stats['samples_per_speaker']['min']} - {stats['samples_per_speaker']['max']}")

    print(f"\n🎯 ANÁLISIS DE ESTRATEGIAS:")
    print(f"  Voces fijas candidatas (>={args.min_samples_fixed} muestras):")
    print(f"    - {len(speakers_for_fixed)} hablantes")
    print(f"    - {stats['strategy_analysis']['fixed_voices']['total_samples']} muestras totales")

    if speakers_for_fixed:
        print(f"\n  Top 5 candidatos para voces fijas:")
        for i, (speaker, count) in enumerate(speakers_for_fixed[:5], 1):
            duration = speaker_durations[speaker] / 60  # minutos
            gender = list(speaker_genders[speaker])[0] if speaker_genders[speaker] else 'unknown'
            print(f"    {i}. {speaker[:10]}... : {count} muestras, {duration:.1f} min, {gender}")
    else:
        print(f"    ⚠️  No hay hablantes con suficientes muestras para voces fijas")

    print(f"\n  Multi-speaker candidatos (>={args.min_samples_multispeaker} muestras):")
    print(f"    - {len(speakers_for_multispeaker)} hablantes")
    print(f"    - {stats['strategy_analysis']['multispeaker']['total_samples']} muestras totales")

    # Distribución
    distribution = Counter()
    for count in samples_per_speaker:
        if count < 10:
            distribution['1-9'] += 1
        elif count < 50:
            distribution['10-49'] += 1
        elif count < 100:
            distribution['50-99'] += 1
        elif count < 200:
            distribution['100-199'] += 1
        else:
            distribution['200+'] += 1

    print(f"\n  Distribución de hablantes:")
    for range_name, count in sorted(distribution.items()):
        pct = count / total_speakers * 100
        print(f"    {range_name} muestras: {count} hablantes ({pct:.1f}%)")

    return stats


def create_visualizations(all_stats, output_dir):
    """Crea visualizaciones del análisis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use('Agg')  # Backend sin GUI

        # Gráfico de distribución de muestras por hablante
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
            sorted_samples = sorted(samples, reverse=True)
            cumsum = [sum(sorted_samples[:i+1]) for i in range(len(sorted_samples))]
            axes[1, 0].plot(cumsum)
            axes[1, 0].set_xlabel('Número de hablantes')
            axes[1, 0].set_ylabel('Muestras acumuladas')
            axes[1, 0].set_title('Muestras acumuladas por hablantes')
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
        f.write("# Análisis de Distribución de Hablantes\n\n")
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

            # Top hablantes
            if stats['strategy_analysis']['fixed_voices']['speakers']:
                f.write("#### Top Candidatos para Voces Fijas\n\n")
                f.write("| Hablante | Muestras | % del Total |\n")
                f.write("|----------|----------|-------------|\n")
                for speaker, count in stats['strategy_analysis']['fixed_voices']['speakers'][:10]:
                    pct = count / stats['total_samples'] * 100
                    f.write(f"| {speaker[:15]}... | {count} | {pct:.1f}% |\n")
                f.write("\n")

        # Recomendaciones finales
        f.write("## Recomendaciones Finales\n\n")

        total_fixed_speakers = sum(
            s['strategy_analysis']['fixed_voices']['count'] for s in all_stats
        )
        total_multi_speakers = sum(
            s['strategy_analysis']['multispeaker']['count'] for s in all_stats
        )

        f.write(f"### Estrategia Recomendada\n\n")

        if total_fixed_speakers >= 5:
            f.write("**Estrategia Híbrida**: Combinar voces fijas + multi-speaker\n\n")
            f.write("1. Seleccionar top 2-3 hablantes por dialecto para voces fijas\n")
            f.write("2. Usar el resto para entrenamiento multi-speaker\n")
            f.write("3. Resultado: Voces de alta calidad + flexibilidad zero-shot\n\n")
        else:
            f.write("**Estrategia Multi-Speaker**: Entrenamiento para zero-shot cloning\n\n")
            f.write("1. Usar todos los hablantes (min 5 muestras cada uno)\n")
            f.write("2. Entrenar con in-context learning\n")
            f.write("3. Resultado: Capacidad de clonar cualquier voz catalana\n\n")

        f.write("### Comandos Sugeridos\n\n")
        f.write("```bash\n")
        f.write("# Para voces fijas (si aplica)\n")
        f.write("python scripts/prepare_fixed_voices.py \\\n")
        f.write(f"    --min_samples_per_speaker {args.min_samples_fixed} \\\n")
        f.write("    --speakers_per_dialect 2\n\n")
        f.write("# Para multi-speaker\n")
        f.write("python scripts/prepare_multispeaker_catalan.py \\\n")
        f.write(f"    --min_samples_per_speaker {args.min_samples_multispeaker} \\\n")
        f.write("    --max_samples_per_speaker 50\n")
        f.write("```\n")

    print(f"\n✅ Reporte generado: {report_path}")


def main():
    args = parse_args()

    print("="*70)
    print("ANÁLISIS DE DISTRIBUCIÓN DE HABLANTES")
    print("="*70)

    all_stats = []

    for dataset_name in args.datasets:
        stats = analyze_dataset(dataset_name, args)
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
    main()
