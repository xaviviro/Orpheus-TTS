"""
Script para tokenizar datasets de audio con SNAC usando múltiples procesos GPU.

Basado en el pipeline de KaniTTS pero adaptado para Orpheus TTS.
Usa multiprocessing para cargar múltiples instancias de SNAC y procesar en paralelo.

Requiere: snac, torch
"""

import argparse
import torch
import torch.multiprocessing as mp
from datasets import load_from_disk, Dataset
from pathlib import Path
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description='Tokenizar audio con SNAC usando multiprocessing GPU'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directorio con dataset procesado (output de prepare_by_dialect.py)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directorio de salida para dataset tokenizado'
    )
    parser.add_argument(
        '--num_gpu_workers',
        type=int,
        default=None,
        help='Número de workers GPU (default: detectar GPUs disponibles)'
    )
    parser.add_argument(
        '--snac_model',
        type=str,
        default='hubertsiuzdak/snac_24khz',
        help='Modelo SNAC a usar'
    )
    parser.add_argument(
        '--queue_size',
        type=int,
        default=100,
        help='Tamaño de la queue de comunicación'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size por worker GPU'
    )

    return parser.parse_args()


def snac_worker(gpu_id, input_queue, output_queue, snac_model, batch_size):
    """
    Worker GPU que procesa audio con SNAC.

    Args:
        gpu_id: ID de la GPU a usar
        input_queue: Queue de entrada con audio samples
        output_queue: Queue de salida con tokens SNAC
        snac_model: Nombre del modelo SNAC
        batch_size: Tamaño de batch para procesar
    """
    import snac

    # Configurar GPU
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(gpu_id)

    print(f"🔥 GPU Worker {gpu_id}: Cargando modelo SNAC en {device}...")

    # Cargar modelo SNAC
    try:
        model = snac.SNAC.from_pretrained(snac_model).eval().to(device)
        print(f"✅ GPU Worker {gpu_id}: Modelo cargado")
    except Exception as e:
        print(f"❌ GPU Worker {gpu_id}: Error cargando modelo: {e}")
        return

    processed = 0

    with torch.no_grad():
        while True:
            # Obtener item de la queue
            item = input_queue.get()

            # Sentinel para terminar
            if item is None:
                break

            try:
                idx, audio_array, sample_rate = item

                # Convertir a tensor
                audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0).to(device)

                # Encode con SNAC
                codes = model.encode(audio_tensor)

                # codes es una lista de tensores [batch, time] para cada nivel
                # Convertir a numpy y enviar a output queue
                codes_np = []
                for level in codes:
                    codes_np.append(level.cpu().numpy())

                # Enviar resultado
                output_queue.put((idx, codes_np))

                processed += 1

                if processed % 100 == 0:
                    print(f"  GPU {gpu_id}: {processed} samples procesados", end='\r')

            except Exception as e:
                print(f"❌ GPU Worker {gpu_id}: Error procesando sample {idx}: {e}")
                output_queue.put((idx, None))

    print(f"✅ GPU Worker {gpu_id}: Completado - {processed} samples procesados")


def process_dataset_with_snac(dataset, args):
    """
    Procesa un dataset con SNAC usando múltiples GPU workers.
    """
    print("\n" + "="*70)
    print("TOKENIZACIÓN CON SNAC (Multi-GPU)")
    print("="*70)

    # Detectar GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("❌ No se detectaron GPUs. SNAC requiere GPU.")

    num_workers = args.num_gpu_workers if args.num_gpu_workers else num_gpus
    num_workers = min(num_workers, num_gpus)

    print(f"🔥 GPUs detectadas: {num_gpus}")
    print(f"🚀 Usando {num_workers} GPU workers")
    print(f"📦 Queue size: {args.queue_size}")
    print(f"📊 Batch size: {args.batch_size}")

    # Crear queues
    mp.set_start_method('spawn', force=True)
    input_queue = mp.Queue(maxsize=args.queue_size)
    output_queue = mp.Queue()

    # Iniciar workers
    workers = []
    for gpu_id in range(num_workers):
        p = mp.Process(
            target=snac_worker,
            args=(gpu_id, input_queue, output_queue, args.snac_model, args.batch_size)
        )
        p.start()
        workers.append(p)

    # Enviar datos a los workers
    print(f"\n📤 Enviando {len(dataset)} samples a los workers...")

    for idx, example in enumerate(tqdm(dataset, desc="Enviando samples")):
        audio_array = example['audio']['array']
        sample_rate = example['audio']['sampling_rate']
        input_queue.put((idx, audio_array, sample_rate))

    # Enviar sentinel a todos los workers
    for _ in range(num_workers):
        input_queue.put(None)

    print("✅ Todos los samples enviados")

    # Recolectar resultados
    print(f"\n📥 Recolectando resultados...")
    results = {}

    for _ in tqdm(range(len(dataset)), desc="Recolectando tokens"):
        idx, codes = output_queue.get()
        if codes is not None:
            results[idx] = codes

    # Esperar a que terminen los workers
    for p in workers:
        p.join()

    print(f"✅ Tokens recolectados: {len(results)}/{len(dataset)} samples")

    # Crear nuevo dataset con tokens
    print("\n📦 Creando dataset tokenizado...")

    tokenized_examples = []
    for idx in range(len(dataset)):
        if idx not in results:
            continue

        example = dataset[idx]
        codes = results[idx]

        # Extraer tokens de cada nivel (SNAC tiene 3 niveles)
        tokenized_example = {
            'text': example['text'],
            'original_text': example['original_text'],
            'voice_name': example['voice_name'],
            'dialect': example['dialect'],
            'speaker_id': example['speaker_id'],
            'gender': example['gender'],
            'age': example['age'],
            'duration': example['duration'],
            # SNAC tokens por nivel
            'snac_codes_0': codes[0][0].tolist(),  # Nivel 0 (más grueso)
            'snac_codes_1': codes[1][0].tolist(),  # Nivel 1
            'snac_codes_2': codes[2][0].tolist(),  # Nivel 2 (más fino)
        }

        tokenized_examples.append(tokenized_example)

    tokenized_dataset = Dataset.from_list(tokenized_examples)

    return tokenized_dataset


def main():
    args = parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("TOKENIZACIÓN DE DATASETS CON SNAC")
    print("="*70)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Modelo SNAC: {args.snac_model}")

    # Buscar subdirectorios (dialectos)
    dialect_dirs = [d for d in input_path.iterdir() if d.is_dir()]

    if not dialect_dirs:
        # No hay subdirectorios, cargar directamente
        print("\n📦 Cargando dataset único...")
        dataset = load_from_disk(str(input_path))

        # Procesar
        tokenized = process_dataset_with_snac(dataset, args)

        # Guardar
        print(f"\n💾 Guardando en {output_path}")
        tokenized.save_to_disk(str(output_path))

    else:
        # Procesar cada dialecto
        print(f"\n📂 Encontrados {len(dialect_dirs)} dialectos")

        stats = {}

        for dialect_dir in dialect_dirs:
            dialect_name = dialect_dir.name
            print(f"\n{'='*70}")
            print(f"Procesando dialecto: {dialect_name}")
            print(f"{'='*70}")

            # Cargar dataset del dialecto
            dataset = load_from_disk(str(dialect_dir))
            print(f"📊 {len(dataset)} samples en {dialect_name}")

            # Procesar con SNAC
            tokenized = process_dataset_with_snac(dataset, args)

            # Guardar
            output_dialect_dir = output_path / dialect_name
            print(f"\n💾 Guardando en {output_dialect_dir}")
            tokenized.save_to_disk(str(output_dialect_dir))

            # Estadísticas
            stats[dialect_name] = {
                'samples': len(tokenized),
                'avg_tokens_level_0': np.mean([len(ex['snac_codes_0']) for ex in tokenized]),
                'avg_tokens_level_1': np.mean([len(ex['snac_codes_1']) for ex in tokenized]),
                'avg_tokens_level_2': np.mean([len(ex['snac_codes_2']) for ex in tokenized]),
            }

        # Guardar estadísticas
        stats_path = output_path / 'tokenization_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print("\n" + "="*70)
        print("ESTADÍSTICAS DE TOKENIZACIÓN")
        print("="*70)

        for dialect, stat in stats.items():
            print(f"\n{dialect}:")
            print(f"  Samples: {stat['samples']}")
            print(f"  Tokens promedio (nivel 0): {stat['avg_tokens_level_0']:.1f}")
            print(f"  Tokens promedio (nivel 1): {stat['avg_tokens_level_1']:.1f}")
            print(f"  Tokens promedio (nivel 2): {stat['avg_tokens_level_2']:.1f}")

    print("\n" + "="*70)
    print("✅ TOKENIZACIÓN COMPLETADA")
    print("="*70)


if __name__ == '__main__':
    main()
