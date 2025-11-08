"""
Script simple para subir el dataset tokenizado a HuggingFace Hub.
"""

import argparse
from datasets import load_from_disk
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Subir dataset tokenizado a HuggingFace Hub')
    parser.add_argument(
        '--dataset_dir',
        type=str,
        required=True,
        help='Directorio con el dataset tokenizado (ej: /workspace/data/tokenized_simple)'
    )
    parser.add_argument(
        '--hf_repo',
        type=str,
        required=True,
        help='Nombre del repositorio en HuggingFace (ej: username/catalan-tts-tokenized)'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Hacer el dataset privado'
    )
    parser.add_argument(
        '--commit_message',
        type=str,
        default='Upload tokenized Catalan TTS dataset',
        help='Mensaje del commit'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("="*60)
    print("SUBIR DATASET TOKENIZADO A HUGGINGFACE HUB")
    print("="*60)

    # Verificar que el directorio existe
    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        print(f"❌ Error: El directorio {args.dataset_dir} no existe")
        return

    # Cargar dataset
    print(f"\n📂 Cargando dataset desde: {args.dataset_dir}")
    dataset = load_from_disk(args.dataset_dir)

    print(f"   Train: {len(dataset['train'])} ejemplos")
    if 'validation' in dataset:
        print(f"   Validation: {len(dataset['validation'])} ejemplos")

    # Subir a HuggingFace
    print(f"\n☁️  Subiendo a HuggingFace: {args.hf_repo}")
    print(f"   Privado: {'Sí' if args.private else 'No'}")

    try:
        dataset.push_to_hub(
            args.hf_repo,
            private=args.private,
            commit_message=args.commit_message
        )
        print(f"\n✅ Dataset subido exitosamente")
        print(f"   URL: https://huggingface.co/datasets/{args.hf_repo}")
    except Exception as e:
        print(f"\n❌ Error subiendo a HuggingFace: {e}")
        print("\nAsegúrate de:")
        print("  1. Estar autenticado: huggingface-cli login")
        print("  2. Tener permisos de escritura en el repositorio")
        return

    print("\n✅ ¡Proceso completado!")


if __name__ == '__main__':
    main()
