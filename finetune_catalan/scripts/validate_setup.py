"""
Script de validación para verificar que todo está configurado correctamente
antes de empezar el entrenamiento.

Basado en las mejores prácticas oficiales de Canopy Labs.
"""

import argparse
from pathlib import Path
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Validar setup de entrenamiento')
    parser.add_argument(
        '--model_name',
        type=str,
        default='canopylabs/orpheus-tts-0.1-pretrained',
        help='Modelo base a usar'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default=None,
        help='Ruta al dataset tokenizado (opcional)'
    )
    return parser.parse_args()


def validate_tokenizer(model_name):
    """Valida que el tokenizer tenga vocabulario extendido para SNAC."""
    print("\n" + "="*70)
    print("VALIDANDO TOKENIZER")
    print("="*70)

    try:
        from transformers import AutoTokenizer

        print(f"Cargando tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        vocab_size = len(tokenizer)
        print(f"\n✓ Tokenizer cargado")
        print(f"  Tamaño de vocabulario: {vocab_size:,}")

        # Verificar que tiene tokens SNAC extendidos
        # SNAC añade: 7 × 4096 + 20 = 28,692 tokens adicionales
        # Llama base: 128,000 tokens
        # Total esperado: ~156,692 tokens

        if vocab_size > 128000:
            print(f"  ✅ PASS: Vocabulario extendido detectado ({vocab_size:,} tokens)")
            print(f"  → Incluye tokens SNAC para audio")
        else:
            print(f"  ❌ FAIL: Vocabulario base ({vocab_size:,} tokens)")
            print(f"  → NO incluye tokens SNAC extendidos")
            print(f"\n  SOLUCIÓN:")
            print(f"    Usa: canopylabs/orpheus-tts-0.1-pretrained")
            print(f"    NO uses modelos multilingües (tienen bug de tokenizer)")
            return False

        # Verificar tokens especiales
        print(f"\n  Tokens especiales:")
        print(f"    PAD token: {tokenizer.pad_token_id}")
        print(f"    EOS token: {tokenizer.eos_token_id}")
        print(f"    BOS token: {tokenizer.bos_token_id}")

        # Test de tokenización
        test_text = "central: Bon dia! Com estàs?"
        tokens = tokenizer(test_text, return_tensors='pt')
        print(f"\n  Test de tokenización:")
        print(f"    Input: '{test_text}'")
        print(f"    Tokens: {tokens['input_ids'].shape[1]}")
        print(f"    Primeros tokens: {tokens['input_ids'][0][:10].tolist()}")

        return True

    except Exception as e:
        print(f"\n❌ ERROR al cargar tokenizer: {e}")
        return False


def validate_dependencies():
    """Valida que todas las dependencias estén instaladas."""
    print("\n" + "="*70)
    print("VALIDANDO DEPENDENCIAS")
    print("="*70)

    dependencies = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'datasets': 'Datasets',
        'accelerate': 'Accelerate',
        'snac': 'SNAC (codec de audio)',
        'librosa': 'Librosa',
        'soundfile': 'SoundFile',
    }

    optional_dependencies = {
        'flash_attn': 'Flash Attention 2 (acelera 2x)',
        'wandb': 'Weights & Biases (logging)',
        'peft': 'PEFT (para LoRA)',
    }

    all_ok = True

    print("\nDependencias requeridas:")
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - NO instalado")
            print(f"     pip install {module}")
            all_ok = False

    print("\nDependencias opcionales:")
    for module, name in optional_dependencies.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ⚠️  {name} - No instalado (opcional)")

    # Verificar versión de PyTorch
    try:
        import torch
        print(f"\nPyTorch:")
        print(f"  Versión: {torch.__version__}")
        print(f"  CUDA disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except:
        pass

    return all_ok


def validate_dataset(dataset_path):
    """Valida el dataset procesado."""
    print("\n" + "="*70)
    print("VALIDANDO DATASET")
    print("="*70)

    if not dataset_path:
        print("⚠️  No se especificó ruta de dataset (opcional)")
        return True

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        print(f"❌ Dataset no encontrado: {dataset_path}")
        return False

    try:
        from datasets import load_from_disk
        from collections import Counter

        print(f"Cargando dataset: {dataset_path}")
        ds = load_from_disk(str(dataset_path))

        print(f"\n✓ Dataset cargado")
        print(f"  Splits: {list(ds.keys())}")
        print(f"  Train: {len(ds['train']):,} ejemplos")
        if 'validation' in ds:
            print(f"  Validation: {len(ds['validation']):,} ejemplos")

        # Validar ejemplo
        example = ds['train'][0]
        print(f"\n  Campos del ejemplo:")
        for key in example.keys():
            print(f"    - {key}")

        # Validar que tiene input_ids
        if 'input_ids' not in example:
            print(f"\n  ❌ FAIL: No tiene campo 'input_ids'")
            return False

        print(f"\n  ✓ Ejemplo de datos:")
        print(f"    Longitud input_ids: {len(example['input_ids']):,}")

        if 'labels' in example:
            print(f"    Longitud labels: {len(example['labels']):,}")

        # Validar rango de tokens
        max_token = max(example['input_ids'])
        min_token = min(example['input_ids'])
        print(f"    Rango de tokens: {min_token} - {max_token}")

        # Contar tokens de audio (>128263)
        audio_tokens = [t for t in example['input_ids'] if t > 128263]
        text_tokens = [t for t in example['input_ids'] if t <= 128263]
        print(f"    Tokens de texto: {len(text_tokens)}")
        print(f"    Tokens de audio: {len(audio_tokens)}")

        if len(audio_tokens) == 0:
            print(f"\n  ⚠️  WARNING: No se detectaron tokens de audio")
            print(f"     Verifica que el dataset esté tokenizado correctamente")

        # Validar distribución de dialectos (si disponible)
        if 'dialect' in example or 'variant' in example:
            dialect_key = 'dialect' if 'dialect' in example else 'variant'
            dialects = [ex[dialect_key] for ex in ds['train'] if dialect_key in ex]
            dialect_counts = Counter(dialects)

            print(f"\n  Distribución de dialectos:")
            for dialect, count in dialect_counts.most_common():
                pct = count / len(ds['train']) * 100
                print(f"    {dialect}: {count:,} ({pct:.1f}%)")

        # Validar hablantes (si disponible)
        if 'speaker_id' in example or 'client_id' in example:
            speaker_key = 'speaker_id' if 'speaker_id' in example else 'client_id'
            speakers = set(ex[speaker_key] for ex in ds['train'] if speaker_key in ex)
            print(f"\n  Hablantes únicos: {len(speakers):,}")

            if len(speakers) < 50:
                print(f"    ⚠️  WARNING: Pocos hablantes (<50)")
                print(f"       Recomendado: 100+ para pretraining dialectal")
            elif len(speakers) < 100:
                print(f"    ⚠️  OK pero podría mejorarse (100+ ideal)")
            else:
                print(f"    ✅ Excelente diversidad de hablantes")

        return True

    except Exception as e:
        print(f"\n❌ ERROR validando dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_gpu():
    """Valida GPU y VRAM disponible."""
    print("\n" + "="*70)
    print("VALIDANDO GPU")
    print("="*70)

    try:
        import torch

        if not torch.cuda.is_available():
            print("❌ CUDA no disponible")
            print("   Entrenamiento será MUY lento en CPU")
            return False

        print(f"✓ CUDA disponible")
        print(f"  Versión CUDA: {torch.version.cuda}")
        print(f"  GPUs detectadas: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024**3)

            print(f"\n  GPU {i}: {props.name}")
            print(f"    VRAM total: {vram_gb:.1f} GB")
            print(f"    Compute capability: {props.major}.{props.minor}")

            # Recomendaciones
            if vram_gb < 16:
                print(f"    ⚠️  VRAM baja - Usa batch_size=1, gradient_checkpointing")
            elif vram_gb < 24:
                print(f"    ✓ VRAM OK - batch_size=1-2 recomendado")
            else:
                print(f"    ✅ VRAM excelente - batch_size=2-4 posible")

        return True

    except Exception as e:
        print(f"Error validando GPU: {e}")
        return False


def main():
    args = parse_args()

    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "VALIDACIÓN DE SETUP DE ENTRENAMIENTO" + " "*15 + "║")
    print("╚" + "="*68 + "╝")

    results = {
        'Dependencias': validate_dependencies(),
        'Tokenizer': validate_tokenizer(args.model_name),
        'GPU': validate_gpu(),
    }

    if args.dataset_path:
        results['Dataset'] = validate_dataset(args.dataset_path)

    # Resumen
    print("\n" + "="*70)
    print("RESUMEN DE VALIDACIÓN")
    print("="*70)

    all_passed = True
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check:20} {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*70)

    if all_passed:
        print("✅ TODAS LAS VALIDACIONES PASARON")
        print("\n¡Listo para entrenar!")
        print("\nSiguiente paso:")
        print("  accelerate launch scripts/train_catalan.py --config configs/config_catalan.yaml")
    else:
        print("❌ ALGUNAS VALIDACIONES FALLARON")
        print("\nRevisa los errores arriba y corrígelos antes de entrenar")
        sys.exit(1)


if __name__ == '__main__':
    main()
