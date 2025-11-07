"""
Script de inferencia para Orpheus TTS entrenado con dialectos catalanes.

Permite generar audio especificando el dialecto deseado.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import soundfile as sf
from pathlib import Path
import json


# Mapeo de dialectos a nombres de voz usados en entrenamiento
DIALECT_VOICES = {
    'central': 'central',
    'balearic': 'balear',
    'balear': 'balear',  # Alias
    'valencian': 'valencia',
    'valencia': 'valencia',  # Alias
    'northern': 'nord',
    'nord': 'nord',  # Alias
    'northwestern': 'occidental',
    'occidental': 'occidental',  # Alias
}


def parse_args():
    parser = argparse.ArgumentParser(description='Inferencia con dialectos catalanes')
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Ruta al modelo entrenado'
    )
    parser.add_argument(
        '--text',
        type=str,
        required=True,
        help='Texto a sintetizar'
    )
    parser.add_argument(
        '--dialect',
        type=str,
        required=True,
        choices=['central', 'balearic', 'balear', 'valencian', 'valencia', 'northern', 'nord', 'northwestern', 'occidental'],
        help='Dialecto a usar'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output.wav',
        help='Ruta del archivo de salida'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=2048,
        help='Longitud máxima de generación'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Temperature para generación'
    )
    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=1.1,
        help='Penalización por repetición (debe ser >= 1.1)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Dispositivo (cuda/cpu)'
    )

    return parser.parse_args()


def format_prompt(text, dialect):
    """
    Formatea el texto con el prefijo de dialecto.

    Ejemplos:
        format_prompt("Bon dia", "central") -> "central: Bon dia"
        format_prompt("Bon dia", "balear") -> "balear: Bon dia"
    """
    voice_name = DIALECT_VOICES.get(dialect, dialect)
    return f"{voice_name}: {text}"


def load_model(model_path, device):
    """Carga el modelo y tokenizer."""
    print(f"Cargando modelo desde: {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map=device
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"Modelo cargado en: {device}")
    return model, tokenizer


def decode_audio_tokens(audio_tokens, sample_rate=24000):
    """
    Decodifica tokens de audio a waveform usando SNAC.

    NOTA: Esta es una función placeholder. La implementación real
    depende de cómo Orpheus estructura los tokens de audio.
    """
    try:
        from snac import SNAC

        # Cargar modelo SNAC
        snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")

        # TODO: Implementar la lógica de decodificación
        # Esto depende de cómo el modelo genera los tokens de audio
        # Típicamente necesitas:
        # 1. Separar tokens de texto de tokens de audio
        # 2. Dividir tokens de audio en 3 niveles jerárquicos
        # 3. Decodificar con SNAC

        print("⚠️  Decodificación de audio en desarrollo")
        print("   Por ahora, usa orpheus-speech o vllm para inferencia completa")

        return None

    except ImportError:
        print("ERROR: snac no está instalado")
        print("Instala con: pip install snac")
        return None


def generate_speech(model, tokenizer, text, dialect, args):
    """
    Genera audio para el texto en el dialecto especificado.
    """
    # Formatear prompt con dialecto
    prompt = format_prompt(text, dialect)

    print(f"\nGenerando audio:")
    print(f"  Texto: {text}")
    print(f"  Dialecto: {dialect}")
    print(f"  Prompt: {prompt}")

    # Tokenizar texto
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=args.max_length
    ).to(args.device)

    print(f"  Tokens de entrada: {inputs['input_ids'].shape[1]}")

    # Generar
    print("\nGenerando tokens...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=args.max_length,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    print(f"  Tokens generados: {outputs.shape[1]}")

    # Los tokens generados incluyen texto + audio
    # Necesitamos decodificar la parte de audio

    # NOTA: Esta parte requiere la implementación completa del decoder
    # Por ahora, guardamos los tokens generados

    print("\n⚠️  NOTA IMPORTANTE:")
    print("   Este script genera los tokens pero no los decodifica a audio.")
    print("   Para inferencia completa, usa:")
    print("   1. orpheus-speech package (pip install orpheus-speech)")
    print("   2. O vllm con el modelo")
    print("\n   Ver: scripts/inference_with_orpheus_package.py")

    return outputs


def main():
    args = parse_args()

    print("="*70)
    print("INFERENCIA DIALECTAL - ORPHEUS TTS CATALÁN")
    print("="*70)

    # Cargar modelo
    model, tokenizer = load_model(args.model_path, args.device)

    # Generar
    outputs = generate_speech(model, tokenizer, args.text, args.dialect, args)

    print("\n" + "="*70)
    print("Generación completada")
    print("="*70)

    # Guardar información
    output_path = Path(args.output)
    info_path = output_path.with_suffix('.json')

    info = {
        'text': args.text,
        'dialect': args.dialect,
        'prompt': format_prompt(args.text, args.dialect),
        'model_path': args.model_path,
        'num_tokens_generated': outputs.shape[1],
        'parameters': {
            'temperature': args.temperature,
            'repetition_penalty': args.repetition_penalty,
            'max_length': args.max_length,
        }
    }

    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"\nInformación guardada en: {info_path}")


if __name__ == '__main__':
    main()
