"""
Script de inferencia usando orpheus-speech package.

Este script usa el paquete oficial de Orpheus para inferencia completa
incluyendo decodificación de audio.

Uso:
    pip install orpheus-speech vllm==0.7.3

    python inference_with_orpheus_package.py \
        --model_path ./checkpoints/final_model \
        --text "Bon dia! Com estàs avui?" \
        --dialect central \
        --output output.wav
"""

import argparse
import wave
import time
from pathlib import Path
import json


# Mapeo de dialectos
DIALECT_VOICES = {
    'central': 'central',
    'balearic': 'balear',
    'balear': 'balear',
    'valencian': 'valencia',
    'valencia': 'valencia',
    'northern': 'nord',
    'nord': 'nord',
    'northwestern': 'occidental',
    'occidental': 'occidental',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Inferencia con Orpheus-speech')
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
        choices=list(DIALECT_VOICES.keys()),
        help='Dialecto a usar'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output.wav',
        help='Archivo de salida'
    )
    parser.add_argument(
        '--max_model_len',
        type=int,
        default=2048,
        help='Longitud máxima del modelo'
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
        help='Penalización por repetición'
    )

    return parser.parse_args()


def format_prompt(text, dialect):
    """Formatea el texto con el dialecto."""
    voice_name = DIALECT_VOICES[dialect]
    return f"{voice_name}: {text}"


def generate_speech(args):
    """Genera audio usando orpheus-speech."""
    try:
        from orpheus_tts import OrpheusModel
    except ImportError:
        print("ERROR: orpheus-speech no está instalado")
        print("Instala con: pip install orpheus-speech vllm==0.7.3")
        return None

    print("="*70)
    print("GENERACIÓN DE AUDIO - ORPHEUS TTS CATALÁN")
    print("="*70)

    # Cargar modelo
    print(f"\nCargando modelo: {args.model_path}")
    model = OrpheusModel(
        model_name=args.model_path,
        max_model_len=args.max_model_len
    )

    # Formatear prompt
    voice_name = DIALECT_VOICES[args.dialect]
    prompt = format_prompt(args.text, args.dialect)

    print(f"\nGenerando audio:")
    print(f"  Texto original: {args.text}")
    print(f"  Dialecto: {args.dialect}")
    print(f"  Voz: {voice_name}")
    print(f"  Prompt: {prompt}")

    # Generar
    start_time = time.monotonic()

    syn_tokens = model.generate_speech(
        prompt=prompt,
        voice=voice_name,  # Esto es el dialecto en nuestro caso
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
    )

    # Guardar audio
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nGuardando audio en: {output_path}")

    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)

        total_frames = 0
        chunk_counter = 0

        for audio_chunk in syn_tokens:
            chunk_counter += 1
            frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
            total_frames += frame_count
            wf.writeframes(audio_chunk)

        duration = total_frames / wf.getframerate()

    end_time = time.monotonic()
    generation_time = end_time - start_time

    print(f"\n✅ Audio generado:")
    print(f"  Duración: {duration:.2f} segundos")
    print(f"  Tiempo de generación: {generation_time:.2f} segundos")
    print(f"  RTF (Real-Time Factor): {generation_time/duration:.2f}x")
    print(f"  Chunks procesados: {chunk_counter}")

    # Guardar metadata
    info_path = output_path.with_suffix('.json')
    info = {
        'text': args.text,
        'dialect': args.dialect,
        'voice': voice_name,
        'prompt': prompt,
        'output_file': str(output_path),
        'duration_seconds': duration,
        'generation_time_seconds': generation_time,
        'rtf': generation_time / duration,
        'parameters': {
            'temperature': args.temperature,
            'repetition_penalty': args.repetition_penalty,
            'max_model_len': args.max_model_len,
        }
    }

    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"\nMetadata guardada en: {info_path}")

    return output_path


def main():
    args = parse_args()

    try:
        output_path = generate_speech(args)

        if output_path:
            print("\n" + "="*70)
            print("¡GENERACIÓN COMPLETADA CON ÉXITO!")
            print("="*70)
            print(f"\nArchivo de audio: {output_path}")
            print(f"\nPara escucharlo:")
            print(f"  • macOS: afplay {output_path}")
            print(f"  • Linux: aplay {output_path}")
            print(f"  • Windows: start {output_path}")

    except Exception as e:
        print(f"\n❌ Error durante la generación: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
