"""
Script simplificado de tokenización para Orpheus TTS en Catalán.
Solo crea input_ids como el dataset original de Orpheus.
"""

import argparse
from datasets import load_from_disk, load_dataset, Dataset, DatasetDict
from pathlib import Path
import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='canopylabs/orpheus-tts-0.1-pretrained')
    parser.add_argument('--snac_model', type=str, default='hubertsiuzdak/snac_24khz')
    parser.add_argument('--max_length', type=int, default=8192)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=16)
    return parser.parse_args()

def load_snac_model(model_name, device):
    from snac import SNAC
    print(f"Cargando modelo SNAC: {model_name}")
    model = SNAC.from_pretrained(model_name).to(device)
    model.eval()
    return model

def tokenize_audio(audio_array, snac_model, device):
    audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        codes = snac_model.encode(audio_tensor)
    tokens = []
    for level_codes in codes:
        tokens.append(level_codes.cpu().numpy().flatten())
    return np.concatenate(tokens)

def main():
    args = parse_args()

    # Cargar dataset
    print(f"Cargando dataset desde: {args.input_dir}")
    dataset = load_from_disk(args.input_dir)

    # Cargar tokenizer y SNAC
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    snac_model = load_snac_model(args.snac_model, args.device)

    def process_split(split_data, split_name):
        print(f"\nProcesando {split_name}...")
        tokenized = []

        for example in tqdm(split_data, desc=split_name):
            try:
                # Tokenizar texto
                text_tokens = tokenizer(
                    example['text'],
                    truncation=True,
                    max_length=args.max_length // 2
                )['input_ids']

                # Tokenizar audio
                audio_tokens = tokenize_audio(example['audio']['array'], snac_model, args.device)

                # Combinar
                input_ids = np.concatenate([
                    np.array(text_tokens),
                    audio_tokens
                ])

                if len(input_ids) > args.max_length:
                    input_ids = input_ids[:args.max_length]

                tokenized.append({'input_ids': input_ids.tolist()})

            except Exception as e:
                print(f"Error: {e}")
                continue

        return tokenized

    # Procesar splits
    train_data = process_split(dataset['train'], 'train')
    val_data = process_split(dataset['validation'], 'validation')

    # Crear dataset
    tokenized_dataset = DatasetDict({
        'train': Dataset.from_list(train_data),
        'validation': Dataset.from_list(val_data)
    })

    # Guardar
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGuardando en: {output_dir}")
    tokenized_dataset.save_to_disk(str(output_dir))

    print(f"\n✅ Completado!")
    print(f"Train: {len(tokenized_dataset['train'])} ejemplos")
    print(f"Validation: {len(tokenized_dataset['validation'])} ejemplos")

if __name__ == '__main__':
    main()
