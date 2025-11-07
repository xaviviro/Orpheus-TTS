# Uso con Datasets Personalizados (xaviviro/cv_23_ca_*)

Esta guía explica cómo usar tus datasets personalizados de Common Voice 23 en catalán para entrenar Orpheus TTS.

## Tus Datasets

Tienes datasets organizados por variante dialectal:

- `xaviviro/cv_23_ca_central` - Catalán central (Barcelona)
- (Probablemente también tengas):
  - `xaviviro/cv_23_ca_balearic` - Balear
  - `xaviviro/cv_23_ca_valencian` - Valencià
  - `xaviviro/cv_23_ca_northern` - Nord
  - `xaviviro/cv_23_ca_northwestern` - Nord-occidental

## Estructura de tus Datos

Cada muestra contiene:

```python
{
    'audio': {
        'array': [...],           # Array numpy del audio
        'sampling_rate': 48000    # 48kHz (se resampleará a 24kHz)
    },
    'sentence': "Text en català",  # Transcripción
    'client_id': "abc123...",      # ID anónimo del locutor
    'age': "twenties",             # Edad (opcional)
    'gender': "male",              # Género (opcional)
    'accents': "central",          # Acento específico
    'variant': "Central",          # Variante dialectal
    'locale': "ca",                # Código de lengua
    'sentence_domain': "..."       # Dominio (opcional)
}
```

## Pipeline de Preparación

### Opción 1: Procesar una Sola Variante

```bash
python scripts/prepare_custom_catalan.py \
    --datasets xaviviro/cv_23_ca_central \
    --output_dir ./data/processed_central \
    --samples_per_variant 1000 \
    --min_duration 1.0 \
    --max_duration 20.0
```

### Opción 2: Procesar Múltiples Variantes (Recomendado)

```bash
python scripts/prepare_custom_catalan.py \
    --datasets \
        xaviviro/cv_23_ca_central \
        xaviviro/cv_23_ca_balearic \
        xaviviro/cv_23_ca_valencian \
    --output_dir ./data/processed_multidialect \
    --samples_per_variant 500 \
    --min_duration 1.0 \
    --max_duration 20.0 \
    --target_sample_rate 24000
```

**Resultado esperado**:
```
====================================================================
ESTADÍSTICAS DEL DATASET FINAL
====================================================================
Total de ejemplos (train): 1,350
Total de ejemplos (validation): 150

Distribución por variante:
  - central:
      Muestras: 500
      Duración: 0.69 horas
  - balearic:
      Muestras: 500
      Duración: 0.71 horas
  - valencian:
      Muestras: 500
      Duración: 0.68 horas

Duración promedio: 5.02s
Duración total (train): 1.88 horas

Distribución por género:
  - male: 675 muestras (50.0%)
  - female: 675 muestras (50.0%)
```

### Opción 3: Con Filtros Adicionales

```bash
# Solo voces femeninas, edades 20-40
python scripts/prepare_custom_catalan.py \
    --datasets xaviviro/cv_23_ca_central xaviviro/cv_23_ca_balearic \
    --output_dir ./data/processed_female \
    --filter_by_gender female \
    --filter_by_age twenties thirties \
    --samples_per_variant 300
```

## Mapeo de Voces

El script asigna automáticamente nombres de voz según la variante:

| Variante | Código Dataset | Nombre de Voz | Formato Prompt |
|----------|----------------|---------------|----------------|
| Central | `cv_23_ca_central` | `pau` | `pau: text aquí` |
| Balearic | `cv_23_ca_balearic` | `maria` | `maria: text aquí` |
| Northern | `cv_23_ca_northern` | `montse` | `montse: text aquí` |
| Northwestern | `cv_23_ca_northwestern` | `jordi` | `jordi: text aquí` |
| Valencian | `cv_23_ca_valencian` | `carla` | `carla: text aquí` |

**Personalizar voces**:

Edita el archivo `scripts/prepare_custom_catalan.py`:

```python
VOICE_NAMES_BY_VARIANT = {
    'central': 'tu_nombre_preferido',
    'balearic': 'otro_nombre',
    # ...
}
```

## Tokenización

Después de preparar los datos, tokeniza:

```bash
python scripts/tokenize_dataset.py \
    --input_dir ./data/processed_multidialect \
    --output_dir ./data/tokenized_multidialect \
    --model_name canopylabs/orpheus-tts-0.1-pretrained \
    --snac_model hubertsiuzdak/snac_24khz \
    --device cuda
```

**Tiempo estimado**: ~1-2 horas para 1,500 muestras en una RTX 4090.

## Configuración para Entrenamiento

Edita `configs/config_catalan.yaml`:

```yaml
# Usar tu dataset tokenizado
TTS_dataset: "./data/tokenized_multidialect"

# O subir a HuggingFace y usar desde allí
# TTS_dataset: "tu_usuario/orpheus-catalan-cv23"

# Configuración de datos
data:
  accents:
    - central
    - balearic
    - valencian

  voice_mapping:
    central: "pau"
    balearic: "maria"
    valencian: "carla"
```

## Entrenamiento

```bash
# Entrenar con accelerate
accelerate launch scripts/train_catalan.py \
    --config configs/config_catalan.yaml
```

## Subir Dataset a HuggingFace Hub

Para compartir o reutilizar:

```bash
# Durante la preparación
python scripts/prepare_custom_catalan.py \
    --datasets xaviviro/cv_23_ca_central xaviviro/cv_23_ca_balearic \
    --output_dir ./data/processed \
    --push_to_hub \
    --hub_dataset_name tu_usuario/orpheus-catalan-cv23-processed

# O después de tokenizar
python scripts/tokenize_dataset.py \
    --input_dir ./data/processed \
    --output_dir ./data/tokenized \
    --push_to_hub \
    --hub_dataset_name tu_usuario/orpheus-catalan-cv23-tokenized
```

## Ejemplo Completo: Flujo Recomendado

```bash
# 1. Autenticación
huggingface-cli login
wandb login  # opcional

# 2. Preparar datos de múltiples variantes
python scripts/prepare_custom_catalan.py \
    --datasets \
        xaviviro/cv_23_ca_central \
        xaviviro/cv_23_ca_balearic \
        xaviviro/cv_23_ca_valencian \
    --output_dir ./data/processed \
    --samples_per_variant 500 \
    --min_duration 1.5 \
    --max_duration 15.0

# 3. Tokenizar
python scripts/tokenize_dataset.py \
    --input_dir ./data/processed \
    --output_dir ./data/tokenized \
    --device cuda

# 4. Verificar datos
python -c "
from datasets import load_from_disk
ds = load_from_disk('./data/tokenized')
print(f'Train: {len(ds[\"train\"])} samples')
print(f'Validation: {len(ds[\"validation\"])} samples')
print(f'Example: {ds[\"train\"][0]}')
"

# 5. Editar config
nano configs/config_catalan.yaml

# 6. Entrenar
accelerate launch scripts/train_catalan.py \
    --config configs/config_catalan.yaml

# 7. Monitorear (en otra terminal)
watch -n 1 nvidia-smi
# o
tensorboard --logdir ./logs/
```

## Verificación de Calidad

### Inspeccionar Dataset Procesado

```python
from datasets import load_from_disk
import soundfile as sf

# Cargar dataset
ds = load_from_disk('./data/processed')

# Inspeccionar primeros ejemplos
for i in range(5):
    example = ds['train'][i]
    print(f"\nEjemplo {i+1}:")
    print(f"  Texto: {example['original_text']}")
    print(f"  Prompt: {example['text']}")
    print(f"  Voz: {example['voice_name']}")
    print(f"  Variante: {example['variant']}")
    print(f"  Género: {example['gender']}")
    print(f"  Duración: {example['duration']:.2f}s")

    # Guardar audio para escuchar
    audio_array = example['audio']['array']
    sf.write(f'sample_{i}.wav', audio_array, 24000)
```

### Estadísticas Detalladas

```python
import numpy as np
from collections import Counter

ds = load_from_disk('./data/processed')

# Duraciones
durations = [ex['duration'] for ex in ds['train']]
print(f"Duración min/max/media: {np.min(durations):.2f}s / {np.max(durations):.2f}s / {np.mean(durations):.2f}s")

# Variantes
variants = [ex['variant'] for ex in ds['train']]
print("\nDistribución de variantes:")
for variant, count in Counter(variants).most_common():
    print(f"  {variant}: {count} ({count/len(variants)*100:.1f}%)")

# Géneros
genders = [ex['gender'] for ex in ds['train']]
print("\nDistribución de géneros:")
for gender, count in Counter(genders).most_common():
    print(f"  {gender}: {count} ({count/len(genders)*100:.1f}%)")

# Longitud de texto
text_lengths = [len(ex['original_text']) for ex in ds['train']]
print(f"\nLongitud de texto (caracteres):")
print(f"  Min/Max/Media: {np.min(text_lengths)} / {np.max(text_lengths)} / {np.mean(text_lengths):.0f}")
```

## Solución de Problemas

### Error: Dataset no encontrado

```bash
# Verificar que estás autenticado
huggingface-cli whoami

# Verificar que el dataset existe
huggingface-cli repo info xaviviro/cv_23_ca_central --repo-type dataset

# Listar tus datasets
huggingface-cli list | grep cv_23
```

### Error: Audio sample rate incorrecto

El script automáticamente resamplea de 48kHz a 24kHz. Si ves este error, verifica:

```python
# En el dataset
example = ds['train'][0]
print(f"Sample rate: {example['audio']['sampling_rate']}")
# Debe ser 24000 después del procesamiento
```

### Memoria insuficiente durante tokenización

```bash
# Reducir batch size (implícito en el código)
# O procesar en lotes más pequeños

# Dividir dataset en chunks
python scripts/prepare_custom_catalan.py \
    --datasets xaviviro/cv_23_ca_central \
    --output_dir ./data/chunk1 \
    --samples_per_variant 100  # Procesar de 100 en 100

# Luego combinar manualmente
```

## Siguientes Pasos

1. ✓ Preparar datos con `prepare_custom_catalan.py`
2. ✓ Tokenizar con `tokenize_dataset.py`
3. ✓ Configurar `config_catalan.yaml`
4. → Entrenar con `train_catalan.py`
5. → Evaluar e iterar

Para más detalles sobre el entrenamiento, ver [README.md](README.md).
