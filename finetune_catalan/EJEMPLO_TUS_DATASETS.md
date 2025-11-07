# Ejemplo de Uso con Tus Datasets (xaviviro/cv_23_ca_*)

## Confirmación: Estructura de tus Datasets

Tus datasets tienen la siguiente estructura confirmada:

```python
{
    'audio': {
        'array': [...],
        'sampling_rate': 48000
    },
    'sentence': "Text en català",
    'client_id': "abc123...",      # ✅ Este campo identifica al hablante
    'age': "twenties",
    'gender': "male",
    'accents': "central",
    'variant': "Central",
    'locale': "ca",
    'sentence_domain': "..."
}
```

**NOTA IMPORTANTE**: `client_id` es el identificador del hablante (speaker).

## Uso Completo: Estrategia Recomendada

### Paso 0: Verificar tus Datasets

```bash
# Listar todos tus datasets catalanes
huggingface-cli list | grep cv_23_ca

# Ejemplo de salida esperada:
# xaviviro/cv_23_ca_central
# xaviviro/cv_23_ca_balearic
# xaviviro/cv_23_ca_valencian
# (etc.)
```

### Paso 1: Analizar Distribución de Hablantes

```bash
# Analizar cuántos hablantes tienes y sus distribuciones
python scripts/analyze_speaker_distribution.py \
    --datasets \
        xaviviro/cv_23_ca_central \
        xaviviro/cv_23_ca_balearic \
        xaviviro/cv_23_ca_valencian \
    --output_dir ./analysis/ \
    --min_samples_fixed 100 \
    --min_samples_multispeaker 5
```

**Esto te mostrará**:
```
Analizando: xaviviro/cv_23_ca_central
====================================================================
Total hablantes: 350
Muestras por hablante (promedio): 12.5
Muestras por hablante (mediana): 8
Rango: 1 - 156

ANÁLISIS DE ESTRATEGIAS:
  Voces fijas candidatas (>=100 muestras):
    - 5 hablantes
    - 650 muestras totales

  Top 5 candidatos para voces fijas:
    1. client_id_1 : 156 muestras, 13.2 min, male
    2. client_id_2 : 134 muestras, 11.1 min, female
    3. client_id_3 : 121 muestras, 9.8 min, male
    ...

  Multi-speaker candidatos (>=5 muestras):
    - 280 hablantes
    - 4,200 muestras totales
```

Esto confirma que la **estrategia por dialecto** es la mejor opción.

### Paso 2: Preparar Datos por Dialecto

```bash
python scripts/prepare_by_dialect.py \
    --datasets \
        xaviviro/cv_23_ca_central \
        xaviviro/cv_23_ca_balearic \
        xaviviro/cv_23_ca_valencian \
    --output_dir ./data/processed_dialect \
    --min_samples_per_speaker 3 \
    --max_samples_per_speaker 50 \
    --balance_speakers \
    --save_speaker_metadata \
    --min_duration 1.5 \
    --max_duration 15.0
```

**Parámetros explicados**:

- `--min_samples_per_speaker 3`: Solo incluir hablantes (client_id) con al menos 3 audios
  - ✅ Evita hablantes con muy pocos datos
  - ✅ Mantiene diversidad (no descarta demasiados)

- `--max_samples_per_speaker 50`: Máximo 50 audios por hablante
  - ✅ Evita que algunos hablantes dominen el dataset
  - ✅ Balancea la representación

- `--balance_speakers`: Activa el balanceo
  - ✅ Distribución más equitativa entre hablantes
  - ✅ Evita overfitting a voces específicas

- `--save_speaker_metadata`: Guarda info de cada hablante
  - ✅ Para voice cloning posterior
  - ✅ Incluye ejemplos representativos de cada speaker

**Salida esperada**:

```
Processing: xaviviro/cv_23_ca_central
======================================================================
Dialecto detectado: central
Voz de entrenamiento: central

Muestras cargadas: 4,500
Después del filtro: 4,200 muestras

Analizando hablantes...
Hablantes totales: 350
Hablantes con >=3 muestras: 280
Muestras después de filtrar hablantes: 4,100

Balanceando (max 50 por hablante)...
Muestras después de balancear: 3,200

Procesando ejemplos...
✅ Dataset procesado: 3,200 ejemplos

Estadísticas finales:
  Hablantes únicos: 280
  Promedio muestras/hablante: 11.4
  Duración total: 4.2 horas

======================================================================
COMBINANDO DIALECTOS
======================================================================

Total combinado: 9,800 muestras

Dividiendo train/validation (90/10)...
Train: 8,820 muestras
Validation: 980 muestras

Por dialecto:
  central (voz: 'central'):
    • Muestras: 3,200
    • Hablantes: 280
    • Duración: 4.2 horas
  balearic (voz: 'balear'):
    • Muestras: 3,400
    • Hablantes: 310
    • Duración: 4.5 horas
  valencian (voz: 'valencia'):
    • Muestras: 3,200
    • Hablantes: 290
    • Duración: 4.1 horas

✅ PREPARACIÓN COMPLETADA
```

### Paso 3: Verificar los Datos Procesados

```python
from datasets import load_from_disk

# Cargar dataset procesado
ds = load_from_disk('./data/processed_dialect')

# Ver primeros ejemplos
for i in range(3):
    example = ds['train'][i]
    print(f"\nEjemplo {i+1}:")
    print(f"  Texto formateado: {example['text']}")
    print(f"  Texto original: {example['original_text']}")
    print(f"  Dialecto: {example['dialect']}")
    print(f"  Voz: {example['voice_name']}")
    print(f"  Speaker ID: {example['speaker_id'][:20]}...")
    print(f"  Género: {example['gender']}")
    print(f"  Duración: {example['duration']:.2f}s")
```

**Salida esperada**:
```
Ejemplo 1:
  Texto formateado: central: Bon dia! Com estàs avui?
  Texto original: Bon dia! Com estàs avui?
  Dialecto: central
  Voz: central
  Speaker ID: 3a7f8e9c2d1b5a4e...
  Género: female
  Duración: 3.45s

Ejemplo 2:
  Texto formateado: balear: Bon dia! Com estàs avui?
  Texto original: Bon dia! Com estàs avui?
  Dialecto: balearic
  Voz: balear
  Speaker ID: 9c2d1b5a4e3a7f8e...
  Género: male
  Duración: 3.12s
```

### Paso 4: Revisar Metadata de Hablantes

```python
import json

# Cargar metadata guardada
with open('./data/processed_dialect/speaker_metadata.json', 'r') as f:
    speaker_meta = json.load(f)

# Ver estadísticas
for dialect in ['central', 'balearic', 'valencian']:
    if dialect in speaker_meta:
        speakers = speaker_meta[dialect]
        print(f"\n{dialect.upper()}:")
        print(f"  Total hablantes: {len(speakers)}")

        # Ver algunos ejemplos
        for speaker_id, info in list(speakers.items())[:3]:
            print(f"    - {speaker_id[:15]}...")
            print(f"      Género: {info['gender']}")
            print(f"      Edad: {info['age']}")
            print(f"      Ejemplos representativos: {len(info.get('representative_indices', []))}")
```

Esta metadata te será útil después para **seleccionar voces de referencia** para voice cloning.

### Paso 5: Tokenizar

```bash
python scripts/tokenize_dataset.py \
    --input_dir ./data/processed_dialect \
    --output_dir ./data/tokenized_dialect \
    --model_name canopylabs/orpheus-tts-0.1-pretrained \
    --snac_model hubertsiuzdak/snac_24khz \
    --max_length 8192 \
    --device cuda
```

**Tiempo estimado**: 2-3 horas para ~10k muestras en RTX 4090

### Paso 6: Configurar Entrenamiento

```yaml
# configs/config_catalan_dialect.yaml
TTS_dataset: "./data/tokenized_dialect"

model_name: "canopylabs/orpheus-tts-0.1-pretrained"

training:
  epochs: 3
  batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 5.0e-5
  bf16: true

data:
  # Voces por dialecto
  voice_mapping:
    central: "central"
    balearic: "balear"
    valencian: "valencia"
```

### Paso 7: Entrenar

```bash
# Con accelerate (recomendado)
accelerate launch scripts/train_catalan.py \
    --config configs/config_catalan_dialect.yaml

# O sin accelerate
python scripts/train_catalan.py \
    --config configs/config_catalan_dialect.yaml
```

**Recursos esperados**:
- VRAM: ~16GB (batch_size=2)
- RAM: ~32GB
- Tiempo: 6-8 horas (RTX 4090, 10k samples, 3 epochs)

### Paso 8: Monitorear Entrenamiento

```bash
# Terminal 1: Entrenamiento
# (ya ejecutado arriba)

# Terminal 2: Monitoreo GPU
watch -n 1 nvidia-smi

# Terminal 3: TensorBoard
tensorboard --logdir ./logs/

# O WandB
# https://wandb.ai/tu-usuario/orpheus-catalan-tts
```

### Paso 9: Después del Entrenamiento - Voice Cloning

Una vez tengas el modelo entrenado, podrás usarlo con voice cloning:

```python
# Cargar modelo entrenado
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./checkpoints/final_model")
tokenizer = AutoTokenizer.from_pretrained("./checkpoints/final_model")

# Usar con voice cloning
# (Ver ESTRATEGIA_RECOMENDADA.md para detalles de inferencia)
```

## Estadísticas Esperadas de tus Datasets

Basándome en datasets típicos de Common Voice 23:

| Dataset | Hablantes Aprox. | Muestras Totales | Duración Total |
|---------|------------------|------------------|----------------|
| cv_23_ca_central | 300-500 | 3,000-5,000 | 3-5 horas |
| cv_23_ca_balearic | 200-400 | 2,000-4,000 | 2-4 horas |
| cv_23_ca_valencian | 300-600 | 3,000-6,000 | 3-6 horas |

**Después de filtros y balanceo**:
- ~70-80% de las muestras originales
- ~70-80% de los hablantes (eliminando los que tienen <3 audios)
- Dataset balanceado y listo para entrenamiento

## Comandos Todo-en-Uno

```bash
# Preparar todo de una vez (recomendado ejecutar por separado primero)

# 1. Análisis
python scripts/analyze_speaker_distribution.py \
    --datasets xaviviro/cv_23_ca_central xaviviro/cv_23_ca_balearic xaviviro/cv_23_ca_valencian \
    --output_dir ./analysis/

# 2. Preparación por dialecto
python scripts/prepare_by_dialect.py \
    --datasets xaviviro/cv_23_ca_central xaviviro/cv_23_ca_balearic xaviviro/cv_23_ca_valencian \
    --output_dir ./data/processed_dialect \
    --min_samples_per_speaker 3 \
    --max_samples_per_speaker 50 \
    --balance_speakers \
    --save_speaker_metadata

# 3. Tokenización
python scripts/tokenize_dataset.py \
    --input_dir ./data/processed_dialect \
    --output_dir ./data/tokenized_dialect \
    --device cuda

# 4. Entrenamiento
accelerate launch scripts/train_catalan.py \
    --config configs/config_catalan_dialect.yaml
```

## Troubleshooting Específico

### Error: No se encuentra el dataset

```bash
# Verificar autenticación
huggingface-cli whoami

# Verificar que el dataset existe
huggingface-cli repo info xaviviro/cv_23_ca_central --repo-type dataset

# Si el dataset es privado, asegúrate de estar autenticado
huggingface-cli login
```

### Muy pocos hablantes después del filtro

```bash
# Reducir el mínimo de muestras por speaker
python scripts/prepare_by_dialect.py \
    --min_samples_per_speaker 2  # En lugar de 3
    ...
```

### Dataset muy grande, quieres probar primero

```bash
# Limitar muestras para prueba rápida
python scripts/prepare_by_dialect.py \
    --datasets xaviviro/cv_23_ca_central \
    --output_dir ./data/test_small \
    --samples_per_dialect 500  # Solo 500 muestras
    --min_samples_per_speaker 3
```

## Notas Importantes

1. **client_id = speaker_id**: Todos los scripts usan `client_id` correctamente como identificador del hablante

2. **Formato de voz**: El modelo se entrenará con formato `{dialecto}: {texto}` (ej: "central: Bon dia")

3. **Metadata guardada**: La info de cada speaker se guarda en `speaker_metadata.json` para uso posterior en voice cloning

4. **Balanceo importante**: Con `--balance_speakers`, evitas que hablantes con muchos audios dominen el entrenamiento

---

¿Listo para empezar? Comienza con el análisis para ver qué tienes:

```bash
python scripts/analyze_speaker_distribution.py \
    --datasets xaviviro/cv_23_ca_central \
    --output_dir ./analysis/
```
