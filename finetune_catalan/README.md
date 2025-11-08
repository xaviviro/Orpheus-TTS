# Fine-tuning de Orpheus TTS para Catalán con Variantes Dialectales

Este directorio contiene todos los scripts y configuraciones necesarias para entrenar Orpheus TTS con datasets en catalán que incluyen variantes dialectales (balear, central, nord, nord-occidental y valencià).

## Tabla de Contenidos

- [Descripción General](#descripción-general)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Datasets Disponibles](#datasets-disponibles)
- [Preparación del Entorno](#preparación-del-entorno)
- [Pipeline Completo](#pipeline-completo)
- [Configuración](#configuración)
- [Uso en RunPod](#uso-en-runpod)
- [Resolución de Problemas](#resolución-de-problemas)

## Descripción General

Este proyecto adapta Orpheus TTS (un modelo TTS de código abierto basado en Llama-3b) para generar voz en catalán con soporte para múltiples dialectos. El sistema:

- Procesa datasets de Common Voice en catalán
- Mantiene la distinción entre variantes dialectales
- Asigna voces diferentes a cada dialecto
- Permite fine-tuning del modelo preentrenado de Orpheus

### Variantes Dialectales Soportadas

| Dialecto | Código | Voz Asignada | Descripción |
|----------|--------|--------------|-------------|
| Balear | `balearic` | maria | Catalán de las Islas Baleares |
| Central | `central` | pau | Catalán central (Barcelona) |
| Nord | `northern` | montse | Catalán del norte (Francia) |
| Nord-occidental | `northwestern` | jordi | Catalán del noroeste (Lleida) |
| Valencià | `valencian` | carla | Valenciano |

## Estructura del Proyecto

```
finetune_catalan/
├── README.md                    # Esta documentación
├── setup_runpod.sh             # Script de configuración automática para RunPod
│
├── configs/
│   └── config_catalan.yaml     # Configuración de entrenamiento
│
├── scripts/
│   ├── prepare_commonvoice_catalan.py  # Prepara dataset de Common Voice
│   ├── tokenize_dataset.py            # Tokeniza audio y texto
│   └── train_catalan.py               # Script de entrenamiento
│
├── data/                       # Datos (se crea automáticamente)
│   ├── raw/                   # Datasets sin procesar
│   ├── processed/             # Datasets procesados
│   └── tokenized/             # Datasets tokenizados
│
├── checkpoints/               # Checkpoints del modelo
├── logs/                      # Logs de entrenamiento
└── outputs/                   # Modelos finales
```

## Datasets Disponibles

### 1. Dataset Principal (Recomendado)

**Nombre**: `projecte-aina/commonvoice_benchmark_catalan_accents`

- Basado en Common Voice v17
- Pre-anotado con información dialectal experta
- ~2,700 horas de audio de entrenamiento
- Splits de evaluación por dialecto y género
- Metadata de calidad incluida

**Campos importantes**:
- `annotated_accent`: Acento anotado por expertos
- `annotated_gender`: Género anotado por expertos
- `mean quality`: Puntuación de calidad promedio
- `sentence`: Texto transcrito
- `audio`: Audio en formato MP3

### 2. Datasets Alternativos

**Common Voice 13.0**: `mozilla-foundation/common_voice_13_0`
- Más grande pero menos curado
- Sin anotaciones dialectales específicas
- Requiere más preprocesamiento

**Nota**: A partir de octubre 2025, los datasets de Common Voice se distribuyen principalmente a través de Mozilla Data Collective.

## Preparación del Entorno

### Opción A: RunPod (Recomendado)

```bash
# 1. Conectar a RunPod y ejecutar el script de setup
chmod +x setup_runpod.sh
./setup_runpod.sh

# El script instalará automáticamente todo lo necesario
```

### Opción B: Configuración Manual

**Requisitos**:
- Python 3.10+
- CUDA 11.8+ o 12.0+
- GPU con al menos 24GB VRAM (recomendado: RTX 4090, A100)
- ~100GB de espacio en disco

**Instalación**:

```bash
# 1. Instalar PyTorch con CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Instalar dependencias principales
pip install transformers>=4.40.0 datasets accelerate

# 3. Instalar Flash Attention (opcional pero recomendado)
pip install flash-attn --no-build-isolation

# 4. Instalar dependencias de audio
pip install librosa soundfile snac

# 5. Instalar herramientas de logging
pip install wandb tensorboard

# 6. Instalar PEFT para LoRA (opcional)
pip install peft
```

## Pipeline Completo

### Paso 1: Autenticación

```bash
# Hugging Face (requerido)
huggingface-cli login

# WandB (opcional, para logging)
wandb login
```

### Paso 2: Preparar Dataset

```bash
python scripts/prepare_commonvoice_catalan.py \
    --dataset_name projecte-aina/commonvoice_benchmark_catalan_accents \
    --output_dir ./data/processed \
    --samples_per_accent 500 \
    --min_duration 1.0 \
    --max_duration 30.0 \
    --target_sample_rate 24000 \
    --accents balearic central northern northwestern valencian
```

**Parámetros importantes**:
- `--samples_per_accent`: Número de muestras por dialecto (None para todas)
- `--min_duration`/`--max_duration`: Filtros de duración en segundos
- `--target_sample_rate`: 24000 Hz (requerido por Orpheus)
- `--push_to_hub`: Subir dataset procesado a HuggingFace Hub

**Salida esperada**:
- Dataset balanceado por dialectos
- Audio resampleado a 24kHz
- Metadata preservada
- División train/validation (90/10)

### Paso 3: Tokenizar Dataset

```bash
python scripts/tokenize_dataset.py \
    --input_dir ./data/processed \
    --output_dir ./data/tokenized \
    --model_name canopylabs/orpheus-tts-0.1-pretrained \
    --snac_model hubertsiuzdak/snac_24khz \
    --max_length 8192 \
    --device cuda
```

**Nota**: Este paso puede tardar varias horas dependiendo del tamaño del dataset.

### Paso 4: Configurar Entrenamiento

Edita [configs/config_catalan.yaml](configs/config_catalan.yaml):

```yaml
# Cambiar el path del dataset
TTS_dataset: "./data/tokenized"  # O tu usuario/nombre-dataset en HF

# Ajustar hiperparámetros según tu GPU
training:
  batch_size: 2              # Reducir si hay OOM
  gradient_accumulation_steps: 4
  learning_rate: 5.0e-5
  epochs: 3
```

### Paso 5: Entrenar Modelo

```bash
# Entrenamiento básico (1 GPU)
python scripts/train_catalan.py --config configs/config_catalan.yaml

# Entrenamiento multi-GPU con accelerate
accelerate launch scripts/train_catalan.py --config configs/config_catalan.yaml

# Reanudar desde checkpoint
python scripts/train_catalan.py \
    --config configs/config_catalan.yaml \
    --resume_from_checkpoint ./checkpoints/checkpoint-1000
```

### Paso 6: Evaluar Modelo

```bash
# El script de entrenamiento ya incluye evaluación automática
# Los resultados se guardan en ./logs/

# Ver con tensorboard
tensorboard --logdir ./logs/

# O ver en WandB
# https://wandb.ai/tu-usuario/orpheus-catalan-tts
```

## Configuración

### Configuración Básica

El archivo [configs/config_catalan.yaml](configs/config_catalan.yaml) contiene toda la configuración:

```yaml
# Dataset
TTS_dataset: "tu_usuario/orpheus-catalan-tts"

# Modelo
model_name: "canopylabs/orpheus-tts-0.1-pretrained"

# Entrenamiento
training:
  epochs: 3
  batch_size: 2
  learning_rate: 5.0e-5
  bf16: true
```

### Optimizaciones de Memoria

Si tienes problemas de memoria (OOM):

```yaml
training:
  batch_size: 1                    # Reducir batch size
  gradient_accumulation_steps: 8   # Aumentar acumulación

advanced:
  gradient_checkpointing: true     # Activar checkpointing
  use_lora: true                   # Usar LoRA (más eficiente)
```

### Fine-tuning con LoRA

Para entrenamiento más eficiente en memoria:

```yaml
advanced:
  use_lora: true
  lora_config:
    r: 16
    lora_alpha: 32
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
    lora_dropout: 0.05
```

## Uso en RunPod

### Configuración Inicial

1. **Crear Pod en RunPod**:
   - Template: PyTorch 2.0+ con CUDA 12.0+
   - GPU recomendada: RTX 4090, A100 40GB, o H100
   - Volumen de red: 100GB+ (para datos y checkpoints)

2. **Ejecutar Setup**:
```bash
# Clonar repositorio
git clone https://github.com/canopyai/Orpheus-TTS.git
cd Orpheus-TTS/finetune_catalan

# Ejecutar setup automático
chmod +x setup_runpod.sh
./setup_runpod.sh
```

3. **El script de setup creará automáticamente**:
   - Directorios de trabajo
   - Scripts auxiliares
   - Variables de entorno
   - Jupyter notebooks de ejemplo

### Scripts de Utilidad Creados

Después del setup, tendrás estos scripts en `/workspace/`:

```bash
# Preparar datos
/workspace/prepare_data.sh

# Tokenizar datos
/workspace/tokenize_data.sh

# Entrenar modelo
/workspace/train.sh

# Monitorear GPU
/workspace/monitor_gpu.sh
```

### Persistencia de Datos

Los datos y checkpoints se guardan en volúmenes persistentes:

```
/workspace/data/          # Datasets (persistente)
/workspace/checkpoints/   # Checkpoints del modelo (persistente)
/workspace/logs/          # Logs (persistente)
```

Configura el volumen de red en RunPod para no perder los datos.

### Monitoreo

```bash
# Terminal 1: Entrenar
tmux new -s train
/workspace/train.sh

# Terminal 2: Monitorear
tmux new -s monitor
watch -n 1 nvidia-smi

# Terminal 3: Ver logs
tmux new -s logs
tail -f /workspace/logs/training.log
```

## Análisis de los Datasets

### ¿Los Datasets Requieren Tratamiento?

**SÍ**, los datasets de Common Voice requieren preprocesamiento significativo:

#### 1. Filtrado por Calidad

**Problema**: Common Voice contiene grabaciones de calidad variable.

**Solución implementada**:
- Filtrar por puntuación de calidad (`mean quality > 2.5`)
- Filtrar por votos positivos/negativos
- Usar anotaciones expertas del dataset de projecte-aina

```python
def filter_by_quality(example):
    if 'mean quality' in example:
        return example['mean quality'] > 2.5
    # Fallback a votos
    up = example.get('up_votes', 0)
    down = example.get('down_votes', 0)
    return up / (up + down) > 0.7
```

#### 2. Filtrado por Duración

**Problema**: Audios muy cortos o muy largos degradan el entrenamiento.

**Solución implementada**:
- Duración mínima: 1 segundo
- Duración máxima: 30 segundos
- Ajustable según necesidades

#### 3. Resampling de Audio

**Problema**: Common Voice está en 48kHz, Orpheus requiere 24kHz.

**Solución implementada**:
- Resampleo automático a 24kHz usando librosa
- Preservación de calidad de audio

```python
audio_array = librosa.resample(
    audio_array,
    orig_sr=48000,
    target_sr=24000
)
```

#### 4. Balanceo Dialectal

**Problema**: Distribución desigual de dialectos en el dataset original.

**Solución implementada**:
- Sampling estratificado por dialecto
- Opción de limitar muestras por dialecto
- Preservación de diversidad dialectal

```python
balanced_data = balance_dataset_by_accent(
    dataset,
    accents=['balearic', 'central', 'northern', 'northwestern', 'valencian'],
    samples_per_accent=500
)
```

#### 5. Formateo de Texto

**Problema**: Orpheus requiere un formato específico de prompt.

**Solución implementada**:
```python
# Formato requerido por Orpheus
formatted_text = f"{voice_name}: {texto_original}"

# Ejemplo: "pau: Bon dia! Com estàs?"
```

#### 6. Tokenización de Audio

**Problema**: Orpheus usa SNAC para tokenización de audio.

**Solución implementada**:
- Uso del modelo SNAC 24kHz
- Tokenización en 3 niveles jerárquicos
- Concatenación con tokens de texto

### Estadísticas Esperadas

Después del procesamiento, deberías ver algo como:

```
=================================================
ESTADÍSTICAS DEL DATASET FINAL
=================================================
Total de ejemplos (train): 2,250
Total de ejemplos (validation): 250

Distribución por acento:
  - balearic: 450 muestras
  - central: 450 muestras
  - northern: 450 muestras
  - northwestern: 450 muestras
  - valencian: 450 muestras

Duración promedio: 5.2s
Duración total: 3.25 horas
```

## Resolución de Problemas

### Error: OOM (Out of Memory)

```yaml
# Reducir batch size y usar gradient accumulation
training:
  batch_size: 1
  gradient_accumulation_steps: 16

advanced:
  gradient_checkpointing: true
```

### Error: Flash Attention no disponible

```yaml
# Usar implementación estándar
model_config:
  attn_implementation: "eager"  # En lugar de "flash_attention_2"
```

### Error: Dataset no se descarga

```bash
# Descargar manualmente
huggingface-cli download projecte-aina/commonvoice_benchmark_catalan_accents \
    --repo-type dataset \
    --local-dir ./data/raw/
```

### Audio de baja calidad

```python
# Ajustar filtros en prepare_commonvoice_catalan.py
--min_duration 2.0      # Aumentar duración mínima
--samples_per_accent 300  # Ser más selectivo
```

### Entrenamiento muy lento

```yaml
# Optimizaciones
training:
  dataloader_num_workers: 4
  dataloader_pin_memory: true

data:
  preprocessing_num_workers: 8
```

## Mejores Prácticas

1. **Empezar con un subset pequeño**: Prueba con 100 ejemplos primero
2. **Monitorear métricas**: Usa WandB o TensorBoard desde el inicio
3. **Guardar checkpoints frecuentemente**: No confíes solo en el modelo final
4. **Validar calidad de audio**: Escucha samples antes de entrenar todo
5. **Usar tmux/screen**: Para sesiones persistentes en RunPod

## Recursos Adicionales

- [Documentación Orpheus TTS](https://github.com/canopyai/Orpheus-TTS)
- [Common Voice Catalan Dataset](https://huggingface.co/datasets/projecte-aina/commonvoice_benchmark_catalan_accents)
- [Guía de Fine-tuning Orpheus](https://canopylabs.ai/releases/orpheus_can_speak_any_language#training)
- [RunPod Documentation](https://docs.runpod.io/)

## Licencia

Este proyecto sigue las licencias de:
- Orpheus TTS: [LICENSE](../LICENSE)
- Common Voice: CC BY 4.0

## Contacto y Contribuciones

Para problemas o mejoras, abre un issue en el repositorio principal de Orpheus TTS.

---

Creado para el fine-tuning de Orpheus TTS en catalán con soporte dialectal.




python scripts/tokenize_dataset.py \
    --input_dir /workspace/data/processed \
    --output_dir /workspace/data/tokenized \
    --hf_repo xaviviro/cv_23_ca_tokenized \
    --model_name canopylabs/orpheus-tts-0.1-pretrained \
    --snac_model hubertsiuzdak/snac_24khz \
    --batch_size 16 \
    --max_length 8192 \
    --device cuda


python scripts/tokenize_simple.py --input_dir /workspace/data/processed --output_dir /workspace/data/tokenized_simple


python scripts/upload_to_hf.py \
  --dataset_dir /workspace/data/tokenized_simple \
  --hf_repo xaviviro/cv_23_ca_distilled_tokenized \
  --private \
  --commit_message "Initial upload of tokenized dataset