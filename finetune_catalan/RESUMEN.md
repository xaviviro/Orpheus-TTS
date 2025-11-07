# Resumen del Proyecto: Fine-tuning Orpheus TTS para Catalán

## ¿Qué se ha creado?

Se ha montado una estructura completa para hacer fine-tuning de Orpheus TTS al catalán con soporte para variantes dialectales usando tus datasets de Common Voice.

## Estructura del Proyecto

```
finetune_catalan/
│
├── 📄 QUICKSTART.md              ← Guía de inicio rápido (EMPIEZA AQUÍ)
├── 📄 README.md                  ← Documentación completa del proyecto
├── 📄 USAGE_CUSTOM_DATASETS.md   ← Guía específica para tus datasets
├── 📄 TOKENIZATION_GUIDE.md      ← Explicación técnica de cómo funciona la tokenización
├── 📄 requirements.txt           ← Dependencias de Python
├── 📄 .gitignore                 ← Archivos a ignorar en git
│
├── 🔧 setup_runpod.sh           ← Script de setup automático para RunPod
│
├── 📁 configs/
│   └── config_catalan.yaml      ← Configuración de entrenamiento
│
├── 📁 scripts/
│   ├── prepare_custom_catalan.py          ← Para tus datasets (xaviviro/cv_23_ca_*)
│   ├── prepare_commonvoice_catalan.py     ← Para datasets de projecte-aina
│   ├── tokenize_dataset.py                ← Tokenización de audio y texto
│   └── train_catalan.py                   ← Script de entrenamiento
│
└── 📁 data/                     ← Se crea automáticamente
    ├── raw/                    ← Datasets sin procesar
    ├── processed/              ← Datasets procesados
    └── tokenized/              ← Datasets tokenizados
```

## Archivos Clave

### 1. Scripts de Preparación

#### `scripts/prepare_custom_catalan.py`
Script principal para tus datasets. Hace:
- ✓ Carga múltiples variantes dialectales de tus datasets
- ✓ Filtra por duración (min/max)
- ✓ Filtra por calidad, género, edad
- ✓ Resamplea audio de 48kHz a 24kHz
- ✓ Asigna voces por dialecto (pau, maria, carla, etc.)
- ✓ Formatea texto al estilo Orpheus: `{voz}: {texto}`
- ✓ Balancea dataset por variantes
- ✓ Divide en train/validation

**Uso**:
```bash
python scripts/prepare_custom_catalan.py \
    --datasets xaviviro/cv_23_ca_central xaviviro/cv_23_ca_balearic \
    --output_dir ./data/processed \
    --samples_per_variant 500
```

#### `scripts/tokenize_dataset.py`
Tokeniza audio usando SNAC y texto usando el tokenizer de Orpheus:
- ✓ Tokeniza texto con vocabulario de Llama
- ✓ Tokeniza audio con SNAC en 3 niveles jerárquicos
- ✓ Combina tokens de texto + audio en secuencia unificada
- ✓ Crea formato para causal language modeling

**Uso**:
```bash
python scripts/tokenize_dataset.py \
    --input_dir ./data/processed \
    --output_dir ./data/tokenized \
    --device cuda
```

#### `scripts/train_catalan.py`
Script de entrenamiento con Transformers Trainer:
- ✓ Carga modelo preentrenado de Orpheus
- ✓ Soporta multi-GPU con accelerate
- ✓ Logging con WandB/TensorBoard
- ✓ Guardado de checkpoints
- ✓ Evaluación automática

**Uso**:
```bash
accelerate launch scripts/train_catalan.py \
    --config configs/config_catalan.yaml
```

### 2. Configuración

#### `configs/config_catalan.yaml`
Archivo central de configuración:
```yaml
# Dataset
TTS_dataset: "./data/tokenized"  # O tu dataset en HF

# Modelo
model_name: "canopylabs/orpheus-tts-0.1-pretrained"

# Entrenamiento
training:
  epochs: 3
  batch_size: 2
  learning_rate: 5.0e-5
  bf16: true

# Voces por dialecto
data:
  voice_mapping:
    central: "pau"
    balearic: "maria"
    valencian: "carla"
```

### 3. Setup Automático

#### `setup_runpod.sh`
Script bash que configura TODO el entorno en RunPod:
- ✓ Instala PyTorch con CUDA
- ✓ Instala todas las dependencias
- ✓ Configura estructura de directorios
- ✓ Crea variables de entorno
- ✓ Genera scripts auxiliares
- ✓ Configura Jupyter notebooks

**Uso**:
```bash
chmod +x setup_runpod.sh
./setup_runpod.sh
```

## Pipeline Completo

### Fase 1: Setup (1 vez)
```bash
# En RunPod
./setup_runpod.sh

# Autenticación
huggingface-cli login
wandb login
```

### Fase 2: Preparación de Datos
```bash
# Procesar tus datasets
python scripts/prepare_custom_catalan.py \
    --datasets xaviviro/cv_23_ca_central \
    --output_dir ./data/processed \
    --samples_per_variant 500

# Resultado: ~1,500 muestras procesadas y balanceadas
```

### Fase 3: Tokenización
```bash
# Tokenizar audio + texto
python scripts/tokenize_dataset.py \
    --input_dir ./data/processed \
    --output_dir ./data/tokenized \
    --device cuda

# Resultado: Dataset listo para entrenamiento
```

### Fase 4: Entrenamiento
```bash
# Configurar
nano configs/config_catalan.yaml

# Entrenar
accelerate launch scripts/train_catalan.py \
    --config configs/config_catalan.yaml

# Resultado: Modelo fine-tuneado en ./checkpoints/
```

## Análisis de los Datasets

### ¿Por qué necesitan tratamiento?

Los datasets de Common Voice **requieren preprocesamiento significativo**:

1. **Frecuencia de muestreo**: 48kHz → 24kHz (requerido por Orpheus)
2. **Duración**: Filtrar audios muy cortos (<1s) o muy largos (>30s)
3. **Calidad**: Filtrar por votos, calidad anotada
4. **Formato**: Convertir texto a formato `{voz}: {texto}`
5. **Balanceo**: Equilibrar muestras entre dialectos
6. **Tokenización**: Convertir audio a tokens discretos con SNAC

### Tratamiento Implementado

| Problema | Solución |
|----------|----------|
| Audio en 48kHz | Resampleo a 24kHz con librosa |
| Audios muy cortos/largos | Filtros de duración configurables |
| Calidad variable | Filtros de metadatos (votos, género, edad) |
| Formato de texto | Preprocesamiento con prefijo de voz |
| Desbalanceo dialectal | Sampling estratificado por variante |
| Tokenización de audio | SNAC con 3 niveles jerárquicos |

### Proceso de Tokenización

```
Texto: "pau: Bon dia!"
  ↓
Tokenizer de Texto (Llama)
  ↓
[128000, 79, 2933, 25, 13789, 47387, 0]  (7 tokens)

Audio: 5 segundos a 24kHz
  ↓
SNAC Encoder (3 niveles)
  ↓
Nivel 1 (coarse):  375 tokens  (75 Hz)
Nivel 2 (medium):  750 tokens  (150 Hz)
Nivel 3 (fine):    1,500 tokens (300 Hz)

Total: ~2,625 tokens

Secuencia Final:
[text_tokens] + [audio_tokens_l1] + [audio_tokens_l2] + [audio_tokens_l3]
= 7 + 2,625 = 2,632 tokens
```

## Recursos Necesarios

### Hardware Mínimo (para probar)
- GPU: RTX 3090 / 4090 (24GB VRAM)
- RAM: 32GB
- Disco: 100GB
- Tiempo: ~4 horas para 500 samples

### Hardware Recomendado (para producción)
- GPU: A100 40GB o H100
- RAM: 64GB
- Disco: 500GB
- Tiempo: ~6 horas para 2,000 samples

### Uso de Recursos

| Fase | VRAM | RAM | Tiempo (500 samples) |
|------|------|-----|---------------------|
| Preparación | - | 4GB | 5-10 min |
| Tokenización | 4GB | 8GB | 30-60 min |
| Entrenamiento | 16GB | 16GB | 2-4 horas |

## Guías de Documentación

1. **[QUICKSTART.md](QUICKSTART.md)** → Empieza aquí para empezar rápido
2. **[README.md](README.md)** → Documentación completa
3. **[USAGE_CUSTOM_DATASETS.md](USAGE_CUSTOM_DATASETS.md)** → Para tus datasets específicos
4. **[TOKENIZATION_GUIDE.md](TOKENIZATION_GUIDE.md)** → Entender la tokenización

## Próximos Pasos

1. **Probar con dataset pequeño**: 100 samples para validar pipeline
2. **Escalar**: Aumentar a 500-1000 samples por dialecto
3. **Evaluar**: Revisar métricas y calidad de audio
4. **Iterar**: Ajustar hiperparámetros según resultados
5. **Producción**: Entrenar modelo final con todos los datos

## Soporte para Variantes Dialectales

El sistema soporta:

| Dialecto | Dataset | Voz | Ejemplo |
|----------|---------|-----|---------|
| Central | `xaviviro/cv_23_ca_central` | pau | `pau: Bon dia!` |
| Balear | `xaviviro/cv_23_ca_balearic` | maria | `maria: Bon dia!` |
| Valencià | `xaviviro/cv_23_ca_valencian` | carla | `carla: Bon dia!` |
| Nord | `xaviviro/cv_23_ca_northern` | montse | `montse: Bon dia!` |
| Nord-occidental | `xaviviro/cv_23_ca_northwestern` | jordi | `jordi: Bon dia!` |

## Características Principales

✅ **Soporte multi-dialectal** con voces diferenciadas
✅ **Pipeline completo** de datos a modelo entrenado
✅ **Setup automático** para RunPod
✅ **Filtros configurables** de calidad y metadatos
✅ **Tokenización jerárquica** con SNAC
✅ **Entrenamiento optimizado** con bf16 y gradient checkpointing
✅ **Logging completo** con WandB/TensorBoard
✅ **Documentación extensa** con ejemplos

## Comandos Rápidos

```bash
# Setup completo
./setup_runpod.sh

# Pipeline básico
python scripts/prepare_custom_catalan.py --datasets xaviviro/cv_23_ca_central --output_dir ./data/processed
python scripts/tokenize_dataset.py --input_dir ./data/processed --output_dir ./data/tokenized
accelerate launch scripts/train_catalan.py --config configs/config_catalan.yaml

# Monitoreo
watch -n 1 nvidia-smi
tensorboard --logdir ./logs/
```

## Contacto

Para problemas o preguntas:
- Repositorio Orpheus: https://github.com/canopyai/Orpheus-TTS
- Documentación Orpheus: https://canopylabs.ai/

---

**Creado**: 2025-11-06
**Versión**: 1.0
**Para**: Fine-tuning de Orpheus TTS en Catalán con variantes dialectales
