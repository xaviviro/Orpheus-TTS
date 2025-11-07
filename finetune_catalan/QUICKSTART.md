# Guía de Inicio Rápido - Orpheus TTS Catalán

## Para tus datasets (xaviviro/cv_23_ca_*)

### 1. Preparación Rápida (RunPod)

```bash
# En RunPod, ejecutar:
cd /workspace
git clone https://github.com/canopyai/Orpheus-TTS.git
cd Orpheus-TTS/finetune_catalan
chmod +x setup_runpod.sh
./setup_runpod.sh
```

### 2. Autenticación

```bash
huggingface-cli login  # Pegar tu token
wandb login            # Opcional: pegar tu token de WandB
```

### 3. Preparar Datos

```bash
# Opción A: Una sola variante (más rápido para probar)
python scripts/prepare_custom_catalan.py \
    --datasets xaviviro/cv_23_ca_central \
    --output_dir ./data/processed \
    --samples_per_variant 100

# Opción B: Múltiples variantes (recomendado)
python scripts/prepare_custom_catalan.py \
    --datasets \
        xaviviro/cv_23_ca_central \
        xaviviro/cv_23_ca_balearic \
        xaviviro/cv_23_ca_valencian \
    --output_dir ./data/processed \
    --samples_per_variant 500
```

### 4. Tokenizar

```bash
python scripts/tokenize_dataset.py \
    --input_dir ./data/processed \
    --output_dir ./data/tokenized \
    --device cuda
```

### 5. Configurar

```bash
nano configs/config_catalan.yaml
# Cambiar:
# TTS_dataset: "./data/tokenized"
```

### 6. Entrenar

```bash
accelerate launch scripts/train_catalan.py \
    --config configs/config_catalan.yaml
```

### 7. Monitorear

```bash
# En otra terminal
watch -n 1 nvidia-smi

# O WandB
# https://wandb.ai/tu-usuario/orpheus-catalan-tts
```

## Tiempos Estimados (RTX 4090)

| Paso | 500 samples/variante | 1000 samples/variante |
|------|---------------------|----------------------|
| Preparación | 5-10 min | 15-20 min |
| Tokenización | 30-60 min | 1-2 horas |
| Entrenamiento (3 epochs) | 2-4 horas | 6-8 horas |

## Uso de Memoria

- **Preparación**: ~4 GB RAM
- **Tokenización**: ~8 GB RAM + 4 GB VRAM
- **Entrenamiento**: ~16 GB VRAM (batch_size=2)

## Parámetros Clave

### Reducir Memoria (si OOM)

```yaml
# En config_catalan.yaml
training:
  batch_size: 1                     # Reducir
  gradient_accumulation_steps: 8    # Aumentar

advanced:
  gradient_checkpointing: true      # Activar
```

### Acelerar Entrenamiento

```yaml
training:
  batch_size: 4                     # Aumentar si tienes VRAM
  gradient_accumulation_steps: 2

data:
  preprocessing_num_workers: 8      # Más workers
```

### Mejorar Calidad

```bash
# En prepare_custom_catalan.py
--samples_per_variant 1000          # Más datos
--min_duration 2.0                  # Más selectivo
--max_duration 15.0
```

## Verificación Rápida

```python
# Verificar datos procesados
from datasets import load_from_disk
ds = load_from_disk('./data/processed')
print(f"Train: {len(ds['train'])}")
print(f"Sample: {ds['train'][0]['text']}")

# Verificar datos tokenizados
ds = load_from_disk('./data/tokenized')
print(f"Tokens: {len(ds['train'][0]['input_ids'])}")
```

## Problemas Comunes

### Dataset no encontrado
```bash
huggingface-cli login
huggingface-cli whoami
```

### OOM durante entrenamiento
```yaml
# Reducir batch_size a 1
# Activar gradient_checkpointing
```

### Tokenización muy lenta
```bash
# Usar menos samples primero
--samples_per_variant 100
```

## Estructura de Archivos

```
finetune_catalan/
├── setup_runpod.sh           ← Setup automático
├── README.md                 ← Documentación completa
├── QUICKSTART.md            ← Esta guía
├── USAGE_CUSTOM_DATASETS.md ← Guía para tus datasets
├── TOKENIZATION_GUIDE.md    ← Cómo funciona la tokenización
│
├── configs/
│   └── config_catalan.yaml  ← Configuración principal
│
├── scripts/
│   ├── prepare_custom_catalan.py    ← Para xaviviro/cv_23_ca_*
│   ├── prepare_commonvoice_catalan.py ← Para projecte-aina/*
│   ├── tokenize_dataset.py          ← Tokenizar audio+texto
│   └── train_catalan.py             ← Entrenamiento
│
└── data/                    ← Se crea automáticamente
    ├── processed/          ← Datos procesados
    └── tokenized/          ← Datos tokenizados
```

## Próximos Pasos

Después del entrenamiento:

1. **Evaluar**: Ver métricas en WandB/TensorBoard
2. **Inferencia**: Probar el modelo con nuevos textos
3. **Iterar**: Ajustar hiperparámetros y reentrenar
4. **Compartir**: Subir modelo a HuggingFace Hub

Ver [README.md](README.md) para más detalles.
