# Setup Completo - Orpheus TTS Catalán en RunPod

## ✅ Características Implementadas

### 1. Instalación de Componentes PyTorch con Detección CUDA

El script `setup_runpod.sh` ahora detecta automáticamente la versión de CUDA instalada y usa el wheel compatible:

```bash
# Detección automática de CUDA
CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")

# Mapeo a wheel compatible:
# - CUDA 12.1 → cu121
# - CUDA 12.4 → cu124
# - CUDA 11.8 → cu118
# - Otros CUDA 12.x → cu121 (default)

# Instalación con wheel correcto
pip install torchvision torchaudio --index-url https://download.pytorch.org/whl/${TORCH_WHEEL}
pip install torchcodec --index-url https://download.pytorch.org/whl/${TORCH_WHEEL}  # opcional
```

**Componentes instalados:**
- ✅ `torchvision` - Para procesamiento de imágenes/video
- ✅ `torchaudio` - Para procesamiento de audio
- ✅ `torchcodec` - Codecs adicionales (opcional, con fallback si no disponible)

**NO reinstala PyTorch** - RunPod ya lo tiene instalado.

### 2. Configuración de Caches en /workspace

Todos los caches se guardan en `/workspace` para persistencia:

```bash
# HuggingFace
export HF_HOME=/workspace/hf_cache
export HF_DATASETS_CACHE=/workspace/hf_cache/datasets
export TRANSFORMERS_CACHE=/workspace/hf_cache/transformers
export HF_HUB_CACHE=/workspace/hf_cache/hub

# WandB
export WANDB_DIR=/workspace/wandb_cache
export WANDB_CACHE_DIR=/workspace/wandb_cache
export WANDB_PROJECT=orpheus-catalan-tts
```

### 3. Autenticación Automática

**HuggingFace:**
```bash
# Primera vez: pide token
huggingface-cli login

# Si ya existe token: pregunta si quieres reautenticar
¿Quieres reautenticarte? (y/N):
```

**WandB:**
```bash
# Si no está configurado: pregunta si quieres usarlo
¿Quieres configurar WandB para logging? (Y/n):

# Si ya está configurado: pregunta si quieres reautenticar
¿Quieres reautenticarte en WandB? (y/N):
```

### 4. Estructura de Directorios

Creada automáticamente en `/workspace`:

```
/workspace/
├── data/
│   ├── raw/          # Datasets sin procesar
│   ├── processed/    # Datasets procesados
│   └── tokenized/    # Datasets tokenizados
├── checkpoints/      # Checkpoints del modelo
├── logs/             # Logs de entrenamiento
├── outputs/          # Outputs finales
├── analysis/         # Análisis de datasets
├── hf_cache/         # Cache HuggingFace (persistente)
└── wandb_cache/      # Cache WandB (persistente)
```

## 🚀 Uso en RunPod

### Paso 1: Subir archivos

Sube la carpeta completa `finetune_catalan/` a tu pod de RunPod.

### Paso 2: Ejecutar setup

```bash
cd /workspace/Orpheus-TTS
chmod +x finetune_catalan/setup_runpod.sh
./finetune_catalan/setup_runpod.sh
```

El script:
1. ✅ Verifica Python 3.10+
2. ✅ Actualiza pip
3. ✅ Detecta PyTorch y CUDA instalados
4. ✅ Instala torchvision, torchaudio, torchcodec con wheel compatible
5. ✅ Instala dependencias de Orpheus TTS
6. ✅ Instala orpheus-speech y vllm
7. ✅ Instala SNAC para tokenización
8. ✅ Instala Flash Attention 2 (si es posible)
9. ✅ Instala dependencias para fine-tuning
10. ✅ Configura autenticación HuggingFace
11. ✅ Configura autenticación WandB
12. ✅ Crea estructura de directorios
13. ✅ Configura variables de entorno permanentes
14. ✅ Ejecuta validación completa

### Paso 3: Cargar variables de entorno

```bash
source /workspace/.env
```

O agrega al `~/.bashrc` para que se cargue automáticamente:

```bash
echo "source /workspace/.env" >> ~/.bashrc
```

### Paso 4: Validar setup

```bash
cd finetune_catalan
python scripts/validate_setup.py
```

Esto verifica:
- ✅ Tokenizer con vocabulario extendido (>128k tokens)
- ✅ Dependencias instaladas correctamente
- ✅ GPU y VRAM disponibles
- ✅ Caches configurados

## 📊 Workflow Completo

Una vez completado el setup:

### 1. Analizar tus datasets

```bash
python scripts/analyze_speaker_distribution.py \
  --datasets xaviviro/cv_23_ca_central xaviviro/cv_23_ca_balear \
  --output /workspace/analysis/speaker_distribution.json
```

### 2. Preparar datos por dialecto

```bash
python scripts/prepare_by_dialect.py \
  --datasets xaviviro/cv_23_ca_central xaviviro/cv_23_ca_balear xaviviro/cv_23_ca_valencia \
  --output_dir /workspace/data/processed \
  --balance_speakers \
  --max_samples_per_speaker 50
```

### 3. Tokenizar datos

```bash
python scripts/tokenize_dataset.py \
  --input_dir /workspace/data/processed \
  --output_dir /workspace/data/tokenized \
  --model_name canopylabs/orpheus-tts-0.1-pretrained
```

### 4. Entrenar

```bash
python scripts/train_catalan.py \
  --config configs/config_catalan.yaml \
  --data_dir /workspace/data/tokenized \
  --output_dir /workspace/checkpoints/catalan_dialectal
```

### 5. Generar audio (inferencia)

```bash
python scripts/inference_with_orpheus_package.py \
  --model_path /workspace/checkpoints/catalan_dialectal/final_model \
  --text "Bon dia! Com estàs avui?" \
  --dialect central \
  --output /workspace/outputs/test_central.wav
```

## 🔧 Características Técnicas

### Detección CUDA Inteligente

El script detecta la versión exacta de CUDA:

```bash
# Verifica PyTorch instalado
if python3 -c "import torch" 2>/dev/null; then
    CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")

    # Mapea a wheel compatible
    if [[ "$CUDA_VERSION" == "12.1"* ]]; then
        TORCH_WHEEL="cu121"
    elif [[ "$CUDA_VERSION" == "12.4"* ]]; then
        TORCH_WHEEL="cu124"
    # ... etc
fi
```

### Manejo de Errores

- **torchcodec no disponible**: Marca como opcional, continúa sin error
- **Flash Attention falla**: Marca como opcional, continúa sin error
- **Sin GPU**: Advierte pero permite continuar (CPU mode)
- **PyTorch no instalado**: Instala versión completa automáticamente

### Variables de Entorno Persistentes

El archivo `/workspace/.env` contiene:

```bash
# Directorios de trabajo
export WORK_DIR=/workspace/orpheus-catalan
export DATA_DIR=/workspace/data
export CHECKPOINT_DIR=/workspace/checkpoints
export LOG_DIR=/workspace/logs
export OUTPUT_DIR=/workspace/outputs

# PyTorch optimizaciones
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# HuggingFace - TODOS en /workspace
export HF_HOME=/workspace/hf_cache
export HF_DATASETS_CACHE=/workspace/hf_cache/datasets
export TRANSFORMERS_CACHE=/workspace/hf_cache/transformers

# WandB
export WANDB_DIR=/workspace/wandb_cache
export WANDB_CACHE_DIR=/workspace/wandb_cache
export WANDB_PROJECT=orpheus-catalan-tts

# Optimizaciones
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true
```

## 🎯 Estrategia para tus Datasets

Como tus datasets tienen **muchos hablantes con pocos audios cada uno**, la estrategia recomendada es:

### Fase 1: Pretraining Dialectal (ahora)

Entrenar un modelo por dialecto usando TODOS los hablantes:

```bash
# Central: 300+ hablantes → modelo aprende catalán central
# Balear: 200+ hablantes → modelo aprende balear
# Valencià: 300+ hablantes → modelo aprende valencià
```

**Ventajas:**
- ✅ Usas TODOS tus datos
- ✅ El modelo aprende características fonéticas del dialecto
- ✅ No necesitas 300+ muestras por hablante individual

### Fase 2: Voice Cloning (después, opcional)

Para voces específicas de producción:

```bash
# Seleccionar 2-3 voces con mejor calidad
# Usar voice cloning con 3-10 segundos de referencia
```

**Ver más detalles en:**
- [ESTRATEGIA_RECOMENDADA.md](ESTRATEGIA_RECOMENDADA.md)
- [EJEMPLO_TUS_DATASETS.md](EJEMPLO_TUS_DATASETS.md)

## 📚 Documentación Completa

- **[INDEX.md](INDEX.md)** - Índice de toda la documentación
- **[QUICKSTART.md](QUICKSTART.md)** - Guía rápida (10 minutos)
- **[README.md](README.md)** - Documentación completa
- **[MEJORES_PRACTICAS_OFICIAL.md](MEJORES_PRACTICAS_OFICIAL.md)** - Guía oficial de Canopy Labs
- **[TOKENIZATION_GUIDE.md](TOKENIZATION_GUIDE.md)** - Cómo funciona SNAC

## ✅ Checklist Pre-Entrenamiento

Antes de empezar a entrenar, verifica:

- [ ] Setup completado: `./setup_runpod.sh`
- [ ] Variables de entorno cargadas: `source /workspace/.env`
- [ ] Validación exitosa: `python scripts/validate_setup.py`
- [ ] HuggingFace autenticado: `huggingface-cli whoami`
- [ ] WandB autenticado: `wandb status` (opcional)
- [ ] GPU disponible: `nvidia-smi`
- [ ] Datasets analizados: `python scripts/analyze_speaker_distribution.py`
- [ ] Configuración revisada: `configs/config_catalan.yaml`

## 🆘 Resolución de Problemas

### Error: "Token fuera de rango"
**Causa**: Tokenizer sin vocabulario extendido de SNAC
**Solución**: Usar `canopylabs/orpheus-tts-0.1-pretrained` como modelo base

### Error: "CUDA out of memory"
**Causa**: Secuencias muy largas o batch_size alto
**Solución**:
- Reducir `batch_size` en config
- Filtrar audios >15s en preparación de datos
- Activar `gradient_checkpointing`

### Error: "torchcodec no disponible"
**Causa**: No hay wheel para tu versión de CUDA
**Solución**: Es opcional, el script continúa sin error

### Error: "Flash Attention failed"
**Causa**: Compilación requiere CUDA toolkit
**Solución**: Es opcional, entrenarás sin Flash Attention (más lento pero funciona)

### Error: "std::bad_alloc" durante análisis de datasets
**Causa**: Dataset muy grande (1M+ muestras) consume toda la RAM
**Solución**: El script `analyze_speaker_distribution.py` ahora usa streaming mode automáticamente. Simplemente vuelve a ejecutar el script - procesará en modo streaming sin cargar todo en memoria.

## 🎉 ¡Listo!

Ahora tienes un entorno completo para entrenar Orpheus TTS en catalán con variantes dialectales.

**Siguiente paso**: Ejecuta el setup y sigue el [QUICKSTART.md](QUICKSTART.md)

---

**Versión**: 1.0
**Fecha**: 2025-11-07
**Compatible con**: RunPod, CUDA 11.8+, PyTorch 2.0+
