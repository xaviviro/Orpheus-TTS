#!/bin/bash

################################################################################
# Setup Script para RunPod - Orpheus TTS Catalán
################################################################################
# Este script configura el entorno completo en RunPod para entrenar
# Orpheus TTS con datasets en catalán con variantes dialectales.
#
# Uso:
#   chmod +x setup_runpod.sh
#   ./setup_runpod.sh
################################################################################

set -e  # Salir si hay errores

echo "======================================================================"
echo "   CONFIGURACIÓN DE ENTORNO RUNPOD PARA ORPHEUS TTS CATALÁN"
echo "======================================================================"

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Función para imprimir mensajes
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

################################################################################
# 1. VERIFICAR ENTORNO
################################################################################

echo ""
echo "1. Verificando entorno..."

# Verificar GPU
if command -v nvidia-smi &> /dev/null; then
    print_status "GPU detectada:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    print_error "No se detectó GPU NVIDIA"
    exit 1
fi

# Verificar espacio en disco
DISK_SPACE=$(df -h /workspace | awk 'NR==2 {print $4}')
print_status "Espacio disponible en /workspace: $DISK_SPACE"

################################################################################
# 2. ACTUALIZAR SISTEMA Y DEPENDENCIAS BASE
################################################################################

echo ""
echo "2. Actualizando sistema base..."

apt-get update -qq
apt-get install -y -qq git wget curl nano htop tmux ffmpeg libsndfile1 > /dev/null 2>&1

print_status "Dependencias del sistema instaladas"

################################################################################
# 3. CONFIGURAR PYTHON Y ENTORNO VIRTUAL
################################################################################

echo ""
echo "3. Configurando entorno Python..."

# Verificar versión de Python
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
print_status "Python version: $PYTHON_VERSION"

# Actualizar pip
pip install --upgrade pip setuptools wheel -q

print_status "Pip actualizado"

################################################################################
# 4. VERIFICAR PYTORCH (RunPod ya lo tiene instalado)
################################################################################

echo ""
echo "4. Verificando PyTorch..."

# RunPod ya tiene PyTorch instalado, solo verificamos
if python3 -c "import torch" 2>/dev/null; then
    PYTORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())")

    print_status "PyTorch ya está instalado: ${PYTORCH_VERSION}"

    if [ "$CUDA_AVAILABLE" = "True" ]; then
        CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        print_status "CUDA disponible: ${CUDA_VERSION}"
        print_status "GPU: ${GPU_NAME}"
    else
        print_warning "CUDA no disponible - entrenamiento será MUY lento"
    fi
else
    print_error "PyTorch no está instalado"
    echo "Instalando PyTorch..."

    # Solo instalar si realmente no está
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2)
    if [[ "$CUDA_VERSION" == "12."* ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
    else
        pip install torch torchvision torchaudio -q
    fi

    print_status "PyTorch instalado"
fi

################################################################################
# 5. INSTALAR DEPENDENCIAS DE ORPHEUS TTS
################################################################################

echo ""
echo "5. Instalando dependencias de Orpheus TTS..."

# Transformers y datasets
pip install transformers>=4.40.0 datasets -q
print_status "Transformers y Datasets instalados"

# Flash Attention 2 (para mejor rendimiento)
pip install flash-attn --no-build-isolation -q 2>/dev/null || {
    print_warning "Flash Attention 2 no pudo instalarse - continuando sin él"
}

# Accelerate (para entrenamiento distribuido)
pip install accelerate -q
print_status "Accelerate instalado"

# SNAC (codec de audio)
pip install snac -q
print_status "SNAC instalado"

# Orpheus TTS (opcional, para inferencia)
pip install orpheus-speech -q || {
    print_warning "orpheus-speech no disponible - se usará el modelo directamente"
}

################################################################################
# 6. INSTALAR DEPENDENCIAS DE AUDIO Y PROCESAMIENTO
################################################################################

echo ""
echo "6. Instalando dependencias de procesamiento de audio..."

pip install librosa soundfile audioread pydub -q
print_status "Librerías de audio instaladas"

################################################################################
# 7. INSTALAR HERRAMIENTAS DE ENTRENAMIENTO
################################################################################

echo ""
echo "7. Instalando herramientas de entrenamiento..."

# WandB para logging
pip install wandb -q
print_status "Weights & Biases instalado"

# TensorBoard (alternativa)
pip install tensorboard -q
print_status "TensorBoard instalado"

# TRL (para RLHF si se necesita)
pip install trl -q
print_status "TRL instalado"

# PEFT (para LoRA fine-tuning eficiente)
pip install peft -q
print_status "PEFT instalado"

################################################################################
# 8. CLONAR REPOSITORIO DE ORPHEUS TTS
################################################################################

echo ""
echo "8. Configurando repositorio..."

# Crear directorio de trabajo
WORK_DIR="/workspace/orpheus-catalan"
mkdir -p $WORK_DIR
cd $WORK_DIR

# Clonar repositorio si no existe
if [ ! -d "Orpheus-TTS" ]; then
    print_status "Clonando repositorio Orpheus TTS..."
    git clone https://github.com/canopyai/Orpheus-TTS.git
else
    print_status "Repositorio ya existe"
fi

cd Orpheus-TTS

print_status "Repositorio configurado en: $(pwd)"

################################################################################
# 9. CONFIGURAR CACHES EN /workspace
################################################################################

echo ""
echo "9. Configurando caches en /workspace..."

# Crear directorios de cache
mkdir -p /workspace/hf_cache/{hub,datasets,transformers}
mkdir -p /workspace/wandb_cache

# Configurar variables de entorno para caches
export HF_HOME=/workspace/hf_cache
export HF_DATASETS_CACHE=/workspace/hf_cache/datasets
export TRANSFORMERS_CACHE=/workspace/hf_cache/transformers
export HF_HUB_CACHE=/workspace/hf_cache/hub
export WANDB_DIR=/workspace/wandb_cache
export WANDB_CACHE_DIR=/workspace/wandb_cache

print_status "Caches configurados en /workspace:"
echo "  - HF_HOME: /workspace/hf_cache"
echo "  - HF_DATASETS_CACHE: /workspace/hf_cache/datasets"
echo "  - TRANSFORMERS_CACHE: /workspace/hf_cache/transformers"
echo "  - WANDB_DIR: /workspace/wandb_cache"

################################################################################
# 10. AUTENTICACIÓN HUGGING FACE
################################################################################

echo ""
echo "10. Autenticación Hugging Face..."

# Verificar si ya está autenticado
if [ -f ~/.huggingface/token ]; then
    print_status "Token de Hugging Face ya existe"

    # Preguntar si quiere reautenticarse
    echo ""
    read -p "¿Quieres reautenticarte? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Iniciando autenticación de Hugging Face..."
        huggingface-cli login
    else
        print_status "Usando token existente"
    fi
else
    print_warning "No se encontró token de Hugging Face"
    echo ""
    echo "Por favor, ingresa tu token de Hugging Face"
    echo "Obtén tu token en: https://huggingface.co/settings/tokens"
    echo ""

    huggingface-cli login

    if [ $? -eq 0 ]; then
        print_status "✓ Autenticación exitosa en Hugging Face"
    else
        print_error "✗ Error en la autenticación de Hugging Face"
    fi
fi

################################################################################
# 11. AUTENTICACIÓN WANDB
################################################################################

echo ""
echo "11. Autenticación Weights & Biases (WandB)..."

# Verificar si wandb está instalado
if command -v wandb &> /dev/null; then
    # Verificar si ya está autenticado
    if wandb status &> /dev/null; then
        print_status "WandB ya está autenticado"

        # Preguntar si quiere reautenticarse
        echo ""
        read -p "¿Quieres reautenticarte en WandB? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Iniciando autenticación de WandB..."
            wandb login
        else
            print_status "Usando configuración existente de WandB"
        fi
    else
        echo ""
        read -p "¿Quieres configurar WandB para logging? (Y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            print_status "Iniciando autenticación de WandB..."
            echo ""
            echo "Por favor, ingresa tu API key de WandB"
            echo "Obtén tu API key en: https://wandb.ai/authorize"
            echo ""

            wandb login

            if [ $? -eq 0 ]; then
                print_status "✓ Autenticación exitosa en WandB"
            else
                print_warning "✗ Error en la autenticación de WandB (opcional)"
            fi
        else
            print_warning "WandB no configurado (puedes hacerlo después con: wandb login)"
        fi
    fi
else
    print_warning "WandB no está instalado (se instaló con pip, puede requerir reinicio)"
fi

################################################################################
# 12. CONFIGURAR ESTRUCTURA DE DIRECTORIOS
################################################################################

echo ""
echo "12. Creando estructura de directorios..."

# Crear directorios necesarios
mkdir -p /workspace/data/{raw,processed,tokenized}
mkdir -p /workspace/checkpoints
mkdir -p /workspace/logs
mkdir -p /workspace/outputs
mkdir -p /workspace/analysis

print_status "Estructura de directorios creada:"
echo "  - /workspace/data/raw         (datasets sin procesar)"
echo "  - /workspace/data/processed   (datasets procesados)"
echo "  - /workspace/data/tokenized   (datasets tokenizados)"
echo "  - /workspace/checkpoints      (checkpoints del modelo)"
echo "  - /workspace/logs             (logs de entrenamiento)"
echo "  - /workspace/outputs          (outputs finales)"
echo "  - /workspace/analysis         (análisis de datasets)"
echo "  - /workspace/hf_cache         (cache HuggingFace)"
echo "  - /workspace/wandb_cache      (cache WandB)"

################################################################################
# 13. COPIAR SCRIPTS DE FINE-TUNING CATALÁN
################################################################################

echo ""
echo "13. Verificando scripts de fine-tuning catalán..."

if [ -d "finetune_catalan" ]; then
    print_status "Scripts de fine-tuning catalán encontrados"
else
    print_warning "Scripts de fine-tuning catalán no encontrados"
    echo "  Los scripts deben estar en: ./finetune_catalan/"
fi

################################################################################
# 14. CONFIGURAR VARIABLES DE ENTORNO PERMANENTES
################################################################################

echo ""
echo "14. Configurando variables de entorno permanentes..."

# Crear archivo de variables de entorno
cat > /workspace/.env << 'EOF'
# Configuración de entorno para Orpheus TTS Catalán

# Directorios de trabajo
export WORK_DIR=/workspace/orpheus-catalan
export DATA_DIR=/workspace/data
export CHECKPOINT_DIR=/workspace/checkpoints
export LOG_DIR=/workspace/logs
export OUTPUT_DIR=/workspace/outputs

# PyTorch optimizaciones
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# Hugging Face - TODOS los caches en /workspace
export HF_HOME=/workspace/hf_cache
export HF_DATASETS_CACHE=/workspace/hf_cache/datasets
export TRANSFORMERS_CACHE=/workspace/hf_cache/transformers
export HF_HUB_CACHE=/workspace/hf_cache/hub

# WandB - Cache en /workspace
export WANDB_DIR=/workspace/wandb_cache
export WANDB_CACHE_DIR=/workspace/wandb_cache
export WANDB_PROJECT=orpheus-catalan-tts

# Optimizaciones de rendimiento
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true

# Para evitar warnings
export TF_CPP_MIN_LOG_LEVEL=2
EOF

# Agregar al bashrc para persistencia entre sesiones
if ! grep -q "source /workspace/.env" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# Orpheus TTS Environment" >> ~/.bashrc
    echo "source /workspace/.env" >> ~/.bashrc
fi

# Cargar las variables ahora
source /workspace/.env

print_status "Variables de entorno configuradas y cargadas"
print_status "Las variables se cargarán automáticamente en nuevas sesiones"

################################################################################
# 15. CREAR SCRIPTS DE UTILIDAD
################################################################################

echo ""
echo "15. Creando scripts de utilidad..."

# Script para preparar datos
cat > /workspace/prepare_data.sh << 'EOF'
#!/bin/bash
# Script para preparar el dataset catalán

cd /workspace/orpheus-catalan/Orpheus-TTS/finetune_catalan

python scripts/prepare_commonvoice_catalan.py \
    --dataset_name projecte-aina/commonvoice_benchmark_catalan_accents \
    --output_dir /workspace/data/processed \
    --samples_per_accent 500 \
    --target_sample_rate 24000
EOF

chmod +x /workspace/prepare_data.sh
print_status "Script de preparación creado: /workspace/prepare_data.sh"

# Script para tokenizar datos
cat > /workspace/tokenize_data.sh << 'EOF'
#!/bin/bash
# Script para tokenizar el dataset

cd /workspace/orpheus-catalan/Orpheus-TTS/finetune_catalan

python scripts/tokenize_dataset.py \
    --input_dir /workspace/data/processed \
    --output_dir /workspace/data/tokenized \
    --model_name canopylabs/orpheus-tts-0.1-pretrained
EOF

chmod +x /workspace/tokenize_data.sh
print_status "Script de tokenización creado: /workspace/tokenize_data.sh"

# Script para entrenar
cat > /workspace/train.sh << 'EOF'
#!/bin/bash
# Script para iniciar el entrenamiento

cd /workspace/orpheus-catalan/Orpheus-TTS/finetune_catalan

# Configurar WandB si es necesario
# wandb login

# Entrenar con accelerate (para multi-GPU)
accelerate launch scripts/train_catalan.py \
    --config configs/config_catalan.yaml
EOF

chmod +x /workspace/train.sh
print_status "Script de entrenamiento creado: /workspace/train.sh"

# Script para monitorear GPU
cat > /workspace/monitor_gpu.sh << 'EOF'
#!/bin/bash
# Monitorear uso de GPU en tiempo real

watch -n 1 nvidia-smi
EOF

chmod +x /workspace/monitor_gpu.sh
print_status "Script de monitoreo creado: /workspace/monitor_gpu.sh"

################################################################################
# 14. CREAR JUPYTER NOTEBOOK (OPCIONAL)
################################################################################

echo ""
echo "14. Configurando Jupyter Lab (opcional)..."

pip install jupyterlab ipywidgets -q

# Crear notebook de ejemplo
mkdir -p /workspace/notebooks

cat > /workspace/notebooks/test_inference.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Test de Inferencia - Orpheus TTS Catalán\n",
    "\n",
    "Notebook para probar el modelo entrenado."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
    "import torch\n",
    "\n",
    "# Cargar modelo\n",
    "model_path = \"/workspace/checkpoints/final_model\"\n",
    "model = AutoModelForCausalLM.from_pretrained(model_path)\n",
    "tokenizer = AutoTokenizer.from_pretrained(model_path)\n",
    "\n",
    "# Mover a GPU\n",
    "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
    "model = model.to(device)\n",
    "\n",
    "print(f\"Modelo cargado en: {device}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Generar audio\n",
    "prompt = \"pau: Bon dia! Com estàs?\"\n",
    "\n",
    "inputs = tokenizer(prompt, return_tensors=\"pt\").to(device)\n",
    "outputs = model.generate(**inputs, max_length=1024)\n",
    "\n",
    "# Decodificar y guardar\n",
    "# (Aquí agregarías el código para convertir tokens a audio)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

print_status "Jupyter notebook de ejemplo creado"

################################################################################
# 15. INFORMACIÓN FINAL Y SIGUIENTES PASOS
################################################################################

echo ""
echo "======================================================================"
echo "   ✓ CONFIGURACIÓN COMPLETADA"
echo "======================================================================"
echo ""
print_status "Entorno configurado exitosamente!"
echo ""
echo "SIGUIENTES PASOS:"
echo ""
echo "1. Autenticar con Hugging Face:"
echo "   $ huggingface-cli login"
echo ""
echo "2. (Opcional) Configurar WandB:"
echo "   $ wandb login"
echo ""
echo "3. Preparar datos:"
echo "   $ /workspace/prepare_data.sh"
echo ""
echo "4. Tokenizar datos:"
echo "   $ /workspace/tokenize_data.sh"
echo ""
echo "5. Editar configuración:"
echo "   $ nano /workspace/orpheus-catalan/Orpheus-TTS/finetune_catalan/configs/config_catalan.yaml"
echo ""
echo "6. Iniciar entrenamiento:"
echo "   $ /workspace/train.sh"
echo ""
echo "7. Monitorear GPU (en otra terminal):"
echo "   $ /workspace/monitor_gpu.sh"
echo ""
echo "DIRECTORIOS IMPORTANTES:"
echo "  - Trabajo:      /workspace/orpheus-catalan"
echo "  - Datos:        /workspace/data"
echo "  - Checkpoints:  /workspace/checkpoints"
echo "  - Logs:         /workspace/logs"
echo "  - Scripts:      /workspace/*.sh"
echo ""
echo "INFORMACIÓN DEL SISTEMA:"
nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version --format=csv,noheader
echo ""
echo "======================================================================"

# Guardar información de versiones
python3 << 'PYEOF'
import sys
import torch
try:
    import transformers
    print(f"Transformers: {transformers.__version__}")
except:
    pass
try:
    import datasets
    print(f"Datasets: {datasets.__version__}")
except:
    pass
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
PYEOF

echo ""
print_status "Setup completado! Listo para entrenar Orpheus TTS en Catalán"
echo ""
