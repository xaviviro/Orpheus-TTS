#!/bin/bash
#
# Script para preparar TODOS los datasets de catalán por dialecto
# y subirlos automáticamente a HuggingFace Hub
#

set -e

# Configuración
OUTPUT_DIR="${1:-/workspace/data/processed}"
MAX_SAMPLES="${2:-50}"
HF_REPO="${3:-xaviviro/cv_23_ca_distilled}"
UPLOAD_TO_HF="${4:-yes}"  # yes/no para controlar si sube
NUM_WORKERS="${5:-$(nproc)}"  # Número de workers (default: todos los cores)

# Lista completa de tus datasets catalanes
DATASETS=(
    "xaviviro/cv_23_ca_alacanti"
    "xaviviro/cv_23_ca_central"
    "xaviviro/cv_23_ca_balear"
    "xaviviro/cv_23_ca_tortosi"
    "xaviviro/cv_23_ca_valencia"
    "xaviviro/cv_23_ca_septentrional"
    "xaviviro/cv_23_ca_nord_occidental"
)

echo "========================================"
echo "Preparación de Datasets Catalanes"
echo "========================================"
echo ""
echo "Configuración:"
echo "  Output dir: $OUTPUT_DIR"
echo "  Max samples/speaker: $MAX_SAMPLES"
echo "  HF repo: $HF_REPO"
echo "  Upload to HF: $UPLOAD_TO_HF"
echo "  Workers: $NUM_WORKERS"
echo ""
echo "Datasets a preparar:"
for ds in "${DATASETS[@]}"; do
    echo "  - $ds"
done
echo ""

# Construir argumentos para el script de Python
DATASET_ARGS=""
for ds in "${DATASETS[@]}"; do
    DATASET_ARGS="$DATASET_ARGS $ds"
done

# Ejecutar preparación por dialecto
echo "========================================"
echo "Paso 1: Preparando datos por dialecto"
echo "========================================"
echo ""

python "$(dirname "$0")/prepare_by_dialect.py" \
    --datasets $DATASET_ARGS \
    --output_dir "$OUTPUT_DIR" \
    --balance_speakers \
    --max_samples_per_speaker "$MAX_SAMPLES" \
    --num_workers "$NUM_WORKERS" \
    --hf_repo "$HF_REPO"

echo ""
echo "✓ Preparación y subida de datos completada"

echo ""
echo "========================================"
echo "✓ PROCESO COMPLETADO"
echo "========================================"
echo ""
echo "Resumen:"
echo "  - Datos preparados en: $OUTPUT_DIR"
echo "  - Dataset en HuggingFace: https://huggingface.co/datasets/$HF_REPO"
echo ""
echo "Para usar el dataset procesado:"
echo "  from datasets import load_dataset"
echo "  ds = load_dataset('$HF_REPO', 'central')  # o cualquier dialecto"
echo ""
