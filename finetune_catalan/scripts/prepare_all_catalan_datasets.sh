#!/bin/bash
#
# Script para preparar TODOS los datasets de catalán por dialecto
#

set -e

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
python "$(dirname "$0")/prepare_by_dialect.py" \
    --datasets $DATASET_ARGS \
    --output_dir "${1:-/workspace/data/processed}" \
    --balance_speakers \
    --max_samples_per_speaker "${2:-50}"

echo ""
echo "========================================"
echo "✓ Preparación completada"
echo "========================================"
echo ""
echo "Datos preparados en: ${1:-/workspace/data/processed}"
