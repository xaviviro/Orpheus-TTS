#!/bin/bash
#
# Script para analizar TODOS los datasets de catalán automáticamente
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
echo "Análisis de Datasets Catalanes"
echo "========================================"
echo ""
echo "Datasets a analizar:"
for ds in "${DATASETS[@]}"; do
    echo "  - $ds"
done
echo ""

# Construir argumentos para el script de Python
DATASET_ARGS=""
for ds in "${DATASETS[@]}"; do
    DATASET_ARGS="$DATASET_ARGS $ds"
done

# Ejecutar el análisis paralelo
python "$(dirname "$0")/analyze_speaker_distribution_parallel.py" \
    --datasets $DATASET_ARGS \
    --output_dir "${1:-./analysis}" \
    --num_workers "${2:-$(nproc)}" \
    --batch_size "${3:-1000}"

echo ""
echo "========================================"
echo "✓ Análisis completado"
echo "========================================"
echo ""
echo "Resultados guardados en: ${1:-./analysis}"
