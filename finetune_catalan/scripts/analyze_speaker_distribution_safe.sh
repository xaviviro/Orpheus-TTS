#!/bin/bash
#
# Wrapper para ejecutar analyze_speaker_distribution.py sin problemas de memoria
# Deshabilita torchcodec temporalmente para evitar std::bad_alloc
#

set -e

echo "========================================="
echo "Análisis de Distribución de Hablantes"
echo "========================================="
echo ""
echo "Este script deshabilita torchcodec temporalmente para evitar"
echo "problemas de memoria con datasets muy grandes (1M+ muestras)."
echo ""

# Verificar si torchcodec está instalado
TORCHCODEC_INSTALLED=0
if python3 -c "import torchcodec" 2>/dev/null; then
    TORCHCODEC_INSTALLED=1
    echo "✓ torchcodec detectado - será desinstalado temporalmente"
fi

# Desinstalar torchcodec temporalmente
if [ $TORCHCODEC_INSTALLED -eq 1 ]; then
    echo ""
    echo "Desinstalando torchcodec temporalmente..."
    pip uninstall -y torchcodec > /dev/null 2>&1
    echo "✓ torchcodec desinstalado"
fi

# Ejecutar el script de análisis
echo ""
echo "Ejecutando análisis..."
echo ""

python3 "$(dirname "$0")/analyze_speaker_distribution.py" "$@"

ANALYSIS_EXIT_CODE=$?

# Reinstalar torchcodec si estaba instalado
if [ $TORCHCODEC_INSTALLED -eq 1 ]; then
    echo ""
    echo "Reinstalando torchcodec..."

    # Detectar versión de CUDA
    if python3 -c "import torch" 2>/dev/null; then
        CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)

        if [[ "$CUDA_VERSION" == "12.1"* ]]; then
            TORCH_WHEEL="cu121"
        elif [[ "$CUDA_VERSION" == "12.4"* ]]; then
            TORCH_WHEEL="cu124"
        elif [[ "$CUDA_VERSION" == "11.8"* ]]; then
            TORCH_WHEEL="cu118"
        else
            TORCH_WHEEL="cu121"
        fi

        echo "Reinstalando torchcodec 0.6.0 para ${TORCH_WHEEL}..."
        pip install torchcodec==0.6.0 --index-url https://download.pytorch.org/whl/${TORCH_WHEEL} -q 2>/dev/null && \
            echo "✓ torchcodec 0.6.0 reinstalado" || \
            echo "⚠ No se pudo reinstalar torchcodec 0.6.0 (no crítico)"
    fi
fi

echo ""
if [ $ANALYSIS_EXIT_CODE -eq 0 ]; then
    echo "========================================="
    echo "✓ Análisis completado con éxito"
    echo "========================================="
else
    echo "========================================="
    echo "✗ Error durante el análisis"
    echo "========================================="
fi

exit $ANALYSIS_EXIT_CODE
