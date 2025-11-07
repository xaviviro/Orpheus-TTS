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
    --max_samples_per_speaker "$MAX_SAMPLES"

echo ""
echo "✓ Preparación de datos completada"

# Subir a HuggingFace Hub si está habilitado
if [ "$UPLOAD_TO_HF" = "yes" ]; then
    echo ""
    echo "========================================"
    echo "Paso 2: Subiendo a HuggingFace Hub"
    echo "========================================"
    echo ""

    # Verificar autenticación
    if ! huggingface-cli whoami &> /dev/null; then
        echo "❌ Error: No estás autenticado en HuggingFace"
        echo "   Ejecuta: huggingface-cli login"
        exit 1
    fi

    echo "✓ Autenticación verificada"

    # Crear script Python para subir el dataset
    UPLOAD_SCRIPT=$(mktemp)
    cat > "$UPLOAD_SCRIPT" << 'PYTHON_SCRIPT'
import sys
from datasets import load_from_disk, DatasetDict
from pathlib import Path

output_dir = sys.argv[1]
repo_id = sys.argv[2]

print(f"\nCargando datasets procesados desde: {output_dir}")

# Cargar todos los dialectos
datasets = {}
output_path = Path(output_dir)

for dialect_dir in output_path.glob("*"):
    if dialect_dir.is_dir() and not dialect_dir.name.startswith('.'):
        dialect_name = dialect_dir.name
        print(f"  Cargando {dialect_name}...")
        try:
            ds = load_from_disk(str(dialect_dir))
            datasets[dialect_name] = ds
        except Exception as e:
            print(f"    ⚠️ Error cargando {dialect_name}: {e}")

if not datasets:
    print("\n❌ No se encontraron datasets procesados")
    sys.exit(1)

print(f"\n✓ Cargados {len(datasets)} dialectos")

# Crear DatasetDict
dataset_dict = DatasetDict(datasets)

print(f"\nDatasets a subir:")
for name, ds in dataset_dict.items():
    print(f"  - {name}: {len(ds)} muestras")

# Subir a HuggingFace Hub
print(f"\nSubiendo a HuggingFace Hub: {repo_id}")
print("Esto puede tomar varios minutos...")

try:
    dataset_dict.push_to_hub(
        repo_id,
        private=False,  # Cambiar a True si quieres que sea privado
        commit_message=f"Upload processed Catalan dialects (max {sys.argv[3]} samples/speaker)"
    )
    print(f"\n✓ Dataset subido exitosamente a: https://huggingface.co/datasets/{repo_id}")
except Exception as e:
    print(f"\n❌ Error subiendo dataset: {e}")
    sys.exit(1)
PYTHON_SCRIPT

    # Ejecutar script de upload
    python "$UPLOAD_SCRIPT" "$OUTPUT_DIR" "$HF_REPO" "$MAX_SAMPLES"

    # Limpiar
    rm "$UPLOAD_SCRIPT"

    echo ""
    echo "✓ Upload completado"
fi

echo ""
echo "========================================"
echo "✓ PROCESO COMPLETADO"
echo "========================================"
echo ""
echo "Resumen:"
echo "  - Datos preparados en: $OUTPUT_DIR"
if [ "$UPLOAD_TO_HF" = "yes" ]; then
    echo "  - Dataset en HuggingFace: https://huggingface.co/datasets/$HF_REPO"
    echo ""
    echo "Para usar el dataset procesado:"
    echo "  from datasets import load_dataset"
    echo "  ds = load_dataset('$HF_REPO', 'central')  # o cualquier dialecto"
fi
echo ""
