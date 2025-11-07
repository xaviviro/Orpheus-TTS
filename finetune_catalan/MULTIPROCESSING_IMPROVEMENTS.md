# Mejoras de Multiprocessing en prepare_by_dialect.py

## Cambios Implementados

### 1. Multiprocessing para Máximo Rendimiento

El script `prepare_by_dialect.py` ahora utiliza **multiprocessing** para aprovechar todos los cores disponibles de la máquina durante el procesamiento de ejemplos.

**Mejoras clave:**

- **Auto-detección de cores**: Por defecto usa todos los cores disponibles (`cpu_count()`)
- **Batch processing**: Divide los datos en batches para procesamiento eficiente
- **Progress tracking**: Mantiene barra de progreso con `tqdm`
- **Parámetro configurable**: `--num_workers` para controlar número de workers

### 2. Corrección de Error de Audio Encoding

**Problema original:**
```
AttributeError: 'AudioEncoder' object has no attribute 'to_file_like'
```

**Solución:**
- Cambiado de `cast_column()` después de crear el dataset a crear directamente con `Dataset.from_dict()` especificando la columna audio
- Arrays de audio convertidos explícitamente a `float32` para compatibilidad
- Estructura de datos optimizada para evitar encoding prematuro

### 3. Estructura de Procesamiento

```python
# Función de batch processing (ejecutada en paralelo)
def process_batch(batch, target_sr, dialect_name):
    """
    Procesa un batch de ejemplos en paralelo.
    Esta función se ejecuta en múltiples procesos workers.
    """
    results = []
    for example in batch:
        processed = process_example(example, target_sr, dialect_name)
        if processed is not None:
            results.append(processed)
    return results
```

**Flow de procesamiento:**
1. Dataset → Lista completa
2. Dividir en batches (batch_size = total / (num_workers * 4))
3. Procesar batches en paralelo con Pool
4. Combinar resultados
5. Crear Dataset final con Audio feature

## Uso

### Opción 1: Con prepare_by_dialect.py directamente

```bash
python scripts/prepare_by_dialect.py \
    --datasets xaviviro/cv_23_ca_central xaviviro/cv_23_ca_balear \
    --output_dir /workspace/data/processed \
    --balance_speakers \
    --max_samples_per_speaker 50 \
    --num_workers 64  # Especificar número de workers
```

### Opción 2: Con prepare_all_catalan_datasets.sh (RECOMENDADO)

```bash
# Uso básico (detecta cores automáticamente)
bash scripts/prepare_all_catalan_datasets.sh

# Con parámetros personalizados
bash scripts/prepare_all_catalan_datasets.sh \
    /workspace/data/processed \  # OUTPUT_DIR
    50 \                          # MAX_SAMPLES_PER_SPEAKER
    xaviviro/cv_23_ca_distilled \ # HF_REPO
    yes \                         # UPLOAD_TO_HF
    64                            # NUM_WORKERS
```

**Parámetros del wrapper:**
1. `OUTPUT_DIR` (default: `/workspace/data/processed`)
2. `MAX_SAMPLES` (default: `50`)
3. `HF_REPO` (default: `xaviviro/cv_23_ca_distilled`)
4. `UPLOAD_TO_HF` (default: `yes`)
5. `NUM_WORKERS` (default: `$(nproc)` - todos los cores)

## Rendimiento Esperado

**Antes (secuencial):**
- 1 core procesando ejemplos
- ~100-200 ejemplos/segundo

**Después (paralelo con 64 cores):**
- 64 cores procesando en paralelo
- ~6,000-12,000 ejemplos/segundo (estimado)
- **Speedup esperado: 30-60x más rápido**

## Ventajas del Nuevo Sistema

1. **Máximo aprovechamiento de CPU**: Usa todos los cores disponibles
2. **Escalable**: Funciona igual de bien con 4, 16, 64+ cores
3. **Monitoreable**: Barra de progreso en tiempo real
4. **Configurable**: Control total sobre número de workers
5. **Robusto**: Maneja errores por ejemplo sin crashear todo el batch
6. **Compatible**: Mismo formato de salida que versión anterior

## Datasets Procesados

Los 7 datasets catalanes procesados automáticamente:

1. `xaviviro/cv_23_ca_alacanti` (Alacantí)
2. `xaviviro/cv_23_ca_central` (Central)
3. `xaviviro/cv_23_ca_balear` (Balear)
4. `xaviviro/cv_23_ca_tortosi` (Tortosí)
5. `xaviviro/cv_23_ca_valencia` (Valencià)
6. `xaviviro/cv_23_ca_septentrional` (Septentrional)
7. `xaviviro/cv_23_ca_nord_occidental` (Nord-occidental)

Todos se suben automáticamente a `xaviviro/cv_23_ca_distilled` en HuggingFace Hub.

## Notas Técnicas

- **Batch size**: Se calcula automáticamente como `total_ejemplos / (num_workers * 4)` para balance óptimo
- **Memory safety**: Cada worker procesa su batch independientemente
- **Type safety**: Arrays convertidos a `float32` explícitamente
- **Error handling**: Errores en ejemplos individuales no detienen el procesamiento

## Ejemplo de Output

```
Procesando ejemplos...
Usando 64 workers para procesamiento paralelo
Batch size: 1250
Total batches: 256

Procesando central: 100%|████████████| 256/256 [02:30<00:00, 1.70batch/s]

Ejemplos procesados exitosamente: 320000
✅ Dataset procesado: 320000 ejemplos

Estadísticas finales:
  Hablantes únicos: 6400
  Promedio muestras/hablante: 50.0
  Duración total: 266.67 horas
```
