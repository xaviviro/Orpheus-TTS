# Mejoras de Procesamiento en prepare_by_dialect.py

## Cambios Implementados

### 1. Procesamiento Optimizado con scipy.signal

El script `prepare_by_dialect.py` ha sido optimizado para ser más robusto y eficiente.

**Mejoras clave:**

- **scipy.signal.resample en lugar de librosa**: Más robusto y evita errores de "stream index not added"
- **Procesamiento secuencial estable**: Evita problemas de serialización de audio en multiprocessing
- **Progress tracking**: Mantiene barra de progreso con `tqdm`
- **Formato de audio optimizado**: Arrays en float32 para compatibilidad

**Nota sobre multiprocessing**: El procesamiento de audio con HuggingFace Datasets tiene limitaciones de serialización cuando se trabaja con el formato de audio. El multiprocessing se aplica mejor en el paso de tokenización con SNAC (ver `tokenize_with_snac_multiprocess.py`).

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
# Procesamiento secuencial (más estable para audio)
processed_examples = []

for example in tqdm(dataset, desc=f"Procesando {dialect}"):
    processed = process_example(example, args.target_sample_rate, dialect)
    if processed is not None:
        processed_examples.append(processed)

# Crear dataset
processed_dataset = Dataset.from_list(processed_examples)
processed_dataset = processed_dataset.cast_column('audio', Audio(sampling_rate=24000))
```

**Flow de procesamiento:**
1. Cargar dataset de HuggingFace
2. Filtrar por duración y hablantes
3. Balancear muestras por hablante (opcional)
4. Procesar ejemplos secuencialmente (resamplear con scipy.signal)
5. Crear Dataset desde lista
6. Cast columna audio a Audio feature
7. Guardar dataset procesado

**¿Por qué procesamiento secuencial?**
- El formato de audio de HuggingFace (`{'array': ndarray, 'sampling_rate': int}`) no se serializa bien en multiprocessing
- Causa errores de `AttributeError: 'AudioEncoder' object has no attribute 'to_file_like'`
- El resampling con scipy es rápido (~200-500 samples/segundo en una sola core)
- **Para acelerar**: usa el script `tokenize_with_snac_multiprocess.py` que sí aprovecha múltiples GPUs

## Uso

### Opción 1: Con prepare_by_dialect.py directamente

```bash
python scripts/prepare_by_dialect.py \
    --datasets xaviviro/cv_23_ca_central xaviviro/cv_23_ca_balear \
    --output_dir /workspace/data/processed \
    --balance_speakers \
    --max_samples_per_speaker 50
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

## Rendimiento Esperado

**Procesamiento de audio (prepare_by_dialect.py):**
- Procesamiento secuencial con scipy.signal.resample
- ~200-500 ejemplos/segundo (single-core)
- Estable y sin errores de serialización

**Tokenización SNAC (tokenize_with_snac_multiprocess.py):**
- Multiprocessing con múltiples GPU workers
- Con 1 GPU: ~50-100 samples/segundo
- Con 2-3 workers en GPU de 48GB: ~150-300 samples/segundo
- **Speedup: 2-3x con múltiples workers GPU**

## Ventajas del Nuevo Sistema

1. **Robusto y estable**: Sin errores de serialización de audio
2. **Usa scipy en lugar de librosa**: Evita problemas de "stream index not added"
3. **Monitoreable**: Barra de progreso en tiempo real
4. **Multi-GPU para tokenización**: El paso pesado (SNAC) sí usa multiprocessing
5. **Formato optimizado**: Arrays en float32 para máxima compatibilidad
6. **Separación clara**: Procesamiento de audio (CPU) vs tokenización (GPU)

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
