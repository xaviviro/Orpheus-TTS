# Estrategia Recomendada: Entrenar por Dialecto + Voice Clone

## Tu Situación

**Problema**: Datasets extensos con muchos hablantes pero pocos audios por hablante
- Dataset: `xaviviro/cv_23_ca_*` (Common Voice 23)
- Muchos hablantes diferentes
- Pocos audios de cada hablante (típicamente 5-50)

**Solución**: Entrenar voces por DIALECTO + Voice Cloning posterior

## La Estrategia en 3 Pasos

### Paso 1: Entrenar por Variante Dialectal

En lugar de entrenar voces por hablante individual, entrenar **una voz por dialecto**:

```
Dialecto Central → Voz "central" (agregando todos los hablantes centrales)
Dialecto Balear → Voz "balear" (agregando todos los hablantes baleares)
Dialecto Valencià → Voz "valencia" (agregando todos los hablantes valencianos)
```

**Formato de entrenamiento**:
```
"central: Bon dia! Com estàs?"
"balear: Bon dia! Com estàs?"
"valencia: Bon dia! Com estàs?"
```

**Resultado**: El modelo aprende las características dialectales (pronunciación, entonación, vocabulario específico).

### Paso 2: Voice Cloning para Uniformar

Después del entrenamiento, usar **zero-shot voice cloning** para adaptar a voces específicas:

```python
# 1. Cargar modelo entrenado en dialectos
model = load_model("tu_modelo_dialectos_catalan")

# 2. Proporcionar audio de referencia del hablante deseado
reference_audio = "audio_muestra_hablante.wav"
reference_text = "Hola, aquest és un exemple"

# 3. Generar con voice cloning
output = model.generate(
    text="Bon dia! Com estàs?",
    dialect="central",           # Usa dialecte correcto
    voice_reference=reference_audio,  # Clona esta voz
    reference_text=reference_text
)
```

**Resultado**: Habla en catalán central (correcto) con la voz del hablante específico.

### Paso 3: Usar en Producción

```python
# Para cada dialecto + voz específica que necesites:

# Ejemplo 1: Central con voz masculina
generate(text="...", dialect="central", voice_ref="speaker_male_central.wav")

# Ejemplo 2: Valencià con voz femenina
generate(text="...", dialect="valencia", voice_ref="speaker_female_valencia.wav")

# Ejemplo 3: Balear con voz específica
generate(text="...", dialect="balear", voice_ref="maria_balear.wav")
```

## Ventajas de esta Estrategia

### ✅ Aprovecha TODO tu Dataset
- No descartas hablantes con pocos audios
- Usas miles de muestras en lugar de cientos
- Mayor diversidad de voces en entrenamiento

### ✅ Aprende Dialectos Correctamente
- El modelo aprende pronunciación dialectal real
- Vocabulario y expresiones específicas de cada variante
- Entonación característica de cada región

### ✅ Flexibilidad con Voice Cloning
- Puedes clonar cualquier voz después
- No estás limitado a 5 voces predefinidas
- Adaptas a cada uso específico

### ✅ Escalable
- Funciona con 10 o 10,000 hablantes
- No necesitas reentrenar para nueva voz
- Más económico en tiempo y recursos

## Implementación Práctica

### 1. Análisis (Opcional pero Recomendado)

```bash
# Ver distribución de hablantes en tus datasets
python scripts/analyze_speaker_distribution.py \
    --datasets xaviviro/cv_23_ca_central xaviviro/cv_23_ca_balearic \
    --output_dir ./analysis/
```

### 2. Preparar Datos por Dialecto

```bash
# Preparar datasets agrupando por dialecto
python scripts/prepare_by_dialect.py \
    --datasets \
        xaviviro/cv_23_ca_central \
        xaviviro/cv_23_ca_balearic \
        xaviviro/cv_23_ca_valencian \
    --output_dir ./data/processed_by_dialect \
    --min_samples_per_speaker 3 \
    --max_samples_per_speaker 50 \
    --balance_speakers \
    --save_speaker_metadata
```

**Parámetros clave**:
- `--min_samples_per_speaker 3`: Incluir hablantes con al menos 3 audios
- `--max_samples_per_speaker 50`: Limitar a 50 por hablante (balanceo)
- `--balance_speakers`: Evitar que algunos hablantes dominen
- `--save_speaker_metadata`: Guardar info para voice cloning posterior

**Resultado esperado**:
```
Central: 1,500 muestras de 200 hablantes → voz "central"
Balear: 1,200 muestras de 150 hablantes → voz "balear"
València: 1,800 muestras de 250 hablantes → voz "valencia"
Total: 4,500 muestras de 600 hablantes únicos
```

### 3. Tokenizar

```bash
python scripts/tokenize_dataset.py \
    --input_dir ./data/processed_by_dialect \
    --output_dir ./data/tokenized_by_dialect \
    --device cuda
```

### 4. Configurar Entrenamiento

```yaml
# configs/config_dialect_training.yaml
TTS_dataset: "./data/tokenized_by_dialect"

data:
  # Voces por dialecto (no por hablante)
  voice_mapping:
    central: "central"
    balearic: "balear"
    valencian: "valencia"

training:
  epochs: 3
  batch_size: 2
  learning_rate: 5.0e-5
```

### 5. Entrenar

```bash
accelerate launch scripts/train_catalan.py \
    --config configs/config_dialect_training.yaml
```

**Tiempo estimado**: 4-6 horas en RTX 4090 para 4,500 muestras

### 6. Voice Cloning en Inferencia

Después del entrenamiento, usa el modelo con voice cloning:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Cargar tu modelo entrenado
model = AutoModelForCausalLM.from_pretrained("./checkpoints/final_model")
tokenizer = AutoTokenizer.from_pretrained("./checkpoints/final_model")

# Voice cloning workflow (ver notebook detallado)
# 1. Cargar audio de referencia
# 2. Generar con conditioning
# 3. Decodificar a audio
```

## Comparación con Otras Estrategias

| Estrategia | Audios Necesarios | Usa Todo Dataset | Flexibilidad | Calidad Dialecto |
|------------|-------------------|------------------|--------------|------------------|
| **Voces Fijas por Hablante** | 300+/hablante | ❌ No (solo top) | ⭐ Baja | ⭐⭐ Media |
| **Multi-Speaker Genérico** | 5+/hablante | ✅ Sí | ⭐⭐⭐ Alta | ⭐⭐ Media |
| **Por Dialecto + Voice Clone** | 3+/hablante | ✅ Sí | ⭐⭐⭐ Alta | ⭐⭐⭐ Alta |
| **Zero-Shot Pretrained** | 1-3/referencia | N/A | ⭐⭐⭐ Máxima | ⭐ Baja (inglés) |

## Ejemplo Completo de Uso

### Entrenar

```bash
# 1. Preparar
python scripts/prepare_by_dialect.py \
    --datasets xaviviro/cv_23_ca_central \
    --output_dir ./data/dialect_central \
    --balance_speakers

# 2. Tokenizar
python scripts/tokenize_dataset.py \
    --input_dir ./data/dialect_central \
    --output_dir ./data/dialect_central_tokenized

# 3. Entrenar
accelerate launch scripts/train_catalan.py \
    --config configs/config_dialect_training.yaml
```

### Inferencia con Voice Cloning

```python
# Supongamos que tienes el modelo entrenado

# Caso 1: Voz masculina en catalán central
audio_ref_male = load_audio("references/male_central_speaker.wav")
text = "Bon dia! Avui fa molt bon dia a Barcelona."

output = generate_with_cloning(
    model=model,
    text=text,
    dialect="central",
    voice_reference=audio_ref_male
)
save_audio(output, "output_male_central.wav")

# Caso 2: Voz femenina en valencià
audio_ref_female = load_audio("references/female_valencia_speaker.wav")
text = "Bon dia! Hui fa molt bon dia a València."

output = generate_with_cloning(
    model=model,
    text=text,
    dialect="valencia",
    voice_reference=audio_ref_female
)
save_audio(output, "output_female_valencia.wav")
```

## Resultados Esperados

### Después del Entrenamiento Obtendrás

1. **Modelo que habla catalán dialectal correcto**
   - Pronunciación adecuada de cada variante
   - Vocabulario específico (ej: "hui" vs "avui")
   - Entonación característica

2. **Capacidad de voice cloning**
   - Proporciona 3-5 segundos de audio de referencia
   - El modelo clona la voz manteniendo el dialecto correcto
   - Funciona con cualquier hablante (no solo los del training)

3. **Flexibilidad máxima**
   - No estás atado a 5 voces predefinidas
   - Puedes adaptar a cada cliente/proyecto
   - Escalable a nuevos dialectos

## Limitaciones y Consideraciones

### ⚠️ Limitaciones

1. **Necesitas audio de referencia en inferencia**
   - Cada generación requiere proporcionar audio del hablante objetivo
   - ~3-10 segundos de audio de referencia recomendado

2. **Calidad de voice cloning**
   - Depende de la calidad del audio de referencia
   - Mejor con audios limpios y claros
   - Puede requerir ajuste fino para voces muy específicas

3. **Complejidad en producción**
   - Sistema más complejo que voces fijas
   - Requiere gestionar referencias de audio
   - Latencia ligeramente mayor

### 💡 Mejoras Futuras

1. **Voice embeddings precomputados**
   - Calcular embeddings de voces comunes una vez
   - Reutilizar sin procesar audio cada vez
   - Reduce latencia

2. **Fine-tuning de algunas voces específicas**
   - Para clientes/usos frecuentes
   - Híbrido: dialectos + 2-3 voces fijas top

3. **Adapters por dialecto**
   - Usar LoRA/adapters específicos por dialecto
   - Cambiar rápido entre variantes
   - Menor huella de memoria

## Resumen: ¿Por qué esta Estrategia?

### Para tu Caso Específico

✅ **Datasets extensos**: Aprovechas todos los hablantes
✅ **Pocos audios/hablante**: 3-50 es suficiente por hablante
✅ **Múltiples dialectos**: Aprendes cada uno correctamente
✅ **Flexibilidad**: Voice cloning para cualquier voz después
✅ **Escalable**: Funciona con 100 o 10,000 hablantes
✅ **Económico**: Un solo modelo para todos los dialectos

### El Flow Completo

```
Muchos hablantes (5-50 audios cada uno)
          ↓
Agrupar por DIALECTO
          ↓
Entrenar 1 voz por dialecto
          ↓
Modelo que habla cada dialecto correctamente
          ↓
Voice cloning para voces específicas
          ↓
Resultado: Dialecto correcto + Voz personalizada
```

## Siguientes Pasos

1. ✅ Analizar tus datasets (opcional):
   ```bash
   python scripts/analyze_speaker_distribution.py --datasets xaviviro/cv_23_ca_central
   ```

2. ✅ Preparar por dialecto:
   ```bash
   python scripts/prepare_by_dialect.py --datasets xaviviro/cv_23_ca_*
   ```

3. ✅ Entrenar modelo dialectal

4. ✅ Implementar voice cloning en inferencia

---

**Esta es la estrategia MÁS ADECUADA para tus datasets de Common Voice con muchos hablantes y pocos audios cada uno.**
