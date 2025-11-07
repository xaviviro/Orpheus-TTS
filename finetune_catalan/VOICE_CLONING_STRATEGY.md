# Estrategia para Datasets con Muchos Hablantes

## Tu Situación

**Dataset**: Muchos hablantes, pocos audios por hablante (típico de Common Voice)

**Problema con fine-tuning tradicional**:
- Se necesitan ~300 audios/voz para voces fijas
- Tus datasets tienen muchos hablantes con pocos audios cada uno
- Fine-tuning tradicional no aprovecha esta diversidad

## Soluciones

### ✅ Opción 1: Zero-Shot Voice Cloning (RECOMENDADO)

El modelo **preentrenado** de Orpheus ya tiene capacidad de zero-shot voice cloning.

#### ¿Cómo funciona?

En lugar de entrenar voces fijas, usa el modelo preentrenado con **prompt conditioning**:

```python
# No necesitas fine-tuning!
# El modelo preentrenado ya puede clonar voces

from orpheus_tts import OrpheusModel

model = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-pretrained")

# Proporcionar 1-3 ejemplos de la voz objetivo
reference_audio = "path/to/reference_speaker.wav"
reference_text = "Text que dice el audio de referencia"

# Generar con esa voz
output = model.generate_speech(
    prompt=f"Speaker: {nuevo_texto}",
    voice_reference=reference_audio,
    reference_text=reference_text
)
```

**Ventajas**:
- ✓ No necesitas entrenar
- ✓ Funciona con 1-3 audios por hablante
- ✓ Aprovecha toda la diversidad de tu dataset
- ✓ Más rápido

**Desventajas**:
- ✗ Necesitas proporcionar audio de referencia en cada inferencia
- ✗ Menos control sobre voces específicas

### ✅ Opción 2: Fine-tuning Multi-Speaker (HÍBRIDO)

Entrenar el modelo para **mejorar** la capacidad de voice cloning en catalán.

**Estrategia**:
1. Usar TODOS los hablantes (aunque tengan pocos audios)
2. Entrenar con formato de condicionamiento (in-context learning)
3. El modelo aprende a copiar voces del catalán mejor

**Formato de entrenamiento**:
```
[Audio1 de hablante X] [Texto1] [Audio2 de hablante X] [Texto2] [Audio3 de hablante X que queremos generar]
```

El modelo aprende: "Si te doy ejemplos de una voz, genera audio en esa voz"

**Ventajas**:
- ✓ Mejora el zero-shot cloning específicamente para catalán
- ✓ Aprende características dialectales
- ✓ Usa todos tus datos
- ✓ Mantiene flexibilidad

### ✅ Opción 3: Híbrido con Voces Base (EQUILIBRADO)

Combinar ambas estrategias:

1. **Voces fijas para dialectos** (100-300 muestras de hablantes con más datos)
   - 3-5 voces representativas por dialecto
   - Seleccionar hablantes con más audios

2. **Zero-shot para el resto**
   - Usar el modelo entrenado con voice cloning
   - Proporcionar referencias en inferencia

## Implementación Recomendada

Basándome en tu caso, te recomiendo **Opción 2 + 3 combinadas**:

### Fase 1: Análisis de tus Datos

```bash
# Nuevo script que voy a crear
python scripts/analyze_speaker_distribution.py \
    --datasets xaviviro/cv_23_ca_central \
    --output_dir ./analysis/
```

Este script te dirá:
- Cuántos hablantes tienes
- Distribución de audios por hablante
- Qué hablantes tienen suficientes datos para voces fijas
- Estadísticas por dialecto

### Fase 2: Estrategia Mixta

#### A) Voces Fijas (Hablantes con más datos)

```bash
# Seleccionar top hablantes por dialecto
python scripts/prepare_fixed_voices.py \
    --datasets xaviviro/cv_23_ca_central \
    --min_samples_per_speaker 100 \
    --speakers_per_dialect 2 \
    --output_dir ./data/fixed_voices/
```

#### B) Datos para Multi-Speaker Training

```bash
# Preparar TODO el dataset para entrenamiento multi-speaker
python scripts/prepare_multispeaker_catalan.py \
    --datasets xaviviro/cv_23_ca_central \
    --min_samples_per_speaker 5 \
    --max_samples_per_speaker 50 \
    --output_dir ./data/multispeaker/
```

### Fase 3: Entrenamiento Multi-Speaker

```yaml
# config_multispeaker.yaml
training:
  style: "multi-speaker"  # Nuevo modo
  context_length: 3       # Número de ejemplos de referencia

data:
  format: "in-context"    # Formato de condicionamiento
  group_by_speaker: true  # Agrupar por hablante
```

## Formato de Datos Multi-Speaker

### Formato Tradicional (Voces Fijas)
```
{
  'text': 'pau: Bon dia',
  'audio': [audio_array]
}
```

### Formato Multi-Speaker (Voice Cloning)
```
{
  'context': [
    {'text': 'Hola', 'audio': [audio1_speaker_X]},
    {'text': 'Com estàs', 'audio': [audio2_speaker_X]}
  ],
  'target': {
    'text': 'Bon dia',
    'audio': [audio3_speaker_X]
  },
  'speaker_id': 'speaker_X',
  'dialect': 'central'
}
```

El modelo aprende: "Dado contexto de voz X, genera target en voz X"

## Comparación

| Aspecto | Voces Fijas | Multi-Speaker | Zero-Shot (Pretrained) |
|---------|-------------|---------------|------------------------|
| Audios necesarios | 300+/voz | 5+/hablante | 1-3/referencia |
| Entrenamiento | Sí | Sí | No |
| Flexibilidad | Baja | Alta | Máxima |
| Calidad | Muy alta | Alta | Media-Alta |
| Uso de datos | Parcial | Total | N/A |
| Mejor para | Producción | Tu caso | Pruebas rápidas |

## Recomendación Final para Tu Caso

### Plan A (Rápido): Usar Modelo Preentrenado con Zero-Shot

No hagas fine-tuning inicial. Prueba directamente:

```python
from transformers import AutoModelForCausalLM
import torch

# Cargar modelo preentrenado
model = AutoModelForCausalLM.from_pretrained(
    "canopylabs/orpheus-tts-0.1-pretrained"
)

# Usar con conditioning (ver notebook que voy a crear)
```

**Ventaja**: Ves resultados en catalán inmediatamente sin entrenar

### Plan B (Mejor): Fine-tuning Multi-Speaker

1. Analizar distribución de hablantes
2. Preparar datos en formato multi-speaker
3. Entrenar para mejorar zero-shot en catalán
4. Seleccionar top 5-10 voces para fijar (opcional)

**Ventaja**: Mejora significativa en catalán manteniendo flexibilidad

### Plan C (Híbrido): Combinación

1. Identificar 2-3 hablantes con 100+ audios por dialecto
2. Fine-tuning tradicional para esas voces (voces de "producción")
3. Resto del dataset para multi-speaker training
4. Resultado: Voces de alta calidad + capacidad de zero-shot

## Próximos Pasos

Voy a crear:

1. ✅ Script de análisis de distribución de hablantes
2. ✅ Script de preparación multi-speaker
3. ✅ Config de entrenamiento multi-speaker
4. ✅ Notebook de zero-shot inference
5. ✅ Documentación actualizada

¿Qué prefieres? ¿Empezar con zero-shot directo o preparar entrenamiento multi-speaker?
