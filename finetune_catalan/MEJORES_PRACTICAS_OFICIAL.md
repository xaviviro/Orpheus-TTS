# Mejores Prácticas Oficiales para Entrenar Orpheus TTS

Basado en la guía oficial de Canopy Labs y discusiones de la comunidad.

## Fuentes Oficiales

- **Guía oficial**: https://canopylabs.ai/releases/orpheus_can_speak_any_language#training
- **GitHub Discussion**: https://github.com/canopyai/Orpheus-TTS/discussions/123
- **README**: https://github.com/canopyai/Orpheus-TTS

## Conceptos Clave según Canopy Labs

### 1. Pretraining vs Fine-tuning

**Pretraining** (para enseñar un idioma/dialecto):
- Usa datos de **múltiples hablantes** (hundreds to thousands)
- Calidad puede ser menor, cantidad es importante
- Objetivo: El modelo aprende las características del idioma/dialecto
- Escala recomendada: **100+ horas de audio multi-speaker**

**Fine-tuning** (para voces específicas):
- Usa datos de **uno o pocos hablantes**
- Alta calidad es crítica
- Objetivo: Copiar el estilo específico de voz
- Escala recomendada: **300+ ejemplos por voz**

### 2. Para tu Caso (Catalán con Dialectos)

Según la guía oficial, tu estrategia de **entrenar por dialecto** es correcta:

```
┌────────────────────────────────────────────────────────────┐
│ FASE 1: PRETRAINING DIALECTAL (lo que harás)              │
├────────────────────────────────────────────────────────────┤
│ • Usar TODOS los hablantes de cada dialecto               │
│ • Central: 300+ hablantes → modelo aprende catalán central│
│ • Balear: 200+ hablantes → modelo aprende balear          │
│ • Valencià: 300+ hablantes → modelo aprende valencià      │
│                                                            │
│ Objetivo: Enseñar al modelo las características fonéticas │
│ de cada dialecto usando diversidad de hablantes           │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ FASE 2: FINE-TUNING (opcional, después)                   │
├────────────────────────────────────────────────────────────┤
│ • Seleccionar 2-3 hablantes específicos con 300+ audios   │
│ • Fine-tune para voces de "producción" de alta calidad    │
│                                                            │
│ O usar voice cloning como planeado                        │
└────────────────────────────────────────────────────────────┘
```

## Requisitos Técnicos Importantes

### 1. Tokenizer con Vocabulario Extendido

⚠️ **IMPORTANTE**: El tokenizer debe soportar tokens de SNAC (>128k)

```python
# Verificar tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("canopylabs/orpheus-tts-0.1-pretrained")
vocab_size = len(tokenizer)

print(f"Vocab size: {vocab_size}")
# Debe ser: 128000 + (7 × 4096) + 20 = 156,692 tokens

# Verificación
assert vocab_size > 128000, "El tokenizer NO tiene tokens SNAC extendidos"
```

**Si usas modelo multilingüe**, puede faltar el vocabulario extendido (bug reportado).

**Solución**: Usar el modelo base en inglés `canopylabs/orpheus-tts-0.1-pretrained` que tiene el tokenizer correcto.

### 2. Formato de Dataset

Según la guía oficial y ejemplos:

```python
# Formato para fine-tuning (una voz)
{
    'input_ids': [tokens_de_texto + tokens_de_audio],
    'labels': [mismo_que_input_ids],  # Causal LM
}

# Para multi-speaker (dialectos)
{
    'input_ids': [tokens_de_prefijo + tokens_de_texto + tokens_de_audio],
    'labels': [mismo_que_input_ids],
}
```

El prefijo (ej: "central:") se tokeniza como parte del texto.

### 3. Hiperparámetros Recomendados

Basado en los configs oficiales y experiencia de la comunidad:

```yaml
# Para fine-tuning dialectal (multi-speaker)
training:
  epochs: 1-3
  batch_size: 1-2  # Según VRAM
  gradient_accumulation_steps: 4-8
  learning_rate: 5e-5  # Default de Orpheus
  warmup_steps: 500
  max_grad_norm: 1.0

  # Precisión
  bf16: true  # Recomendado para A100/H100
  fp16: false

  # Optimización
  optimizer: "adamw_torch"
  lr_scheduler_type: "cosine"

  # Regularización
  weight_decay: 0.01
```

### 4. Longitud de Secuencia

```yaml
data:
  max_length: 8192  # Máximo usado en pretraining oficial
```

**Cálculo**:
- ~15 segundos de audio ≈ 7,875 tokens de audio
- + texto ≈ 100-200 tokens
- Total: ~8,000 tokens

**Para audios más largos**: Truncar o dividir en chunks.

## Validaciones Importantes

### 1. Verificar Tokenizer Antes de Entrenar

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("canopylabs/orpheus-tts-0.1-pretrained")

# Test 1: Vocabulary size
print(f"Vocab size: {len(tokenizer)}")
assert len(tokenizer) > 128000, "Tokenizer sin SNAC tokens"

# Test 2: Tokenizar ejemplo
text = "central: Bon dia"
tokens = tokenizer(text)
print(f"Tokens: {tokens['input_ids'][:10]}...")

# Test 3: Verificar token especiales
print(f"PAD token: {tokenizer.pad_token_id}")
print(f"EOS token: {tokenizer.eos_token_id}")
```

### 2. Verificar Dataset Procesado

```python
from datasets import load_from_disk

ds = load_from_disk("./data/tokenized")

# Verificar un ejemplo
example = ds['train'][0]

print(f"Input IDs shape: {len(example['input_ids'])}")
print(f"Labels shape: {len(example['labels'])}")
print(f"Match: {example['input_ids'] == example['labels']}")  # Debe ser True

# Verificar rango de tokens
max_token = max(example['input_ids'])
print(f"Max token ID: {max_token}")
assert max_token < len(tokenizer), "Token fuera de vocabulario"
```

### 3. Verificar Audio Tokenizado

```python
# Los tokens de SNAC deben estar en el rango correcto
# SNAC tokens empiezan después de 128263

audio_tokens = [t for t in example['input_ids'] if t > 128263]
print(f"Audio tokens: {len(audio_tokens)}")
print(f"Audio token range: {min(audio_tokens)} - {max(audio_tokens)}")

# Verificar que hay tokens de audio
assert len(audio_tokens) > 0, "No hay tokens de audio en el ejemplo"
```

## Mejores Prácticas de Dataset

### 1. Diversidad de Hablantes

```python
# Para pretraining dialectal
min_speakers = 100  # Mínimo recomendado
ideal_speakers = 300+  # Ideal

# Balanceo
max_samples_per_speaker = 50  # Evitar dominancia
min_samples_per_speaker = 5   # Evitar ruido
```

### 2. Calidad de Audio

```python
# Filtros recomendados
min_duration = 1.0   # segundos
max_duration = 15.0  # segundos (para caber en 8192 tokens)

target_sample_rate = 24000  # SNAC requiere 24kHz
```

### 3. Distribución Train/Val

```python
train_split = 0.9   # 90% training
val_split = 0.1     # 10% validation

# Asegurar diversidad en validation
# (no todos los ejemplos de un solo hablante)
```

## Errores Comunes a Evitar

### ❌ Error 1: Tokenizer sin SNAC tokens

**Síntoma**: Error durante training sobre tokens fuera de rango

**Causa**: Usar modelo multilingüe con tokenizer incompleto

**Solución**: Usar `canopylabs/orpheus-tts-0.1-pretrained` como base

### ❌ Error 2: Secuencias demasiado largas

**Síntoma**: OOM durante training, incluso con batch_size=1

**Causa**: Audios >15s generan secuencias >8192 tokens

**Solución**: Filtrar por `max_duration` en preparación de datos

### ❌ Error 3: Pocos hablantes

**Síntoma**: El modelo sobreajusta a pocas voces

**Causa**: Dataset con <50 hablantes diferentes

**Solución**:
- Incluir más hablantes (incluso con pocos audios cada uno)
- O hacer fine-tuning en lugar de pretraining

### ❌ Error 4: Desbalanceo extremo

**Síntoma**: El modelo genera solo en el estilo de algunos hablantes

**Causa**: Algunos hablantes tienen 200+ muestras, otros solo 5

**Solución**: Usar `--balance_speakers --max_samples_per_speaker 50`

### ❌ Error 5: Formato de prompt incorrecto

**Síntoma**: El modelo no aprende a distinguir dialectos

**Causa**: Formato inconsistente (ej: "central: text" vs "Central: text")

**Solución**: Usar siempre el mismo formato (lowercase, mismo separador)

## Optimizaciones Avanzadas

### 1. Gradient Checkpointing

Para reducir uso de VRAM:

```yaml
advanced:
  gradient_checkpointing: true
```

Reduce VRAM ~40%, aumenta tiempo ~20%

### 2. Flash Attention 2

Para acelerar training:

```bash
pip install flash-attn --no-build-isolation
```

```yaml
model_config:
  attn_implementation: "flash_attention_2"
```

Acelera training ~2x en A100/H100

### 3. DeepSpeed (para multi-GPU)

```bash
pip install deepspeed
```

```yaml
training:
  deepspeed: "configs/ds_config.json"
```

### 4. LoRA para Fine-tuning Eficiente

```yaml
advanced:
  use_lora: true
  lora_config:
    r: 16
    lora_alpha: 32
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
    lora_dropout: 0.05
```

Reduce parámetros entrenables ~90%

## Resumen: Checklist Pre-Training

Antes de empezar el entrenamiento, verifica:

- [ ] Tokenizer tiene >128k tokens (SNAC extended vocab)
- [ ] Dataset tiene 100+ hablantes diferentes
- [ ] Muestras balanceadas (max 50 por speaker)
- [ ] Audio resampleado a 24kHz
- [ ] Duración: 1-15 segundos por muestra
- [ ] Formato de prompt consistente (ej: "central: ...")
- [ ] Split train/val correcto (90/10)
- [ ] Flash Attention instalado (opcional pero recomendado)
- [ ] Wandb/TensorBoard configurado para logging
- [ ] Config de training con lr=5e-5, bf16=true

## Recursos Adicionales

- **Guía oficial completa**: https://canopylabs.ai/releases/orpheus_can_speak_any_language#training
- **Issues reportados**: https://github.com/canopyai/Orpheus-TTS/issues
- **Discussiones**: https://github.com/canopyai/Orpheus-TTS/discussions/123
- **Colab notebooks**: Ver README del repositorio

---

**Última actualización**: Basado en información de abril 2025 (release multilingüe)
