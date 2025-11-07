# Guía de Tokenización para Orpheus TTS

## Introducción

Orpheus TTS es un modelo de texto-a-voz basado en un LLM (Llama-3B) que trata la generación de audio como un problema de modelado de lenguaje. Para lograr esto, tanto el texto como el audio deben convertirse en secuencias de tokens discretos que el modelo pueda procesar.

## Arquitectura de Tokenización

### 1. Visión General

```
Entrada de Texto + Audio → Tokenización → Secuencia Unificada → Modelo LLM → Tokens de Audio → Audio Sintetizado
```

Orpheus utiliza **dos tokenizadores diferentes**:

1. **Tokenizador de Texto**: Tokenizador estándar de LLM (basado en BPE/SentencePiece)
2. **Tokenizador de Audio**: SNAC (Self-Normalizing Audio Codec)

### 2. Tokenización de Texto

El texto se tokeniza usando el tokenizador del modelo base (Llama):

```python
# Ejemplo
text = "pau: Bon dia! Com estàs?"
tokenizer = AutoTokenizer.from_pretrained("canopylabs/orpheus-tts-0.1-pretrained")
text_tokens = tokenizer(text)

# Resultado (conceptual)
# [128000, 79, 2933, 25, 13789, 47387, 0, 1219, 1826, 18366, 30]
```

**Características**:
- Vocabulario: ~128k tokens (Llama base)
- Formato especial: `{nombre_voz}: {texto}`
- Soporte multilingüe (depende del tokenizador base)
- Tokens especiales: `<s>`, `</s>`, `<pad>`, etc.

### 3. Tokenización de Audio con SNAC

SNAC (Self-Normalizing Audio Codec) es el componente clave para convertir audio en tokens discretos.

#### 3.1. ¿Qué es SNAC?

SNAC es un codec de audio neural que:
- Comprime audio a representaciones discretas
- Opera en múltiples escalas temporales (multi-resolución)
- Optimizado para voz humana
- Trabaja a 24kHz de frecuencia de muestreo

#### 3.2. Arquitectura Jerárquica de SNAC

SNAC tokeniza el audio en **3 niveles jerárquicos**:

```
Audio (24kHz) → Encoder → 3 Codebooks Jerárquicos → Tokens Discretos
```

**Nivel 1 (Coarse)**: Captura información prosódica de bajo nivel
- Frecuencia: ~75 Hz
- Captura: Pitch, energía, ritmo general

**Nivel 2 (Medium)**: Captura características fonéticas intermedias
- Frecuencia: ~150 Hz
- Captura: Transiciones fonéticas, formantes

**Nivel 3 (Fine)**: Captura detalles acústicos finos
- Frecuencia: ~300 Hz
- Captura: Timbre, textura vocal, detalles espectrales

#### 3.3. Proceso de Tokenización de Audio

```python
from snac import SNAC
import torch

# 1. Cargar modelo SNAC
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")

# 2. Preparar audio (debe estar en 24kHz)
audio_tensor = torch.from_numpy(audio_array).float()
audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # [batch, channels, samples]

# 3. Codificar (tokenizar)
with torch.no_grad():
    codes = snac_model.encode(audio_tensor)

# codes es una lista de 3 tensores (uno por nivel jerárquico)
# codes[0]: tokens de nivel 1 (coarse)  - shape: [batch, time_steps_1]
# codes[1]: tokens de nivel 2 (medium)  - shape: [batch, time_steps_2]
# codes[2]: tokens de nivel 3 (fine)    - shape: [batch, time_steps_3]
```

**Ejemplo de salida**:

Para 1 segundo de audio (24,000 muestras):
```python
codes[0].shape  # [1, 75]   - 75 tokens (nivel coarse)
codes[1].shape  # [1, 150]  - 150 tokens (nivel medium)
codes[2].shape  # [1, 300]  - 300 tokens (nivel fine)

# Total: 525 tokens por segundo de audio
```

#### 3.4. Rango de Tokens de Audio

Cada nivel tiene su propio codebook:
- Nivel 1: Tokens en rango [128264, 128264 + vocab_size_1]
- Nivel 2: Tokens en rango [nivel1_end, nivel1_end + vocab_size_2]
- Nivel 3: Tokens en rango [nivel2_end, nivel2_end + vocab_size_3]

Típicamente, cada codebook tiene 4096 códigos, dando un total de ~12k tokens adicionales.

### 4. Combinación de Tokens: Texto + Audio

El formato final de una secuencia de entrenamiento es:

```
[text_tokens] [separator] [audio_tokens_level1] [audio_tokens_level2] [audio_tokens_level3]
```

**Ejemplo visual**:

```python
# Texto: "pau: Hola"
text_tokens = [128000, 79, 2933, 25, 47387]  # 5 tokens

# Audio: 2 segundos
audio_tokens_l1 = [128500, 128501, 128503, ...]  # 150 tokens (2s * 75Hz)
audio_tokens_l2 = [129500, 129505, 129510, ...]  # 300 tokens (2s * 150Hz)
audio_tokens_l3 = [130500, 130502, 130508, ...]  # 600 tokens (2s * 300Hz)

# Secuencia final
input_ids = text_tokens + audio_tokens_l1 + audio_tokens_l2 + audio_tokens_l3
# Total: 5 + 150 + 300 + 600 = 1055 tokens para "Hola" (2s audio)
```

### 5. Longitud de Secuencias

**Cálculo de longitud de tokens**:

Para una frase con:
- Texto: N palabras (~1.3 tokens/palabra en promedio)
- Audio: T segundos

```
tokens_text ≈ N × 1.3
tokens_audio ≈ T × 525  (75 + 150 + 300 por segundo)
tokens_total = tokens_text + tokens_audio
```

**Ejemplos**:

| Texto | Audio | Tokens Texto | Tokens Audio | Total |
|-------|-------|--------------|--------------|-------|
| 10 palabras | 5s | ~13 | 2,625 | ~2,638 |
| 20 palabras | 10s | ~26 | 5,250 | ~5,276 |
| 50 palabras | 25s | ~65 | 13,125 | ~13,190 |

**Límite de contexto**:
- Orpheus entrena con secuencias de hasta **8,192 tokens**
- Esto permite aproximadamente **15 segundos de audio** por ejemplo

### 6. Proceso de Tokenización en el Pipeline

#### Paso 1: Preparación del Dataset

```python
# Input
{
    'audio': {'array': [...], 'sampling_rate': 24000},
    'text': 'pau: Bon dia!',
    'voice_name': 'pau',
    'variant': 'central'
}
```

#### Paso 2: Tokenización

```python
# Tokenizar texto
text_tokens = tokenizer(example['text'], truncation=True, max_length=4096)

# Tokenizar audio con SNAC
audio_tensor = torch.from_numpy(example['audio']['array']).float()
audio_codes = snac_model.encode(audio_tensor)

# Concatenar niveles de audio
audio_tokens = np.concatenate([
    audio_codes[0].numpy().flatten(),  # Nivel coarse
    audio_codes[1].numpy().flatten(),  # Nivel medium
    audio_codes[2].numpy().flatten(),  # Nivel fine
])
```

#### Paso 3: Combinar y Crear Labels

```python
# Combinar tokens
input_ids = np.concatenate([
    text_tokens['input_ids'].flatten(),
    audio_tokens
])

# Para causal language modeling, labels = input_ids
labels = input_ids.copy()

# Output final
{
    'input_ids': input_ids,
    'labels': labels,
    'attention_mask': [1] * len(input_ids)
}
```

### 7. Decodificación (Inferencia)

Durante la inferencia, el proceso se invierte:

```python
# 1. Modelo genera tokens
output_tokens = model.generate(
    input_ids=text_tokens,
    max_length=8192
)

# 2. Separar tokens de texto y audio
audio_tokens = output_tokens[len(text_tokens):]

# 3. Dividir en 3 niveles jerárquicos
level1_len = duration_seconds * 75
level2_len = duration_seconds * 150
level3_len = duration_seconds * 300

level1_tokens = audio_tokens[:level1_len]
level2_tokens = audio_tokens[level1_len:level1_len+level2_len]
level3_tokens = audio_tokens[level1_len+level2_len:]

# 4. Decodificar con SNAC
audio_codes = [
    torch.tensor(level1_tokens).unsqueeze(0),
    torch.tensor(level2_tokens).unsqueeze(0),
    torch.tensor(level3_tokens).unsqueeze(0)
]

audio_output = snac_model.decode(audio_codes)

# 5. Guardar como audio
import soundfile as sf
sf.write('output.wav', audio_output.numpy(), 24000)
```

### 8. Ventajas del Sistema de Tokenización

1. **Unificación**: Trata texto y audio como secuencias de tokens
2. **Jerarquía**: Captura múltiples escalas temporales del audio
3. **Eficiencia**: Compresión significativa (24,000 samples/s → 525 tokens/s)
4. **Calidad**: Reconstrucción de alta fidelidad del audio
5. **Flexibilidad**: Permite modelos LLM estándar sin arquitecturas especiales

### 9. Consideraciones Prácticas

#### Memoria

Para un batch de tamaño 2 con secuencias de 8k tokens:
```
Tokens: 2 × 8,192 = 16,384 tokens
Embeddings (bf16): 16,384 × 3,072 × 2 bytes ≈ 100 MB
Activaciones: ~2-4 GB (con gradient checkpointing)
Parámetros: ~6 GB (modelo 3B en bf16)
Total: ~10-15 GB VRAM mínimo
```

#### Velocidad de Procesamiento

En una RTX 4090:
- Tokenización de audio: ~50-100 ejemplos/segundo
- Entrenamiento: ~0.5-1.0 segundos/iteración (batch_size=2)
- Inferencia: ~200ms para 5 segundos de audio (con streaming)

#### Optimizaciones

1. **Caching de audio tokenizado**: Tokenizar una vez, usar múltiples veces
2. **Batching eficiente**: Agrupar por longitud similar
3. **Gradient checkpointing**: Reducir memoria durante entrenamiento
4. **Mixed precision (bf16)**: Acelerar entrenamiento sin perder calidad

## Resumen

Orpheus TTS utiliza un sistema de tokenización dual:

1. **Texto**: Tokenización estándar LLM (BPE/SentencePiece)
2. **Audio**: SNAC con 3 niveles jerárquicos (coarse, medium, fine)

Esto permite que un LLM estándar genere audio de alta calidad tratando la síntesis de voz como un problema de modelado de lenguaje. La clave está en la representación jerárquica de SNAC que captura desde la prosodia hasta los detalles acústicos finos.

---

**Referencias**:
- [SNAC Paper](https://arxiv.org/abs/2310.06825)
- [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS)
- [SNAC Model](https://huggingface.co/hubertsiuzdak/snac_24khz)
