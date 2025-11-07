# 📚 Índice de Documentación - Orpheus TTS Catalán

## 🚀 Por donde empezar

1. **[SETUP_COMPLETO.md](SETUP_COMPLETO.md)** - ⭐ Setup completo en RunPod (EMPIEZA AQUÍ)
2. **[RESUMEN.md](RESUMEN.md)** - Visión general del proyecto
3. **[QUICKSTART.md](QUICKSTART.md)** - Guía rápida para empezar en 10 minutos
4. **[README.md](README.md)** - Documentación completa y detallada

## 📖 Guías Específicas

### Para tus Datasets
- **[EJEMPLO_TUS_DATASETS.md](EJEMPLO_TUS_DATASETS.md)** - ⭐ Ejemplo completo paso a paso con tus datasets
- **[ESTRATEGIA_RECOMENDADA.md](ESTRATEGIA_RECOMENDADA.md)** - ⭐ Estrategia por dialecto + voice cloning (RECOMENDADO)
- **[USAGE_CUSTOM_DATASETS.md](USAGE_CUSTOM_DATASETS.md)** - Guía general para datasets personalizados
- **[VOICE_CLONING_STRATEGY.md](VOICE_CLONING_STRATEGY.md)** - Comparación de estrategias de voice cloning

### Información Técnica
- **[TOKENIZATION_GUIDE.md](TOKENIZATION_GUIDE.md)** - Cómo funciona la tokenización (SNAC + texto)
- **[MEJORES_PRACTICAS_OFICIAL.md](MEJORES_PRACTICAS_OFICIAL.md)** - ⭐ Guía oficial de Canopy Labs

## 🔧 Archivos de Configuración

### Setup
- **[setup_runpod.sh](setup_runpod.sh)** - Script de configuración automática para RunPod
- **[requirements.txt](requirements.txt)** - Dependencias de Python

### Configuración de Entrenamiento
- **[configs/config_catalan.yaml](configs/config_catalan.yaml)** - Configuración principal

## 💻 Scripts

### Preparación de Datos
- **[scripts/analyze_speaker_distribution.py](scripts/analyze_speaker_distribution.py)** - Analizar distribución de hablantes
- **[scripts/prepare_by_dialect.py](scripts/prepare_by_dialect.py)** - ⭐ Preparar por dialecto (RECOMENDADO)
- **[scripts/prepare_custom_catalan.py](scripts/prepare_custom_catalan.py)** - Para tus datasets (xaviviro/cv_23_ca_*)
- **[scripts/prepare_commonvoice_catalan.py](scripts/prepare_commonvoice_catalan.py)** - Para datasets públicos (projecte-aina)

### Procesamiento y Entrenamiento
- **[scripts/tokenize_dataset.py](scripts/tokenize_dataset.py)** - Tokenización de audio y texto con SNAC
- **[scripts/train_catalan.py](scripts/train_catalan.py)** - Script de entrenamiento con Transformers
- **[scripts/validate_setup.py](scripts/validate_setup.py)** - Validar configuración antes de entrenar

### Inferencia
- **[scripts/inference_with_orpheus_package.py](scripts/inference_with_orpheus_package.py)** - ⭐ Inferencia completa (RECOMENDADO)
- **[scripts/inference_dialectal.py](scripts/inference_dialectal.py)** - Inferencia básica con especificación de dialecto

## 📊 Estadísticas del Proyecto

- **1,254 líneas de código Python** (scripts)
- **1,686 líneas de documentación** (markdown)
- **4 scripts principales**
- **5 documentos guía**
- **1 script de setup automático**

## 🎯 Flujo de Trabajo Recomendado

```
1. Leer RESUMEN.md (5 min)
   ↓
2. Ejecutar setup_runpod.sh (10 min)
   ↓
3. Seguir QUICKSTART.md (1 hora)
   ↓
4. Preparar datos con prepare_custom_catalan.py
   ↓
5. Tokenizar con tokenize_dataset.py
   ↓
6. Entrenar con train_catalan.py
   ↓
7. Evaluar y iterar
```

## 🔍 Búsqueda Rápida

### "¿Cómo empiezo?"
→ [QUICKSTART.md](QUICKSTART.md)

### "¿Cómo uso mis datasets?"
→ [USAGE_CUSTOM_DATASETS.md](USAGE_CUSTOM_DATASETS.md)

### "¿Cómo funciona la tokenización?"
→ [TOKENIZATION_GUIDE.md](TOKENIZATION_GUIDE.md)

### "¿Qué hace cada script?"
→ [RESUMEN.md](RESUMEN.md#archivos-clave)

### "¿Configuración para RunPod?"
→ [setup_runpod.sh](setup_runpod.sh)

### "¿Problemas comunes?"
→ [README.md](README.md#resolución-de-problemas)

## 📦 Estructura de Carpetas

```
finetune_catalan/
├── 📄 INDEX.md                      ← Estás aquí
├── 📄 RESUMEN.md                    ← Visión general
├── 📄 QUICKSTART.md                 ← Inicio rápido
├── 📄 README.md                     ← Documentación completa
├── 📄 USAGE_CUSTOM_DATASETS.md      ← Guía para tus datasets
├── 📄 TOKENIZATION_GUIDE.md         ← Guía técnica de tokenización
│
├── 🔧 setup_runpod.sh              ← Setup automático
├── 📄 requirements.txt              ← Dependencias
├── 📄 .gitignore                    ← Git ignore
│
├── 📁 configs/
│   └── config_catalan.yaml         ← Configuración
│
├── 📁 scripts/
│   ├── prepare_custom_catalan.py          ← Preparar tus datasets
│   ├── prepare_commonvoice_catalan.py     ← Preparar datasets públicos
│   ├── tokenize_dataset.py                ← Tokenizar
│   └── train_catalan.py                   ← Entrenar
│
└── 📁 data/                        ← (se crea automáticamente)
```

## 🎓 Nivel de Complejidad

| Documento | Nivel | Tiempo de Lectura |
|-----------|-------|-------------------|
| RESUMEN.md | 🟢 Principiante | 10 min |
| QUICKSTART.md | 🟢 Principiante | 15 min |
| USAGE_CUSTOM_DATASETS.md | 🟡 Intermedio | 20 min |
| README.md | 🟡 Intermedio | 30 min |
| TOKENIZATION_GUIDE.md | 🔴 Avanzado | 25 min |

## 💡 Tips

- **Primera vez**: Lee RESUMEN.md y luego QUICKSTART.md
- **Ya tienes experiencia**: Ve directo a USAGE_CUSTOM_DATASETS.md
- **Quieres entender a fondo**: Lee TOKENIZATION_GUIDE.md
- **Problemas**: Busca en la sección de resolución de problemas del README.md
- **RunPod**: Ejecuta setup_runpod.sh y sigue el output

## 🆘 Ayuda

Si encuentras problemas:
1. Revisa [README.md - Resolución de Problemas](README.md#resolución-de-problemas)
2. Verifica que seguiste todos los pasos del [QUICKSTART.md](QUICKSTART.md)
3. Consulta la documentación oficial de [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS)

---

**Total**: 2,940 líneas de código y documentación | **Versión**: 1.0 | **Fecha**: 2025-11-06
