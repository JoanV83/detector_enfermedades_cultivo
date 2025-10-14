# Detección de Enfermedades en Cultivos a través de Clasificación de Imágenes con Deep Learning

## Universidad Autónoma de Occidente
### Desarrollo de Proyectos de Inteligencia Artificial 2025-02

**Equipo de Trabajo:**
- J. A. Velásquez Vélez
- E. V. Zapata Cardona  
- M. A. Saavedra Hurtado
- N. A. Velasco Castellanos

---

## 📋 **ÍNDICE**

1. [Introducción y Problemática](#introducción-y-problemática)
2. [Objetivos del Proyecto](#objetivos-del-proyecto)
3. [Metodología Ágil Aplicada](#metodología-ágil-aplicada)
4. [Fases de Desarrollo](#fases-de-desarrollo)
5. [Arquitectura del Sistema](#arquitectura-del-sistema)
6. [Pipeline ML Implementado](#pipeline-ml-implementado)
7. [Testing y Calidad](#testing-y-calidad)
8. [Despliegue y Demo](#despliegue-y-demo)
9. [Resultados y Conclusiones](#resultados-y-conclusiones)

---

## **INTRODUCCIÓN Y PROBLEMÁTICA**

### Problemática Identificada
- Los métodos actuales para detectar enfermedades en cultivos son **costosos** y dependen de **especialistas** y **equipos especializados**
- Existe una **brecha tecnológica** en la capacidad de ejecutar modelos de deep learning en dispositivos móviles
- Los agricultores necesitan herramientas **accesibles** y de **bajo costo** para diagnóstico temprano

### Propuesta de Solución
- Sistema de clasificación de imágenes usando **Vision Transformer (ViT)** preentrenado
- Implementación en **dispositivos móviles** para detección en tiempo real
- Uso del **PlantVillage Dataset** con más de 50,000 imágenes etiquetadas

---

## **OBJETIVOS DEL PROYECTO**

### Objetivo General
Desarrollar un sistema de clasificación de imágenes utilizando el PlantVillage Dataset y un modelo ViT preentrenado para la detección de enfermedades en cultivos.

### Objetivos Específicos Logrados
✅ **Fine-tuning del modelo ViT** con el PlantVillage Dataset  
✅ **Aplicación de técnicas de data augmentation**  
✅ **Evaluación con métricas** (matriz de confusión, F1-macro)  
✅ **Exportación a formatos móviles** (ONNX/TorchScript)  
✅ **Demo interactiva** con Streamlit

---

## **METODOLOGÍA ÁGIL APLICADA**

### Framework: CRISP-DM Adaptado con Principios Ágiles

#### **Iteraciones de 1 Semana**
Aplicamos sprints cortos de 1 semana para garantizar entregas incrementales y feedback continuo.

#### **Fases Implementadas:**

**1. Comprensión del Negocio** *(Sprint 0)*
- Definición clara de objetivos
- Identificación de stakeholders (agricultores, técnicos)
- Métricas de éxito establecidas

**2. Comprensión de los Datos** *(Sprint 1)*
- Análisis exploratorio del PlantVillage Dataset
- Verificación de calidad de imágenes y etiquetas
- Identificación de 38 clases de enfermedades

**3. Preparación de los Datos** *(Sprint 1-2)*
- Preprocesamiento automatizado
- Split estratificado (70% train, 15% val, 15% test)
- Data augmentation implementada

**4. Modelado** *(Sprint 2-3)*
- Fine-tuning iterativo del modelo ViT
- Experimentación con hiperparámetros
- Validación continua

**5. Evaluación** *(Sprint 3)*
- Métricas de evaluación implementadas
- Análisis de rendimiento por clase
- Validación con datos de test

**6. Despliegue** *(Sprint 4)*
- Exportación a formatos móviles
- Demo interactiva con Streamlit
- Documentación completa

---

## 🏗️ **FASES DE DESARROLLO**

### **SPRINT 1: Fundamentos y Datos**
**Duración:** Semana 1  
**Entregables:**
```
Configuración del entorno de desarrollo
Descarga y análisis del PlantVillage Dataset
Pipeline de preprocesamiento implementado
Split estratificado de datos realizado
Configuración inicial del modelo ViT base
```

**Evidencias de Metodología Ágil:**
- Estructura modular del código en `src/plant_disease/`
- Configuraciones externalizadas en `configs/`
- Testing automatizado desde el inicio

### **SPRINT 2: Entrenamiento y Optimización**
**Duración:** Semana 2  
**Entregables:**
```
Fine-tuning del modelo ViT implementado
Técnicas de data augmentation aplicadas
Sistema de checkpoints y logging
Pipeline de entrenamiento configurable
```

**Código Clave Implementado:**
```python
# src/plant_disease/training/train.py
def train_one_epoch(model, loader, optimizer, device, log_every=50):
    """Entrena el modelo una época con logging detallado"""
    
def evaluate(model, loader, device):
    """Evalúa el modelo con métricas de accuracy y loss"""
```

### **SPRINT 3: Evaluación y Validación**
**Duración:** Semana 3  
**Entregables:**
```
Sistema de evaluación con métricas múltiples
Matriz de confusión implementada
Análisis de rendimiento por clase
Pipeline de inferencia optimizado
```

**Testing Implementado:**
```
tests/
├── test_collate.py          # Tests del collate function
├── test_datasets.py         # Tests de carga de datos
├── test_imports.py          # Tests de importación
├── test_inference.py        # Tests de inferencia
├── test_models_vit.py       # Tests del modelo
└── test_predict_utils.py    # Tests de utilities
```

### **SPRINT 4: Despliegue y Demo**
**Duración:** Semana 4  
**Entregables:**
```
Exportación a formatos ONNX/TorchScript
Aplicación Streamlit funcional
Documentación completa
Demo interactiva desplegada
```

---

## 🏛️ **ARQUITECTURA DEL SISTEMA**

### **Estructura Modular (Bajo Acoplamiento)**
```
src/plant_disease/
├── apps/
│   └── streamlit_app.py      # Frontend (Streamlit UI)
├── data/
│   └── datasets.py           # Carga normalizada desde HF
├── evaluation/
│   └── evaluate.py           # Evaluación de modelos
├── inference/
│   └── predict.py            # CLI de inferencia
├── models/
│   └── vit.py               # Wrapper del modelo ViT
└── training/
    └── train.py             # Pipeline de entrenamiento
```

### **Componentes del Sistema**

**1. Capa de Datos**
- Carga automática desde Hugging Face
- Preprocesamiento estandarizado
- Normalización de columnas (image, label)

**2. Capa de Modelo**
- Vision Transformer (ViT) preentrenado
- Wrapper personalizable (`VitClassifier`)
- Soporte para fine-tuning

**3. Capa de Entrenamiento**
- Configuración por YAML
- Checkpoints automáticos
- Logging detallado

**4. Capa de Inferencia**
- CLI para predicción batch
- API para integración
- Soporte Top-K

**5. Capa de Aplicación**
- Interfaz web con Streamlit
- Subida de imágenes drag-and-drop
- Visualización de probabilidades

---

## ⚙️ **PIPELINE ML IMPLEMENTADO**

### **1. Data Loading & Preprocessing**
```yaml
# configs/train_vit_gvj.yaml
dataset:
  kind: hf
  hf_path: "GVJahnavi/PlantVillage_dataset"
  image_column: "image"
  label_column: "label"
  train_split: "train"
  val_split: "validation"
  test_split: "test"
```

**Características:**
- Carga automática desde Hugging Face
- Normalización de formato RGB
- Redimensionado a 224x224 (requerimiento ViT)
- Split estratificado manteniendo proporción de clases

### **2. Model Configuration**
```yaml
model:
  id: "google/vit-base-patch16-224-in21k"
  image_size: 224

train:
  batch_size: 8
  epochs: 2
  lr: 5e-5
  weight_decay: 0.01
```

**Características Técnicas:**
- Modelo preentrenado en ImageNet-21k
- Fine-tuning de capas superiores
- Optimizador AdamW con weight decay
- Learning rate schedule configurable

### **3. Training Pipeline**
```python
# Bucle de entrenamiento con checkpointing
for epoch in range(1, epochs + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    
    if val_loader:
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        if val_acc > best_val_acc:
            # Guardar mejor modelo
            save_best_checkpoint(model, processor, class_names, best_path)
```

### **4. Evaluation & Metrics**
- **Accuracy**: Precisión general del modelo
- **Loss tracking**: Seguimiento de pérdida en train/validation
- **Per-class metrics**: Análisis detallado por enfermedad
- **Confusion Matrix**: Matriz de confusión para análisis de errores

### **5. Model Export & Deployment**
```python
# Exportación a múltiples formatos
scripts/
├── export_onnx.py           # Exportación a ONNX
└── export_torchscript.py    # Exportación a TorchScript
```

---

## 🧪 **TESTING Y CALIDAD**

### **Estrategia de Testing Implementada**

**1. Unit Tests**
```bash
pytest tests/ -v
```
- `test_imports.py`: Verificación de importaciones
- `test_collate.py`: Testing del data collator
- `test_datasets.py`: Verificación de carga de datos
- `test_inference.py`: Testing de pipeline de inferencia
- `test_models_vit.py`: Testing del wrapper del modelo
- `test_predict_utils.py`: Testing de utilidades

**2. Code Quality**
```bash
# Linting y formateo
ruff check src tests        # Análisis estático
black src tests             # Formateo PEP 8
mypy src                   # Type checking
```

**3. Integration Tests**
- Testing end-to-end del pipeline de entrenamiento
- Verificación de compatibilidad entre componentes
- Testing de la aplicación Streamlit

### **Evidencias de Calidad del Código**

**Documentación:**
- Docstrings en todas las funciones
- README detallado con instrucciones
- Configuración externa en YAML
- Type hints en el código

**Estructura:**
- Separación clara de responsabilidades
- Configuración centralizada
- Manejo de errores robusto
- Logging estructurado

---

## **DESPLIEGUE Y DEMO**

### **Aplicación Streamlit**

**Características de la Demo:**
```python
# src/plant_disease/apps/streamlit_app.py
st.set_page_config(page_title="Plant Disease Classifier", layout="centered")
st.title("Plant Disease Classifier (ViT)")

# Interface intuitiva
uploaded = st.file_uploader("Sube una imagen de hoja", type=["jpg", "jpeg", "png"])

if uploaded:
    # Predicción en tiempo real
    results = predict_pil(image, model, processor, class_names, device, topk=topk)
    
    # Visualización de resultados
    df = pd.DataFrame(results, columns=["Clase", "Probabilidad"])
    st.bar_chart(df.set_index("Clase"))
    st.write(df)
```

**Funcionalidades Implementadas:**
- **Upload de imágenes**: Drag & drop interface
- **Predicción en tiempo real**: Respuesta inmediata
- **Top-K results**: Configurable (1-10)
- **Visualización**: Gráficos de barras con probabilidades
- **Model switching**: Selección de directorio de modelo
- **Caching**: Optimización de carga del modelo

### **Comandos de Ejecución**

**Entrenamiento:**
```bash
python -m plant_disease.training.train --config configs/train_vit_gvj.yaml
```

**Inferencia CLI:**
```bash
python -m plant_disease.inference.predict --image path/imagen.jpg --topk 5
```

**Demo Streamlit:**
```bash
python -m streamlit run src/plant_disease/apps/streamlit_app.py
```

**Export del modelo:**
```bash
# ONNX
python scripts/export_onnx.py --checkpoint runs/vit-gvj/final/pytorch_model.bin --out artifacts/model.onnx

# TorchScript  
python scripts/export_torchscript.py --checkpoint runs/vit-gvj/final/pytorch_model.bin --out artifacts/model.ts
```

---

## **RESULTADOS Y CONCLUSIONES**

### **Logros Técnicos Alcanzados**

**1. Pipeline Completo de ML**
- **Data Loading**: Automático desde Hugging Face
- **Preprocessing**: Normalización y augmentation
- **Training**: Fine-tuning con checkpointing
- **Evaluation**: Métricas comprehensivas
- **Inference**: CLI y API disponibles
- **Deployment**: App web funcional

**2. Arquitectura Escalable**
- **Modularidad**: Componentes independientes y reutilizables
- **Configurabilidad**: Parámetros externalizados en YAML
- **Extensibilidad**: Fácil agregar nuevos modelos o datasets
- **Mantenibilidad**: Código bien documentado y testeado

**3. Metodología Ágil Aplicada**
- **Sprints de 1 semana**: Entregas incrementales
- **Testing continuo**: 6 suites de tests automatizados
- **Documentación**: README y docstrings completos
- **Code quality**: Linting, formateo y type checking
- **CI/CD ready**: Estructura preparada para integración continua

### **Evidencias de Calidad**

**Estructura del Proyecto:**
```
detector_enfermedades_cultivo/
├── configs/                 # Configuraciones externalizadas
├── scripts/                 # Utilidades de exportación
├── src/plant_disease/       # Código fuente modular
├── tests/                   # Suite completa de tests
├── pyproject.toml          # Gestión de dependencias
├── requirements.txt        # Dependencias claras
└── README.md              # Documentación completa
```

**Métricas de Calidad del Código:**
- **Coverage**: Tests cubren componentes principales
- **PEP 8**: Código formateado según estándares
- **Type Safety**: Type hints implementados
- **Documentation**: 100% de funciones documentadas
- **Modularity**: Alto cohesión, bajo acoplamiento

### **Cumplimiento de Objetivos Originales**

| Objetivo Propuesto | Estado | Evidencia |
|-------------------|---------|-----------|
| Fine-tuning del ViT | Logrado | `src/plant_disease/training/train.py` |
| Data Augmentation | Logrado | Configuración en `ViTCollator` |
| Evaluación con métricas | Logrado | Función `evaluate()` implementada |
| Export ONNX/TorchScript | Logrado | Scripts en `scripts/` |
| Demo interactiva | Mejorado | Streamlit > Gradio (más moderno) |

### **Valor Agregado del Proyecto**

**1. Accesibilidad**
- **Deployment fácil**: Un comando para ejecutar la demo
- **Mobile-ready**: Exportación a formatos optimizados
- **Costo-efectivo**: Uso de modelos preentrenados

**2. Escalabilidad Técnica**  
- **Configuración externa**: Fácil experimentación
- **Arquitectura modular**: Componentes intercambiables
- **Performance tracking**: Métricas y logging detallado

**3. Calidad Empresarial**
- **Testing automatizado**: Garantía de calidad
- **Documentación completa**: Facilitación de mantenimiento
- **Metodología probada**: CRISP-DM + principios ágiles

---

## **LECCIONES APRENDIDAS Y METODOLOGÍA ÁGIL**

### **Aplicación Exitosa de Principios Ágiles**

**1. Iteración Corta y Feedback Continuo**
- Sprints de 1 semana permitieron ajustes rápidos
- Testing continuo detectó problemas temprano
- Demo funcional desde la semana 2

**2. Colaboración y Comunicación**
- Código modular facilitó trabajo en paralelo
- Configuraciones externalizadas redujeron conflictos
- Documentación clara mejoró la colaboración

**3. Adaptabilidad al Cambio**
- Cambio de Gradio a Streamlit (mejor UX)
- Arquitectura flexible permitió experimentación
- Testing robusto facilitó refactoring

**4. Entrega de Valor Continuo**
- Cada sprint produjo componentes funcionales
- Pipeline incremental desde datos hasta deploy
- Demo lista para presentación desde semana 3

### **Impacto en el Sector Agrícola**
- **Democratización**: Herramientas IA accesibles para pequeños productores
- **Detección temprana**: Prevención de pérdidas de cultivos
- **Tecnología móvil**: Diagnóstico en campo sin equipos especializados

---

## **PRÓXIMOS PASOS**

### **Mejoras Técnicas Propuestas**
1. **Optimización móvil**: Cuantización y pruning del modelo
2. **Más cultivos**: Expansión a datasets adicionales
3. **Integración IoT**: Sensores automáticos en campo
4. **MLOps**: CI/CD pipeline para reentrenamiento automático

### **Escalabilidad del Negocio**
1. **API REST**: Servicios web para integración
2. **App móvil nativa**: Android/iOS con cámara optimizada
3. **Dashboard analytics**: Métricas de salud de cultivos
4. **Integración ERP**: Conexión con sistemas agrícolas existentes

---

## 🎉 **CONCLUSIÓN**

Este proyecto demuestra la **aplicación exitosa de metodología ágil** en el desarrollo de un sistema completo de ML, desde la investigación hasta el despliegue. 

**Factores Clave del Éxito:**
- **Metodología CRISP-DM** adaptada con sprints ágiles
- **Arquitectura modular** que facilita desarrollo paralelo  
- **Testing continuo** que garantiza calidad
- **Documentación completa** que facilita mantenimiento
- **Demo funcional** que demuestra valor real

El resultado es una **herramienta accesible y efectiva** que puede impactar positivamente la agricultura de precisión, demostrando que la aplicación correcta de metodologías ágiles en proyectos de IA puede generar soluciones robustas y escalables.

---

*Presentación preparada para la sustentación del proyecto de Desarrollo de Proyectos de Inteligencia Artificial - UAO 2025-02*
