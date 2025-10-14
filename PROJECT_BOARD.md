# Project Board - Detector de Enfermedades en Cultivos

## 📋 **INFORMACIÓN DEL PROYECTO**

**Repositorio:** detector_enfermedades_cultivo  
**Metodología:** CRISP-DM Adaptado con Principios Ágiles  
**Framework de Desarrollo:** Sprints de duración variable  
**Estado:** En Desarrollo

---

## 🎯 **OBJETIVOS DEL PROJECT BOARD**

1. **Seguimiento de progreso** de tareas y sprints
2. **Gestión de iteraciones** con duraciones flexibles
3. **Coordinación del equipo** de desarrollo
4. **Documentación de decisiones** técnicas
5. **Control de calidad** y testing continuo

---

## 📊 **ESTRUCTURA DEL BOARD**

### **Columnas del Project Board:**

| Columna | Descripción | Criterios de Entrada |
|---------|-------------|---------------------|
| 📝 **Backlog** | Tareas identificadas pero no planificadas | Issues creados, features propuestos |
| 🚀 **Sprint Planning** | Tareas seleccionadas para el próximo sprint | Estimadas, priorizadas, asignadas |
| 🏗️ **In Progress** | Tareas actualmente en desarrollo | Branch creada, trabajo iniciado |
| 🔍 **In Review** | Tareas completadas esperando revisión | PR abierto, tests pasando |
| ✅ **Done** | Tareas completadas y aprobadas | PR mergeado, features desplegadas |

---

## 🔄 **SPRINTS Y FASES**

### **Sprint 0: Configuración Inicial**
**Duración:** Variable según configuración del entorno  
**Objetivo:** Establecer base del proyecto

**Tasks:**
- [ ] Configuración del repositorio principal
- [ ] Setup del entorno de desarrollo
- [ ] Definición de estructura modular
- [ ] Configuración de testing inicial
- [ ] Documentación base (README, CONTRIBUTING)

### **Sprint 1: Comprensión y Preparación de Datos**
**Duración:** Variable según complejidad del dataset  
**Objetivo:** Análisis y preparación del PlantVillage Dataset

**Tasks:**
- [ ] Análisis exploratorio del dataset
- [ ] Implementación de data loaders
- [ ] Pipeline de preprocesamiento
- [ ] Validación de calidad de datos
- [ ] Split estratificado de datos

### **Sprint 2: Desarrollo del Modelo Base**
**Duración:** Variable según experimentación necesaria  
**Objetivo:** Implementación del modelo ViT y pipeline de entrenamiento

**Tasks:**
- [ ] Implementación del wrapper ViT
- [ ] Pipeline de entrenamiento básico
- [ ] Sistema de checkpointing
- [ ] Configuración externa (YAML)
- [ ] Logging y métricas básicas

### **Sprint 3: Optimización y Evaluación**
**Duración:** Variable según métricas obtenidas  
**Objetivo:** Fine-tuning y evaluación comprehensiva

**Tasks:**
- [ ] Fine-tuning del modelo ViT
- [ ] Data augmentation avanzada
- [ ] Sistema de evaluación completo
- [ ] Métricas detalladas y matriz de confusión
- [ ] Análisis de rendimiento por clase

### **Sprint 4: Despliegue y Demo**
**Duración:** Variable según complejidad de deployment  
**Objetivo:** Aplicación funcional y exportación del modelo

**Tasks:**
- [ ] Implementación de la aplicación Streamlit
- [ ] Exportación a ONNX/TorchScript
- [ ] Testing de integración completo
- [ ] Documentación de usuario final
- [ ] Demo interactiva lista

---

## 🏷️ **SISTEMA DE LABELS/TAGS**

### **Por Prioridad:**
- 🔴 **Critical**: Bloquea el desarrollo
- 🟡 **High**: Importante para el sprint actual  
- 🟢 **Medium**: Deseable pero no urgente
- 🔵 **Low**: Mejoras futuras

### **Por Categoría:**
- 🧠 **model**: Relacionado con el modelo ViT
- 📊 **data**: Manejo de datos y datasets
- 🧪 **testing**: Tests y calidad de código
- 📱 **app**: Aplicación Streamlit
- 📚 **docs**: Documentación
- 🏗️ **infrastructure**: Setup y configuración
- 🐛 **bug**: Errores a corregir
- ✨ **enhancement**: Mejoras y nuevas features

### **Por Estado Técnico:**
- 🔬 **research**: Requiere investigación
- ⚡ **quick**: Tarea rápida (< 2 horas)
- 🎯 **ready**: Lista para desarrollo
- 🚧 **blocked**: Esperando dependencia

---

## 👥 **ASIGNACIÓN DE RESPONSABILIDADES**

### **Roles del Equipo:**
- **J. A. Velásquez Vélez**: Model Architecture & Training
- **E. V. Zapata Cardona**: Data Pipeline & Preprocessing  
- **M. A. Saavedra Hurtado**: Testing & Quality Assurance
- **N. A. Velasco Castellanos**: Application & Deployment

### **Rotation Policy:**
- Los miembros pueden rotar entre tareas según disponibilidad
- Cada PR debe ser revisado por al menos un miembro diferente
- Knowledge sharing sessions al final de cada sprint

---

## 📈 **MÉTRICAS DE SEGUIMIENTO**

### **Velocity Metrics:**
- **Story Points completados** por sprint
- **Tiempo promedio** de resolución de issues
- **Número de PRs** mergeados por semana
- **Code coverage** de tests automatizados

### **Quality Metrics:**
- **Bugs reportados** vs **bugs resueltos**
- **Test success rate** en CI/CD
- **Documentation coverage** (funciones documentadas)
- **Model performance** (accuracy, F1-score)

### **Burndown Tracking:**
- Gráfico de tasks restantes por sprint
- Tiempo estimado vs tiempo real
- Identificación de bottlenecks

---

## 🔧 **WORKFLOW Y PROCESOS**

### **Git Workflow:**
```
main (stable)
├── develop (integration)
├── feature/[task-name] (development)
└── hotfix/[bug-name] (emergency fixes)
```

### **Definition of Done:**
Para que una tarea se considere "Done", debe cumplir:

✅ **Código:**
- Funcionalidad implementada según especificación
- Code review aprobado por al menos 1 peer
- Estándares de código seguidos (PEP 8, type hints)

✅ **Testing:**
- Unit tests escritos y pasando
- Integration tests cuando aplique
- Coverage mínimo del 70%

✅ **Documentación:**
- Docstrings actualizados
- README actualizado si es necesario
- Comentarios en código complejo

✅ **Integración:**
- Branch mergeado a develop
- CI/CD pipeline pasando
- No rompe funcionalidad existente

### **Sprint Review Process:**
1. **Demo**: Mostrar funcionalidad desarrollada
2. **Retrospective**: Identificar mejoras del proceso
3. **Planning**: Seleccionar tasks para próximo sprint
4. **Estimation**: Story points para nuevas tasks

---

## 📋 **TEMPLATES DE ISSUES**

### **Feature Request Template:**
```markdown
## Feature Description
[Descripción clara de la feature]

## Acceptance Criteria
- [ ] Criterio 1
- [ ] Criterio 2
- [ ] Criterio 3

## Technical Notes
[Detalles técnicos, dependencias, consideraciones]

## Definition of Done
- [ ] Código implementado
- [ ] Tests escritos
- [ ] Documentación actualizada
- [ ] Code review aprobado
```

### **Bug Report Template:**
```markdown
## Bug Description
[Descripción del error]

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected vs Actual Behavior
**Expected:** [comportamiento esperado]
**Actual:** [comportamiento actual]

## Environment
- Python version: 
- Dependencies version:
- OS: 

## Priority
[Critical/High/Medium/Low]
```

---

## 🚀 **AUTOMATION Y HERRAMIENTAS**

### **GitHub Actions (CI/CD):**
- **Test Pipeline**: Ejecuta tests en cada PR
- **Code Quality**: Linting y formateo automático
- **Documentation**: Genera docs automáticamente
- **Security**: Escaneo de vulnerabilidades

### **Herramientas de Desarrollo:**
- **pytest**: Testing framework
- **black**: Code formatting
- **ruff**: Linting
- **mypy**: Type checking
- **pre-commit**: Git hooks

### **Integration Tools:**
- **GitHub Projects**: Visual project board
- **GitHub Milestones**: Sprint tracking
- **GitHub Discussions**: Team communication
- **GitHub Wiki**: Documentation repository

---

## 📊 **REPORTING Y COMUNICACIÓN**

### **Sprint Reports:**
Generados automáticamente cada viernes:
- Tasks completadas vs planificadas
- Blockers identificados
- Métricas de calidad (tests, coverage)
- Próximos pasos

### **Weekly Standup Format:**
- **Ayer**: Qué trabajé
- **Hoy**: En qué voy a trabajar  
- **Blockers**: Qué me impide avanzar
- **Ayuda**: Qué necesito del equipo

### **Communication Channels:**
- **GitHub Issues**: Discusiones técnicas
- **PR Comments**: Code review
- **Discussions**: Arquitectura y decisiones
- **README Updates**: Cambios en proceso

---

## 🎯 **OBJETIVOS POR SPRINT**

### **Sprint Actual: [Número/Nombre]**
**Fechas:** [DD/MM/YYYY - DD/MM/YYYY]  
**Objetivo:** [Descripción del objetivo principal]

**Tasks Planificadas:**
- [ ] Task 1 - Asignado a: [Miembro] - Priority: [High/Medium/Low]
- [ ] Task 2 - Asignado a: [Miembro] - Priority: [High/Medium/Low]
- [ ] Task 3 - Asignado a: [Miembro] - Priority: [High/Medium/Low]

**Métricas del Sprint:**
- Story Points planificados: [X]
- Story Points completados: [Y]
- Velocity: [Y/X * 100%]
- Bugs encontrados: [Z]
- Tests coverage: [W%]

---

## 📚 **RECURSOS Y DOCUMENTACIÓN**

### **Enlaces Importantes:**
- [Repository Main](https://github.com/JoanV83/detector_enfermedades_cultivo)
- [Fork Personal](https://github.com/mash4403/detector_enfermedades_cultivo)
- [Project Board](https://github.com/JoanV83/detector_enfermedades_cultivo/projects)
- [Wiki Documentation](https://github.com/JoanV83/detector_enfermedades_cultivo/wiki)

### **Documentos de Referencia:**
- `README.md`: Setup y uso básico
- `CONTRIBUTING.md`: Guías de contribución
- `PRESENTACION_PROYECTO.md`: Sustentación del proyecto
- `pyproject.toml`: Configuración de dependencies

### **External Resources:**
- [PlantVillage Dataset](https://www.tensorflow.org/datasets/catalog/plant_village)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [Hugging Face ViT](https://huggingface.co/google/vit-base-patch16-224)

---

*Este Project Board es un documento vivo que se actualiza según evoluciona el proyecto y la metodología ágil aplicada.*