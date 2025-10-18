# Plant Disease Detection (ViT)

Clasificador de enfermedades en cultivos basado en **Vision Transformer (ViT)**
entrenado con el conjunto **PlantVillage** (`GVJahnavi/PlantVillage_dataset`).
Incluye entrenamiento configurable, inferencia por CLI y **Streamlit**, exportación
(ONNX/TorchScript) y una batería mínima de pruebas con **pytest**.

---

## Contenidos

- [Características](#características)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Docker](#docker)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Dataset](#dataset)
- [Entrenamiento](#entrenamiento)
- [Inferencia por CLI](#inferencia-por-cli)
- [Aplicación Streamlit](#aplicación-streamlit)
- [Exportación de modelos](#exportación-de-modelos)
- [Pruebas](#pruebas)
- [Calidad de código](#calidad-de-código)
- [Notas y solución de problemas](#notas-y-solución-de-problemas)
- [Licencia](#licencia)

---

## Características

- Entrenamiento controlado por **YAML** (ruta de salida, epochs, LR, etc.).
- Carga robusta del dataset desde Hugging Face con normalización de columnas.
- Inferencia en:
  - **CLI** con `predict.py`.
  - **Streamlit** en `src/plant_disease/apps/streamlit_app.py`.
- Exportación a **ONNX** y **TorchScript** (scripts en `scripts/`).
- Suite básica de **pytest** y herramientas de estilo (`ruff`, `black`).

---

## Requisitos

- Python **3.10+**
- Opcional: GPU CUDA para acelerar el entrenamiento.
- Gestor de entornos recomendado: **[uv](https://docs.astral.sh/uv/)**.

Las versiones de dependencias se declaran en `pyproject.toml` y se reflejan
también en `requirements.txt`.

---

## Instalación

### Opción A · Docker (recomendada para producción)

**Usando Docker Compose (más fácil)**

```bash
# Ejecutar la aplicación Streamlit
docker-compose up plant-disease-app

# Entrenar el modelo
docker-compose --profile training up plant-disease-training

# Ejecutar inferencia
docker-compose --profile inference run plant-disease-inference python -m plant_disease.inference.predict --image path/to/image.jpg
```

**Usando scripts de Docker**

```bash
# Ejecutar aplicación Streamlit
./docker_scripts/run_streamlit.sh

# Entrenar modelo
./docker_scripts/run_training.sh [config_file]

# Inferencia en una imagen
./docker_scripts/run_inference.sh path/to/image.jpg [model_dir] [topk]
```

### Opción B · uv (recomendada para desarrollo)

**Windows (PowerShell)**

```powershell
uv venv
.\.venv\Scripts\Activate.ps1

$env:UV_LINK_MODE = "copy"
uv pip install -e .         # paquete en modo editable
uv pip install -e ".[dev]"  # opcional: ruff/black/mypy/pytest
```

**Linux/macOS (bash)**

```bash
uv venv
source .venv/bin/activate

uv pip install -e .
uv pip install -e ".[dev]"  # opcional
```

### Opción C · venv + pip estándar

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Docker

La aplicación está completamente dockerizada para facilitar el despliegue y la reproducción del entorno.

### Archivos Docker incluidos

- `Dockerfile`: Imagen base con Python 3.10 y todas las dependencias
- `docker-compose.yml`: Configuración para ejecutar diferentes servicios
- `.dockerignore`: Exclusión de archivos innecesarios del contexto de construcción
- `docker_scripts/`: Scripts bash para facilitar el uso

### Uso rápido con Docker

**1. Aplicación Streamlit**
```bash
docker-compose up plant-disease-app
# o usar el script
./docker_scripts/run_streamlit.sh
```
Disponible en http://localhost:8501

**2. Entrenamiento**
```bash
docker-compose --profile training up plant-disease-training
# o usar el script con configuración personalizada
./docker_scripts/run_training.sh configs/mi_config.yaml
```

**3. Inferencia**
```bash
./docker_scripts/run_inference.sh data/mi_imagen.jpg runs/vit-gvj/final 5
```

### Ventajas del uso con Docker

- **Consistencia**: Mismo entorno en desarrollo, testing y producción
- **Aislamiento**: No afecta el sistema host
- **Portabilidad**: Ejecuta en cualquier sistema con Docker
- **Persistencia**: Los modelos y datos se mantienen con volúmenes montados

---

## Estructura del proyecto

```
proyecto_final/
├── configs/
│   └── train_vit_gvj.yaml
├── scripts/
│   ├── export_onnx.py
│   ├── export_torchscript.py
│   └── save_samples.py
├── src/
│   └── plant_disease/
│       ├── apps/
│       │   └── streamlit_app.py
│       ├── data/
│       │   └── datasets.py
│       ├── inference/
│       │   └── predict.py
│       ├── models/
│       │   └── vit.py
│       └── training/
│           └── train.py
├── tests/
│   ├── test_collate.py
│   ├── test_imports.py
│   └── test_inference.py
├── artifacts/           # onnx/ts (salidas)
├── checkpoints/         # checkpoints por época + best
├── runs/                # modelo final (save_pretrained)
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Dataset

El proyecto usa **GVJahnavi/PlantVillage_dataset** (Hugging Face). La carga se
realiza con `datasets.load_dataset` y se normaliza a:

- `image` · imagen PIL
- `label` · entero (`ClassLabel`) o mapeado desde texto

Para disponer de imágenes de prueba locales puedes ejecutar:

```bash
python scripts/save_samples.py --out-dir data/test_images --per-class 1
```

---

## Entrenamiento

Revisa y ajusta `configs/train_vit_gvj.yaml`. Ejemplo:

```yaml
model:
  id: google/vit-base-patch16-224-in21k

dataset:
  hf_path: GVJahnavi/PlantVillage_dataset
  image_column: image
  label_column: label
  train_split: train
  val_split: validation
  test_split: test

train:
  epochs: 2
  batch_size: 32
  lr: 5e-5
  weight_decay: 0.01
  seed: 42
  # Entrenamiento rápido:
  # max_train_samples: 4000
  # max_eval_samples: 1000
  # num_workers: 0  # en Windows suele ser seguro 0

paths:
  output_dir: runs/vit-gvj
  checkpoint_dir: checkpoints/vit-gvj
  save_every: 1
  log_every: 50
```

Ejecuta:

```bash
python -m plant_disease.training.train --config configs/train_vit_gvj.yaml
```

Salidas relevantes:

- Mejor checkpoint: `checkpoints/vit-gvj/best/`
- Checkpoints por época: `checkpoints/vit-gvj/epoch-*/`
- Modelo final (HF `save_pretrained`): `runs/vit-gvj/final/`
  - Incluye `class_names.json`

---

## Inferencia por CLI

```bash
python -m plant_disease.inference.predict --image path/a/imagen.jpg --topk 5
# opcional
# --model_dir runs/vit-gvj/final
```

---

## Aplicación Streamlit

Desde la raíz del repo:

```bash
python -m streamlit run src/plant_disease/apps/streamlit_app.py
```

En la barra lateral se configura el directorio del modelo
(por defecto `runs/vit-gvj/final`) y el Top-K. Carga una imagen para ver el
ranking por clase y la clase ganadora.

---

## Exportación de modelos

**ONNX**

```bash
python scripts/export_onnx.py   --checkpoint runs/vit-gvj/final/pytorch_model.bin   --out artifacts/model.onnx   --model_name google/vit-base-patch16-224-in21k   --num_labels 38
```

**TorchScript**

```bash
python scripts/export_torchscript.py   --checkpoint runs/vit-gvj/final/pytorch_model.bin   --out artifacts/model.ts   --model_name google/vit-base-patch16-224-in21k   --num_labels 38
```

---

## Pruebas

```bash
pytest -q
```

Los tests que requieren un modelo entrenado se omiten automáticamente si no se
encuentra `runs/vit-gvj/final`.

---

## Calidad de código

```bash
ruff check src tests
black src tests
# opcional
mypy src
```

---

## Notas y solución de problemas

- **Windows + PIL**: evita escribir sobre `NamedTemporaryFile(delete=True)`;
  usa `TemporaryDirectory()` (reflejado en tests).
- **Solo CPU**: usa `max_train_samples`/`max_eval_samples` para iteraciones
  rápidas.
- **CUDA**: si hay GPU disponible se utilizará automáticamente.

---

## Licencia

MIT. Consulta el archivo `LICENSE`.

---

## Autores

- **Joan Andres Velasquez** — https://github.com/JoanV83
- **Edwin Vicente Zapata** — https://github.com/edwinviz
- **Miguel Saavedra** — https://github.com/mash4403
- **Andres Velasco** — https://github.com/Andres-Velasco07