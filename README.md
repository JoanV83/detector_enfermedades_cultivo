# Plant Disease Detection (ViT + UV)

Clasificador de enfermedades en cultivos usando **Vision Transformer (ViT)** y el
dataset **PlantVillage** (división oficial `train/validation/test` del repo
`GVJahnavi/PlantVillage_dataset`). Incluye entrenamiento, inferencia por CLI,
app de **Streamlit**, exportación a **ONNX/TorchScript** y **tests** con `pytest`.

## Características

- Código estilo **PEP 8**, con **docstrings** y tipado.
- Estructura limpia en `src/` (cohesión y bajo acoplamiento).
- Entrenamiento configurable por **YAML** en `configs/`.
- Inferencia por **CLI** y **Streamlit** (`src/plant_disease/apps/streamlit_app.py`).
- Exportadores a **ONNX** y **TorchScript** en `scripts/`.
- **pytest** listo (`tests/`) y herramientas de calidad: **ruff**, **black**.

---

## Requisitos

- **Python 3.10+**
- **[uv](https://docs.astral.sh/uv/)** (gestor rápido de entornos)
- GPU opcional (CUDA) para acelerar entrenamiento. En CPU también funciona.

---

## Instalación (con `uv`)

### Windows (PowerShell)
```powershell
# 1) Crear y activar entorno
uv venv
.\.venv\Scripts\Activate.ps1

# 2) Instalar el paquete en modo editable
$env:UV_LINK_MODE = "copy"
uv pip install -e .

# (opcional) herramientas de estilo
uv pip install -e ".[dev]"
```

### Linux / macOS
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
uv pip install -e ".[dev]"   # opcional (ruff/black/mypy/pytest)
```

> Todas las dependencias con versiones están en `pyproject.toml` y `requirements.txt`.

---

## Estructura del proyecto

```text
proyecto_final/
├── configs/
│   └── train_vit_gvj.yaml            # ejemplo de config para ViT + PlantVillage
├── scripts/
│   ├── export_onnx.py                # exportación a ONNX
│   └── export_torchscript.py         # exportación a TorchScript
├── src/
│   └── plant_disease/
│       ├── apps/
│       │   └── streamlit_app.py      # app Streamlit (UI)
│       ├── data/
│       │   └── datasets.py           # carga normalizada desde Hugging Face
│       ├── inference/
│       │   └── predict.py            # CLI de inferencia por imagen
│       ├── models/
│       │   └── vit.py                # wrapper ligero de ViT (opcional, útil en export)
│       └── training/
│           └── train.py              # bucle de entrenamiento + validación/test
├── tests/
│   ├── test_collate.py
│   ├── test_imports.py
│   ├── test_inference.py
│   ├── test_datasets_loading.py      # (nuevo sugerido)
│   └── test_training_config_io.py    # (nuevo sugerido)
├── runs/                             # artefactos de entrenamiento (salida)
├── checkpoints/                      # checkpoints (mejor y por época)
├── artifacts/                        # exportaciones (onnx/ts/…)
├── notebooks/                        # opcional
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

> La carpeta `src/plant_disease_uv.egg-info/` la crea la instalación editable; no la edites.

---

## Dataset

Se usa `GVJahnavi/PlantVillage_dataset` (Hugging Face). La carga es automática
desde `datasets` y el módulo `datasets.py` **normaliza** columnas a:
- `image` (PIL)  
- `label` (int `ClassLabel` o mapeada a int si venía como texto)

---

## Entrenamiento

Ajusta `configs/train_vit_gvj.yaml` según tus recursos. Ejemplo mínimo:

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
  # opcional (entrenamiento rápido)
  # max_train_samples: 4000
  # max_eval_samples: 1000
  # num_workers: 0   # en Windows suele ser más seguro dejar 0

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

Salida típica:
- Mejor checkpoint en `checkpoints/vit-gvj/best/`
- Checkpoints por época en `checkpoints/vit-gvj/epoch-*/`
- **Modelo final** en `runs/vit-gvj/final/` (+ `class_names.json`)

---

## Inferencia por línea de comandos

Predicción Top-K (sobre una imagen):

```bash
python -m plant_disease.inference.predict --image path/a/tu_imagen.jpg --topk 5
```

Por defecto usa el directorio `runs/vit-gvj/final`. Puedes cambiarlo con
`--model_dir`.

---

## App (Streamlit)

La app vive en `src/plant_disease/apps/streamlit_app.py`. Lánzala desde la raíz:

```bash
python -m streamlit run src/plant_disease/apps/streamlit_app.py
```

En la barra lateral puedes indicar el **directorio del modelo**
(p. ej. `runs/vit-gvj/final`) y el **Top-K**.  
Sube una imagen y verás las probabilidades por clase.

> Nota: eliminamos la demo de Gradio para simplificar. Todo queda en Streamlit.

---

## Exportar el modelo

### ONNX
```bash
python scripts/export_onnx.py   --checkpoint runs/vit-gvj/final/pytorch_model.bin   --out artifacts/model.onnx   --model_name google/vit-base-patch16-224-in21k   --num_labels 38
```

### TorchScript
```bash
python scripts/export_torchscript.py   --checkpoint runs/vit-gvj/final/pytorch_model.bin   --out artifacts/model.ts   --model_name google/vit-base-patch16-224-in21k   --num_labels 38
```

Ambos scripts usan el wrapper `VitClassifier` de `src/plant_disease/models/vit.py`.

---

## Pruebas

Ejecuta todos los tests:

```bash
pytest -q
```

Incluye pruebas de:
- **Importación** del paquete (`test_imports.py`).
- **Collate** de ViT (`test_collate.py`).
- **Inferencia** con imagen dummy (`test_inference.py`).
- **Carga de dataset** y mapeo de etiquetas (`test_datasets_loading.py`).
- **I/O de configuración de entrenamiento** (`test_training_config_io.py`).

> Los tests que requieren un modelo entrenado se **omiten** automáticamente si
no se encuentra `runs/vit-gvj/final`.

---

## Estilo y calidad (PEP 8)

Formateo y análisis:

```bash
ruff check src tests        # linter + import order
black src tests             # formateo
mypy src                    # (opcional) chequeo estático de tipos
```

Directrices PEP 8 aplicadas:
- Nombres de funciones/variables en `snake_case`; clases en `CamelCase`.
- Líneas ≤ 88 caracteres (configurado en `ruff`/`black`).
- Importaciones agrupadas (estándar / terceros / locales) con líneas en blanco.
- **Docstrings** en módulos, clases y funciones (triple comilla, propósito claro).

---

## Consejos y solución de problemas

- **Windows + PIL + archivos temporales**: evita `NamedTemporaryFile(delete=True)`
  cuando vayas a escribir con PIL; usa `TemporaryDirectory()` y guarda el archivo
  dentro (ya está contemplado en los tests).
- **CPU**: el entrenamiento puede ser lento; usa `max_train_samples`/`max_eval_samples`
  para iteraciones rápidas.
- **CUDA**: si está disponible, el entrenamiento usa GPU automáticamente.

---

## Licencia

Este proyecto se distribuye bajo licencia **MIT** (ver `LICENSE`).
