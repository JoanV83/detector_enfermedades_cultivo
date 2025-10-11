"""Aplicación Streamlit para clasificar enfermedades en plantas (ViT).

- Carga un modelo entrenado (formato `save_pretrained`).
- Permite subir una imagen y devuelve el Top-K de clases con probabilidades.
- Usa caché para acelerar la carga del modelo.
"""

from __future__ import annotations

from io import BytesIO
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, ViTForImageClassification

from plant_disease.inference.predict import DEFAULT_MODEL_DIR, load_model


st.set_page_config(page_title="Plant Disease Classifier", layout="centered")
st.title("🌿 Plant Disease Classifier (ViT)")


def _predict_pil(
    img: Image.Image,
    model: ViTForImageClassification,
    processor: AutoImageProcessor,
    class_names: List[str],
    device: torch.device,
    topk: int = 5,
) -> List[Tuple[str, float]]:
    """Predice Top-K clases a partir de una imagen PIL.

    Args:
        img: Imagen PIL convertida a RGB.
        model: Modelo ViT ya cargado.
        processor: Procesador de imágenes del modelo.
        class_names: Lista de nombres de clases en el mismo orden del modelo.
        device: Dispositivo en el que está el modelo.
        topk: Número de resultados a devolver (ordenados por probabilidad).

    Returns:
        Lista de tuplas (nombre_clase, probabilidad).
    """
    img = img.convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    idx = np.argsort(probs)[::-1][:topk]
    return [(class_names[i], float(probs[i])) for i in idx]


# --- Panel lateral ---
model_dir = st.sidebar.text_input(
    "Directorio del modelo", value=DEFAULT_MODEL_DIR
)
topk = st.sidebar.slider("Top-K", min_value=1, max_value=10, value=5)


@st.cache_resource(show_spinner=True)
def _load(model_dir_path: str):
    """Carga (y cachea) el modelo y el procesador desde `model_dir_path`."""
    return load_model(model_dir_path)


# Cargar modelo
try:
    model, processor, class_names, device = _load(model_dir)
    st.sidebar.success(f"Modelo cargado en: {device}")
except Exception as exc:
    st.sidebar.error(f"No se pudo cargar el modelo: {exc}")
    st.stop()

# Cargador de archivos
uploaded = st.file_uploader(
    "Sube una imagen de hoja", type=["jpg", "jpeg", "png"]
)

if uploaded:
    # Mostrar imagen
    img_bytes = BytesIO(uploaded.read())
    image = Image.open(img_bytes).convert("RGB")
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Predicción
    with st.spinner("Prediciendo..."):
        results = _predict_pil(
            image, model, processor, class_names, device, topk=topk
        )

    # Visualización
    df = pd.DataFrame(results, columns=["Clase", "Probabilidad"])
    st.bar_chart(df.set_index("Clase"))
    st.write(df)
else:
    st.info("Sube una imagen para obtener predicciones.")