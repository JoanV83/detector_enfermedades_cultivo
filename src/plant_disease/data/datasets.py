"""Utilidades de carga y normalización de datasets desde Hugging Face Hub.

Este módulo expone funciones para cargar un dataset de imágenes y dejarlo en
un formato estándar con columnas:

- ``image``: objeto PIL (vía ``datasets.Image``).
- ``label``: entero o ``ClassLabel``.

Casos soportados para la columna de etiqueta:
- ``ClassLabel``: se usan sus ``names`` como ``class_names``.
- Entero: se generan nombres genéricos ``class_{i}``.
- Texto: se unifica vocabulario global entre splits y se remapea a IDs.

Si no existe split de validación en el Hub, se crea tomando el 10 % del train.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import ClassLabel, Dataset, DatasetDict
from datasets import Image as HFImage
from datasets import Features, Value, load_dataset


def _rename_and_clean_columns(
    dset: Dataset,
    image_column: str,
    label_column: str,
) -> Dataset:
    """Renombra y tipa columnas a un estándar ``image`` (PIL) y ``label``.

    - Fuerza la columna de imagen a ``datasets.Image`` (PIL).
    - Renombra a ``image``/``label`` si los nombres originales difieren.
    - Elimina columnas no usadas.

    Args:
        dset: Split de ``datasets.Dataset``.
        image_column: Nombre de la columna de imagen en el split original.
        label_column: Nombre de la columna de etiqueta en el split original.

    Returns:
        Split con columnas normalizadas.
    """
    # Asegurar tipo de imagen PIL
    if not isinstance(dset.features.get(image_column), HFImage):
        dset = dset.cast_column(image_column, HFImage())

    rename: Dict[str, str] = {}
    if image_column != "image":
        rename[image_column] = "image"
    if label_column != "label":
        rename[label_column] = "label"
    if rename:
        dset = dset.rename_columns(rename)

    # Mantener solo columnas de interés
    keep = {"image", "label"}
    drop = [c for c in dset.column_names if c not in keep]
    if drop:
        dset = dset.remove_columns(drop)

    return dset


def _infer_label_column(raw: DatasetDict, user_label_col: str) -> str:
    """Determina la columna de etiqueta a usar.

    Si ``user_label_col`` no es ``ClassLabel``, se intenta localizar alguna
    otra columna ``ClassLabel`` dentro del split de referencia.

    Args:
        raw: Conjunto de splits cargados desde el Hub.
        user_label_col: Columna sugerida por el usuario.

    Returns:
        Nombre de la columna de etiqueta a utilizar.
    """
    probe_split = "train" if "train" in raw else next(iter(raw.keys()))
    probe = raw[probe_split]
    feat = probe.features.get(user_label_col)
    if isinstance(feat, ClassLabel):
        return user_label_col
    # Buscar otra columna que sí sea ClassLabel
    for name, f in probe.features.items():
        if isinstance(f, ClassLabel):
            return name
    return user_label_col


def _normalize_label_value(val, label2id: Dict[str, int]) -> int:
    """Convierte un valor de etiqueta a entero.

    - Si ya es entero, lo retorna tal cual.
    - Si es ``bytes``, decodifica a UTF-8.
    - Si es texto, mapea usando ``label2id``.

    Args:
        val: Valor de etiqueta (int | str | bytes).
        label2id: Mapeo global texto → id.

    Returns:
        Entero ID de clase.
    """
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (bytes, bytearray)):
        val = val.decode("utf-8")
    return int(label2id[str(val)])


def _map_text_labels_to_ids(dset: Dataset, label2id: Dict[str, int]) -> Dataset:
    """Remapea etiquetas textuales a enteros usando ``label2id``.

    Args:
        dset: Split normalizado con columnas ``image`` y ``label``.
        label2id: Vocabulario global texto → id.

    Returns:
        Split con ``label`` entero.
    """
    return dset.map(lambda ex: {"label": _normalize_label_value(ex["label"], label2id)})


def _collect_text_labels_from_splits(splits: Dict[str, Dataset]) -> List[str]:
    """Recopila todas las etiquetas *textuales* desde los splits normalizados.

    Se usa cuando ``label`` no es ``ClassLabel`` ni entero, sino texto/bytes.

    Args:
        splits: Diccionario de splits ya normalizados (p. ej. ``out``).

    Returns:
        Lista de etiquetas (str) para construir vocabulario global.
    """
    values: List[str] = []
    for d in splits.values():
        if len(d) == 0:
            continue
        col = d["label"]
        # ``col`` puede ser una lista o un array Arrow; convertir a lista
        arr = list(col) if not isinstance(col, list) else col
        for v in arr:
            if isinstance(v, (bytes, bytearray)):
                values.append(v.decode("utf-8"))
            else:
                values.append(str(v))
    return values


def load_generic_from_hub(
    hf_path: str,
    image_column: str = "image",
    label_column: str = "label",
    splits: Optional[Dict[str, Optional[str]]] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[Dict[str, Dataset], List[str]]:
    """Carga un dataset del Hub y lo normaliza a (image, label).

    Soporta etiquetas como ``ClassLabel``, enteros o texto. Si son texto, se
    remapean a IDs globales consistentes entre splits.

    Si no existe ``validation``, se crea a partir del 10 % de ``train``.

    Args:
        hf_path: Nombre/ruta del dataset en el Hub.
        image_column: Columna de imagen a usar.
        label_column: Columna de etiqueta a usar.
        splits: Mapeo deseado de salida → split real.
            Por defecto: ``{"train": "train", "val": "validation", "test": "test"}``.
        cache_dir: Directorio de caché de datasets.

    Returns:
        Tuple con: (``splits_dict``, ``class_names``).
    """
    if splits is None:
        splits = {"train": "train", "val": "validation", "test": "test"}

    raw: DatasetDict = load_dataset(hf_path, cache_dir=cache_dir)
    label_column = _infer_label_column(raw, label_column)

    # Normalizar y filtrar los splits presentes en el Hub
    present = {k: s for k, s in splits.items() if s and s in raw}
    out: Dict[str, Dataset] = {}
    for k, s in present.items():
        d = raw[s]
        d = _rename_and_clean_columns(
            d, image_column=image_column, label_column=label_column
        )
        out[k] = d

    # Crear validación si no existe
    if "val" not in out and "train" in out:
        n = len(out["train"])
        m = max(1, int(0.1 * n))
        out["val"] = out["train"].select(range(m))
        out["train"] = out["train"].select(range(m, n))

    # Determinar tipo de etiqueta a partir de un split de ejemplo
    example_split = "train" if "train" in out else next(iter(out.keys()))
    label_feat = out[example_split].features.get("label")

    # Caso A: ClassLabel → usar sus nombres directamente
    if isinstance(label_feat, ClassLabel):
        class_names = label_feat.names
        return out, class_names

    # Caso B: etiqueta no es ClassLabel
    #   B1) ya son enteros → generar nombres genéricos class_i
    #   B2) son texto/bytes → construir vocabulario global desde *out* (no raw)
    sample = out[example_split][0]["label"] if len(out[example_split]) > 0 else 0
    if isinstance(sample, (int, np.integer)):
        max_label = -1
        for d in out.values():
            if len(d) > 0:
                max_label = max(max_label, int(max(d["label"])))
        class_names = [f"class_{i}" for i in range(max_label + 1)]
        return out, class_names

    # B2) Etiquetas textuales: crear vocabulario global recorriendo *out*
    all_text_labels = _collect_text_labels_from_splits(out)
    class_names = sorted(set(all_text_labels))
    label2id = {name: i for i, name in enumerate(class_names)}

    # Remapear cada split a ids
    for k in list(out.keys()):
        out[k] = _map_text_labels_to_ids(out[k], label2id)

    return out, class_names


def load_plantvillage(
    dataset_name: str = "GVJahnavi/PlantVillage_dataset",
    image_column: str = "image",
    label_column: str = "label",
    cache_dir: Optional[str] = None,
) -> Tuple[Dict[str, Dataset], List[str]]:
    """Carga PlantVillage desde el Hub con las mismas garantías que el genérico.

    Args:
        dataset_name: Identificador en Hugging Face Hub.
        image_column: Columna de imagen a usar.
        label_column: Columna de etiqueta a usar.
        cache_dir: Directorio de caché de datasets.

    Returns:
        Tuple con: (``splits_dict``, ``class_names``).
    """
    splits, class_names = load_generic_from_hub(
        hf_path=dataset_name,
        image_column=image_column,
        label_column=label_column,
        splits={"train": "train", "val": "validation", "test": "test"},
        cache_dir=cache_dir,
    )
    return splits, class_names
