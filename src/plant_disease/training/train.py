"""Rutina de entrenamiento/validación para ViT con Hugging Face.

Lee una configuración YAML, carga datos desde el Hub, entrena un modelo
ViT y guarda checkpoints (mejor y por época) además del artefacto final.

El módulo también exporta:
- ``ViTCollator``: collate para convertir ejemplos (PIL/ndarray) a tensores.
- ``collate_fn``: atajo compatible con tests (requiere un ``processor``).

Ejemplo de ejecución:
    python -m plant_disease.training.train --config configs/train_vit.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from PIL import Image
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

from plant_disease.data.datasets import load_generic_from_hub
from plant_disease.models.vit import VitClassifier, VitConfig


# --------------------------------------------------------------------- #
# Utilidades                                                            #
# --------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Fija semillas de Python/NumPy/PyTorch para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    # Evitar determinismo estricto en CPU/Windows que puede ser muy lento
    torch.use_deterministic_algorithms(False)


def ensure_dir(path: str) -> None:
    """Crea el directorio si no existe (idempotente)."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def maybe_slice(dset: Dataset, n: Optional[int]) -> Dataset:
    """Devuelve un subconjunto de tamaño ``n`` si se indica."""
    if not n:
        return dset
    return dset.select(range(min(n, len(dset))))


def _worker_init_fn(worker_id: int) -> None:
    """Inicializa semillas independientes por worker de DataLoader."""
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


# --------------------------------------------------------------------- #
# Collate                                                               #
# --------------------------------------------------------------------- #
@dataclass
class ViTCollator:
    """Convierte una lista de ejemplos en batch para ViT.

    Toma imágenes PIL o arreglos numpy (H, W, C), las fuerza a RGB y
    aplica el ``processor`` para obtener tensores listos para el modelo.
    """

    processor: AutoImageProcessor

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images: List[Image.Image] = []
        labels: List[int] = []
        for item in batch:
            img = item["image"]
            if isinstance(img, Image.Image):
                pil = img.convert("RGB")
            else:
                pil = Image.fromarray(np.asarray(img)).convert("RGB")
            images.append(pil)
            labels.append(int(item["label"]))

        inputs = self.processor(images=images, return_tensors="pt")
        inputs["labels"] = torch.tensor(labels, dtype=torch.long)
        return inputs


def collate_fn(
    batch: List[Dict[str, Any]],
    processor: Optional[AutoImageProcessor] = None,
) -> Dict[str, torch.Tensor]:
    """Atajo compatible con tests para crear batches.

    Args:
        batch: Lista de ejemplos con ``image`` y ``label``.
        processor: ``AutoImageProcessor`` ya inicializado.

    Returns:
        Diccionario con tensores listos para el modelo.
    """
    if processor is None:
        raise ValueError("Se requiere un 'processor' para 'collate_fn'.")
    return ViTCollator(processor)(batch)


# --------------------------------------------------------------------- #
# Entrenamiento / Evaluación                                            #
# --------------------------------------------------------------------- #
def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_every: int = 50,
) -> float:
    """Entrena el modelo una época y devuelve la pérdida media."""
    model.train()
    total_loss = 0.0
    seen = 0

    for step, batch in enumerate(loader, start=1):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        bs = batch["labels"].size(0)
        total_loss += loss.item() * bs
        seen += bs

        if log_every and step % log_every == 0:
            print(f"[train] step {step}/{len(loader)}  loss={loss.item():.4f}")

    return total_loss / max(1, seen)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evalúa el modelo (loss y accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss
        preds = out.logits.argmax(dim=-1)

        bs = batch["labels"].size(0)
        total_loss += loss.item() * bs
        correct += (preds == batch["labels"]).sum().item()
        seen += bs

    avg_loss = total_loss / max(1, seen)
    acc = correct / max(1, seen)
    return avg_loss, acc


# --------------------------------------------------------------------- #
# Main                                                                  #
# --------------------------------------------------------------------- #
def main() -> None:
    """Punto de entrada del script de entrenamiento."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Ruta al archivo YAML de configuración.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg["train"].get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)

    out_dir = cfg["paths"]["output_dir"]
    ckpt_dir = cfg["paths"]["checkpoint_dir"]
    ensure_dir(out_dir)
    ensure_dir(ckpt_dir)

    dcfg = cfg["dataset"]
    splits, class_names = load_generic_from_hub(
        hf_path=dcfg["hf_path"],
        image_column=dcfg.get("image_column", "image"),
        label_column=dcfg.get("label_column", "label"),
        splits={
            "train": dcfg.get("train_split", "train"),
            "val": dcfg.get("val_split", "validation"),
            "test": dcfg.get("test_split", "test"),
        },
        cache_dir=dcfg.get("cache_dir"),
    )
    preview = ", ".join(class_names[:10])
    suffix = "..." if len(class_names) > 10 else ""
    print(f"Clases: {len(class_names)} -> [{preview}{suffix}]")

    max_tr = cfg["train"].get("max_train_samples")
    max_ev = cfg["train"].get("max_eval_samples")
    if max_tr:
        splits["train"] = maybe_slice(splits["train"], max_tr)
    if "val" in splits and max_ev:
        splits["val"] = maybe_slice(splits["val"], max_ev)

    model_id = cfg["model"]["id"]
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = VitClassifier(
        VitConfig(model_name=model_id, num_labels=len(class_names))
    ).to(device)

    collate = ViTCollator(processor)
    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"].get("num_workers", 0))
    pin_mem = torch.cuda.is_available()

    train_loader = DataLoader(
        splits["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=pin_mem,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )

    val_loader = None
    if "val" in splits:
        val_loader = DataLoader(
            splits["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=pin_mem,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        )

    test_loader = None
    if "test" in splits:
        test_loader = DataLoader(
            splits["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=pin_mem,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        )

    lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"].get("weight_decay", 0.0))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    epochs = int(cfg["train"]["epochs"])
    log_every = int(cfg["paths"].get("log_every", 50))
    save_every = int(cfg["paths"].get("save_every", 1))

    # Guardamos nombres de clases como referencia
    with open(os.path.join(out_dir, "labels.json"), "w", encoding="utf-8") as fjs:
        json.dump(class_names, fjs, ensure_ascii=False, indent=2)

    best_val_acc = -1.0
    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, log_every=log_every
        )
        print(f"[epoch {epoch}] train_loss={train_loss:.4f}")

        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, device)
            print(f"[epoch {epoch}] val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_path = os.path.join(ckpt_dir, "best")
                ensure_dir(best_path)
                model.model.save_pretrained(best_path)
                processor.save_pretrained(best_path)
                with open(
                    os.path.join(best_path, "labels.json"),
                    "w",
                    encoding="utf-8",
                ) as fjs:
                    json.dump(class_names, fjs, ensure_ascii=False, indent=2)
                print(f"✅ Mejor modelo guardado en: {best_path}")

        if save_every and (epoch % save_every == 0):
            ep_path = os.path.join(ckpt_dir, f"epoch-{epoch}")
            ensure_dir(ep_path)
            model.model.save_pretrained(ep_path)
            processor.save_pretrained(ep_path)
            with open(
                os.path.join(ep_path, "labels.json"),
                "w",
                encoding="utf-8",
            ) as fjs:
                json.dump(class_names, fjs, ensure_ascii=False, indent=2)
            print(f"💾 Checkpoint guardado: {ep_path}")

    final_path = os.path.join(out_dir, "final")
    ensure_dir(final_path)
    model.model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    with open(os.path.join(final_path, "labels.json"), "w", encoding="utf-8") as fjs:
        json.dump(class_names, fjs, ensure_ascii=False, indent=2)
    print(f"\n🏁 Entrenamiento finalizado. Modelo en: {final_path}")

    if test_loader is not None:
        test_loss, test_acc = evaluate(model, test_loader, device)
        print(f"\n🧪 Test: loss={test_loss:.4f} acc={test_acc:.4f}")


if __name__ == "__main__":
    main()