import os, json
from transformers import AutoImageProcessor, AutoModelForImageClassification

OUT_DIR = "runs/vit-gvj/final"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_ID = "google/vit-base-patch16-224-in21k"
print(f"Descargando y guardando {MODEL_ID} en {OUT_DIR} ...")

proc = AutoImageProcessor.from_pretrained(MODEL_ID)
proc.save_pretrained(OUT_DIR)

model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
model.save_pretrained(OUT_DIR)

id2label = getattr(model.config, "id2label", {})
class_names = [id2label[i] for i in sorted(id2label)] if id2label else []
with open(os.path.join(OUT_DIR, "class_names.json"), "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

print(" Listo:", OUT_DIR)
