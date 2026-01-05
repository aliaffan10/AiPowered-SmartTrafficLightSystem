import os
import subprocess
import yaml

# === CONFIGURATION ===
YOLO_DIR = "yolov5"  # Path to yolov5 directory
DATASET_DIR = "traffic_dataset"  # Path to your dataset
MODEL_NAME = "traffic_detector"
EPOCHS = 3
IMG_SIZE = 416
BATCH_SIZE = 16


# === 1. CHECK DATA.YAML ===
yaml_path = os.path.join(DATASET_DIR, "data.yaml")

if not os.path.exists(yaml_path):
    raise FileNotFoundError(f"Missing data.yaml at {yaml_path}")

with open(yaml_path, 'r') as f:
    data_yaml = yaml.safe_load(f)

print("\n✅ Found data.yaml:")
print(f"  Classes ({data_yaml['nc']}): {data_yaml['names']}")
print(f"  Train path: {data_yaml['train']}")
print(f"  Valid path: {data_yaml['val']}")

# === 2. VERIFY DATA FOLDERS ===
for subset in ["train", "valid"]:
    img_dir = os.path.join(DATASET_DIR, subset, "images")
    lbl_dir = os.path.join(DATASET_DIR, subset, "labels")
    assert os.path.isdir(img_dir), f"Missing images in {img_dir}"
    assert os.path.isdir(lbl_dir), f"Missing labels in {lbl_dir}"
    assert len(os.listdir(img_dir)) > 0, f"No images in {img_dir}"
    assert len(os.listdir(lbl_dir)) > 0, f"No labels in {lbl_dir}"

print("✅ Dataset structure looks good.\n")

# === 3. RUN TRAINING ===
train_script = os.path.join(YOLO_DIR, "train.py")
cmd = [
    "python3", train_script,
    "--img", str(IMG_SIZE),
    "--batch", str(BATCH_SIZE),
    "--epochs", str(EPOCHS),
    "--data", yaml_path,
    "--weights", "yolov5s.pt",
    "--name", MODEL_NAME
]

print(f" Starting training:\n{' '.join(cmd)}\n")
subprocess.run(cmd, check=True)

print("\n Training complete. Check runs/train/traffic_detector for results.")