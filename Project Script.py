"""
Refactored, modular single-file implementation of the
AI-Powered Autonomous Retail Shelf Monitoring System.

Goals:
- Preserve original logic and behavior while improving structure.
- Add clear sections, reusable functions, and safer I/O handling.
- Keep training/benchmarking/inference/Flask app features in one file.
- Provide optional CLI switches to run specific stages.

NOTE:
- Roboflow API key is read from env ROB0FLOW_API_KEY if available, otherwise
  falls back to the literal key present in the original snippet (not recommended).
- Paths and behaviors mirror the original as closely as possible.
- Visualization calls (matplotlib) are kept, but guarded so they run only when selected.
- The HTML template and stock logic are unchanged in spirit.
- Minor typos from the original (e.g., "val" vs. "valid") are preserved to avoid
  altering dataset mapping logic. You may correct them locally if desired.

Usage examples:
    python retail_shelf_monitor.py --stage all
    python retail_shelf_monitor.py --stage train
    python retail_shelf_monitor.py --stage benchmark
    python retail_shelf_monitor.py --stage infer_test
    python retail_shelf_monitor.py --stage video --input_video path/to/video.mp4
    python retail_shelf_monitor.py --stage app
    python retail_shelf_monitor.py --stage dockerfile

"""
from __future__ import annotations

# ==== Imports and Setup =======================================================
import os
import re
import glob
import uuid
import random
import shutil
import tempfile
import subprocess
import argparse
from dataclasses import dataclass, field
from itertools import islice
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch  # noqa: F401 (import kept as in original)
import ultralytics
from ultralytics import YOLO
from roboflow import Roboflow

from flask import Flask, request, render_template_string, send_from_directory, url_for

# ==== Configuration ===========================================================

@dataclass
class Config:
    DATASET_NAME: str = "retail-product-checkout-35"
    DATASET_DIR: str = field(default_factory=lambda: os.path.join(os.getcwd(), 'data'))
    WORKDIR: str = field(default_factory=os.getcwd)

    # Roboflow
    ROBOFLOW_API_KEY_ENV: str = "ROB0FLOW_API_KEY"  # (zero in name preserved intentionally)
    ROBOFLOW_API_KEY_FALLBACK: str = "a7ixfLNh5wtqHUYhxvQf"  # from original (not recommended)
    ROBOFLOW_WORKSPACE: str = "cdio-zmfmj"
    ROBOFLOW_PROJECT: str = "retail-product-checkout"
    ROBOFLOW_VERSION: int = 35

    # Training defaults
    EPOCHS: int = 75
    IMGSZ: int = 640
    BATCH: int = 25
    PATIENCE: int = 10
    MODEL_VARIANTS: Tuple[str, ...] = ("yolov8n.pt", "yolov8s.pt", "yolov8m.pt")

    # Benchmark defaults
    MODEL_FILES: Tuple[str, ...] = (
        "models/yolov8n_best.pt",
        "models/yolov8s_best.pt",
        "models/yolov8m_best.pt",
    )

    # Inference defaults
    CONF_THRESHOLD: float = 0.4

    # Flask app / model path (as given in original)
    APP_MODEL_PATH: str = r"C:\Users\HP\PGC - AI\FSDSS Project\runs\detect\train\weights\best.pt"

    # Video processing defaults
    OUTPUT_VIDEO_DIR: str = field(default_factory=lambda: os.path.join(os.getcwd(), 'static'))

    # Dockerfile
    DOCKERFILE_NAME: str = "Dockerfile"

    def dataset_path(self) -> str:
        return os.path.join(self.DATASET_DIR, self.DATASET_NAME)


CFG = Config()

# ==== Dataset Handling ========================================================

def get_or_download_dataset(cfg: Config) -> Any:
    """Check local dataset path; if missing, download via Roboflow.
    Returns an object with a `.location` attribute (as in original).
    """
    dataset_path = cfg.dataset_path()

    class DummyDataset:
        def __init__(self, location: str):
            self.location = location

    if os.path.exists(dataset_path):
        print(f"[INFO] Dataset already exists at {dataset_path}")
        dataset = DummyDataset(location=dataset_path)
    else:
        api_key = os.environ.get(cfg.ROBOFLOW_API_KEY_ENV, cfg.ROBOFLOW_API_KEY_FALLBACK)
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(cfg.ROBOFLOW_WORKSPACE).project(cfg.ROBOFLOW_PROJECT)
        dataset = project.version(cfg.ROBOFLOW_VERSION).download("yolov8")

    print(f"[INFO] Using dataset at: {dataset.location}")
    return dataset

# ==== Preprocessing & Visualization ==========================================

CLASS_NAMES = [
    "4D_medical_face-mask",
    "Let-green_alcohol_wipes",
    "X-men",
    "aquafina",
    "cart",
    "life-buoy",
    "luong_kho",
    "milo",
    "teppy_orange_juice",
]

LABEL_DIRS = [
    "data/Retail-Product-Checkout-35/train/labels",
    "data/Retail-Product-Checkout-35/valid/labels",
    "data/Retail-Product-Checkout-35/test/labels",
]

IMAGE_DIRS = [
    "data/Retail-Product-Checkout-35/train/images",
    "data/Retail-Product-Checkout-35/val/images",   # (kept from original)
    "data/Retail-Product-Checkout-35/test/images",
]


def count_labels_per_class(label_dirs: List[str], class_names: List[str]) -> pd.DataFrame:
    total_counts = defaultdict(int)
    for labels_dir in label_dirs:
        if not os.path.exists(labels_dir):
            print(f"Warning: Labels directory does not exist: {labels_dir}")
            continue
        for label_file in glob.glob(os.path.join(labels_dir, "*.txt")):
            with open(label_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    class_id = int(parts[0])
                    if 0 <= class_id < len(class_names):
                        total_counts[class_names[class_id]] += 1

    df_counts = pd.DataFrame({"Class": class_names, "Count": [total_counts.get(cls, 0) for cls in class_names]})
    return df_counts


def plot_class_counts(df_counts: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    bars = plt.bar(df_counts['Class'], df_counts['Count'], color='skyblue', edgecolor='black')

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.xlabel("Classes")
    plt.ylabel("Number of labels")
    plt.title("Total Number of Labels Per Class")

    max_count = max(df_counts['Count']) if len(df_counts) else 0
    plt.ylim(0, max_count * 1.15 if max_count else 1)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + (max_count * 0.02 if max_count else 0.1),
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold',
            color='black'
        )

    plt.tight_layout()
    plt.show()


def gather_example_images_per_class(label_dirs: List[str], image_dirs: List[str], class_names: List[str]) -> Dict[int, Dict[str, str]]:
    all_examples_per_class: Dict[int, List[Dict[str, str]]] = {i: [] for i in range(len(class_names))}

    for labels_dir, images_dir in zip(label_dirs, image_dirs):
        for label_file in glob.glob(os.path.join(labels_dir, "*.txt")):
            with open(label_file, "r") as f:
                lines = f.readlines()
            classes_in_file = set()
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                if class_id < len(class_names):
                    classes_in_file.add(class_id)
            base_filename = os.path.basename(label_file).rsplit(".", 1)[0]
            for class_id in classes_in_file:
                image_path = os.path.join(images_dir, base_filename + ".jpg")
                all_examples_per_class[class_id].append({"image_path": image_path, "class_name": class_names[class_id]})

    found_examples = {}
    for class_id, examples in all_examples_per_class.items():
        idx_to_use = 4 if len(examples) > 4 else max(len(examples) - 1, 0)
        if examples:
            found_examples[class_id] = examples[idx_to_use]
    return found_examples


def show_examples_grid(found_examples: Dict[int, Dict[str, str]], class_names: List[str]) -> None:
    num_classes = len(class_names)
    cols = 3
    rows = (num_classes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7.2, rows * 1.9))
    axes = axes.flatten()
    MAX_IMG_HEIGHT = 140

    for idx, class_id in enumerate(sorted(found_examples.keys())):
        ax = axes[idx]
        data = found_examples[class_id]
        img = Image.open(data["image_path"])  # may raise if missing; preserves original behavior
        w, h = img.size
        new_h = MAX_IMG_HEIGHT
        new_w = int(w * (new_h / h))
        img = img.resize((new_w, new_h))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"{data['class_name']}", fontsize=10, fontweight='bold', pad=6, loc='center')

    plt.tight_layout()
    plt.show()

# ==== Training / Benchmarking =================================================

def train_yolo_models(dataset_yaml: str, model_variants: Tuple[str, ...], epochs: int, imgsz: int, batch: int, patience: int) -> None:
    for model_name in model_variants:
        print(f"\n🔹 Training {model_name}...\n")
        model = YOLO(model_name)
        _ = model.train(data=dataset_yaml, epochs=epochs, imgsz=imgsz, batch=batch, patience=patience, plots=True)
        print(f"✅ Finished training {model_name}\n")
    print("🎯 All models trained successfully!")


def benchmark_yolo_models(model_paths: List[str], data_yaml: str) -> Tuple[pd.DataFrame, YOLO]:
    results_list = []
    best_model_path = None
    best_map5095 = -1

    for path in model_paths:
        model_name = os.path.basename(path).replace(".pt", "")
        result = subprocess.run(["yolo", "mode=val", f"model={path}", f"data={data_yaml}"], capture_output=True, text=True)
        output = "\n".join([line for line in (result.stdout + result.stderr).splitlines()
                            if not ("UserWarning" in line or "Cache directory" in line)])
        metrics = {}
        match = re.search(r"all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", output)
        if match:
            p, r, map50, map5095 = map(float, match.groups())
            metrics = {"Model": model_name, "Precision": p, "Recall": r, "mAP@0.5": map50, "mAP@0.5:0.95": map5095}
            results_list.append(metrics)
            if map5095 > best_map5095:
                best_map5095 = map5095
                best_model_path = path

    df = pd.DataFrame(results_list)
    best_model = YOLO(best_model_path) if best_model_path else None
    return df, best_model

# ==== Inference & Post-prediction Analysis ====================================

def get_trimmed_image_name(filename: str) -> str:
    base = os.path.basename(filename)
    base_no_ext = base.split('.')[0]
    parts = base_no_ext.split('_')
    trimmed_name = '_'.join(parts[:2])
    return trimmed_name


def extract_counts_from_detections(detections_list: List[Any], class_names: List[str], image_paths: List[str] | None = None) -> Dict[str, Dict[str, int]]:
    image_counts: Dict[str, Dict[str, int]] = {}
    for i, detections in enumerate(detections_list):
        if not isinstance(detections, np.ndarray):
            detections = np.array(detections.cpu())
            class_ids = detections[:, 5].astype(int)
            counts_dict = {class_names[cid]: count for cid, count in Counter(class_ids).items()}
        else:
            counts_dict = {}
        image_name = get_trimmed_image_name(image_paths[i]) if image_paths and i < len(image_paths) else f"Image_{i}"
        image_counts[image_name] = counts_dict
    return image_counts


DEFAULT_STOCK_THRESHOLDS: Dict[str, int] = {
    "4D_medical_face-mask": 3,
    "Let-green_alcohol_wipes": 2,
    "X-men": 1,
    "aquafina": 4,
    "cart": 1,
    "life-buoy": 2,
    "luong_kho": 1,
    "milo": 3,
    "teppy_orange_juice": 2,
}


def classify_stock_status(product_count: int, total_count_all_products: int, threshold: int) -> str:
    if total_count_all_products == 0:
        return "Out of Stock"
    elif product_count == 0:
        return "Out of Stock"
    elif product_count >= threshold:
        return "OK"
    else:
        return "Below Threshold"


def generate_shelf_status_report(detections_list: List[Any], class_names: List[str], stock_thresholds: Dict[str, int], image_paths: List[str]) -> pd.DataFrame:
    counts_per_image = extract_counts_from_detections(detections_list, class_names, image_paths=image_paths)
    df_rows = []

    for image_name, counts in counts_per_image.items():
        total_count_all_products = sum(counts.values())
        for product, threshold in stock_thresholds.items():
            product_count = counts.get(product, 0)
            status = classify_stock_status(product_count, total_count_all_products, threshold)
            df_rows.append({"Image": image_name, "Product": product, "Count": product_count, "Status": status})

    shelf_status_report_df = pd.DataFrame(df_rows)
    return shelf_status_report_df

# ==== Video Processing =========================================================

def detect_and_save_video(input_path: str, output_path: str, model_path: str, conf_threshold: float = 0.25, frame_dir: str = "detected_frames") -> None:
    model = YOLO(model_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {input_path}")

    os.makedirs(frame_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        cv2.imwrite(os.path.join(frame_dir, f"frame_{frame_count:05d}.jpg"), frame)

    cap.release()

    model.predict(source=frame_dir, conf=conf_threshold, save=True, project=frame_dir, name="preds", exist_ok=True)
    pred_dir = os.path.join(frame_dir, "preds")
    saved_frames = sorted([f for f in os.listdir(pred_dir) if f.lower().endswith(('.jpg'))])

    for frame_file in saved_frames:
        img = cv2.imread(os.path.join(pred_dir, frame_file))
        out.write(cv2.resize(img, (width, height)))

    out.release()
    cv2.destroyAllWindows()
    shutil.rmtree(frame_dir)
    print(f"Processed video saved to: {output_path}")

# ==== Flask App ===============================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>YOLOv8 Multi-Image & Video Detection & Shelf Report</title>
<style>
  body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f0f4f8; color: #222; margin: 0; padding: 20px; }
  h1 { text-align: center; margin-bottom: 30px; color: #0d47a1; }
  .container { max-width: 1200px; margin: 0 auto; display: flex; gap: 40px; flex-wrap: wrap; }
  .panel { background: white; border-radius: 10px; padding: 20px 25px; box-shadow: 0 4px 15px rgb(0 0 0 / 0.1); margin-bottom: 30px; flex: 1 1 400px; }
  h2 { text-align: center; margin-bottom: 20px; color: #0d47a1; }
  .slider-group { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
  .slider-group label { flex: 1 1 130px; font-weight: 600; font-size: 15px; color: #333; }
  .slider-group input[type=range] { flex: 1 1 120px; margin: 0 12px; cursor: pointer; }
  .slider-value { width: 30px; font-weight: 600; color: #0d47a1; text-align: center; }
  .btn-update { display: block; width: 100%; background-color: #0d47a1; color: white; font-size: 16px; padding: 12px 0; border: none; border-radius: 8px; cursor: pointer; margin-top: 25px; transition: background-color 0.3s ease; }
  .btn-update:hover { background-color: #084c8d; }
  form.upload-form { margin-bottom: 30px; display: flex; gap: 15px; justify-content: center; align-items: center; flex-wrap: wrap; }
  form.upload-form input[type="file"] { cursor: pointer; }
  form.upload-form input[type="submit"] { background-color: #0d47a1; border: none; color: white; padding: 10px 25px; font-size: 16px; border-radius: 8px; cursor: pointer; transition: background-color 0.3s ease; }
  form.upload-form input[type="submit"]:hover { background-color: #084c8d; }
  .gallery { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 25px; margin-bottom: 40px; }
  .image-card { background: #fafafa; border-radius: 10px; box-shadow: 0 3px 10px rgb(0 0 0 / 0.12); padding: 15px; text-align: center; }
  .image-card img { max-width: 100%; border-radius: 10px; box-shadow: 0 3px 12px rgb(0 0 0 / 0.15); margin-bottom: 15px; }
  .image-card table { margin: 0 auto; border-collapse: collapse; width: 90%; font-size: 14px; }
  .image-card th, .image-card td { border: 1px solid #ddd; padding: 6px 8px; text-align: center; }
  .image-card th { background-color: #0d47a1; color: white; }
  table.shelf-report { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
  table.shelf-report, table.shelf-report th, table.shelf-report td { border: 1px solid #ddd; }
  table.shelf-report th, table.shelf-report td { padding: 12px 10px; text-align: center; }
  table.shelf-report th { background-color: #0d47a1; color: white; }
  table.shelf-report tr:nth-child(even) { background-color: #f6f9ff; }
  .status-ok { color: green; font-weight: 700; }
  .status-below { color: #d32f2f; font-weight: 700; }
  .status-out { color: #757575; font-style: italic; }
  @media (max-width: 900px) { .container { flex-direction: column; } .panel { flex: 1 1 100%; } }
</style>
</head>
<body>

<h1>AI-Powered Autonomous Retail Shelf Monitoring System</h1>

<div class="container">
  <div class="panel">
    <h2>Adjust Stock Thresholds</h2>
    <form method="POST" id="thresholds-form">
      {% for product, val in thresholds.items() %}
      <div class="slider-group">
        <label for="threshold_{{ loop.index }}">{{ product }}</label>
        <input type="range" min="0" max="10" step="1" id="threshold_{{ loop.index }}" name="threshold_{{ product }}" value="{{ val }}" oninput="document.getElementById('val_{{ loop.index }}').textContent = this.value" />
        <div class="slider-value" id="val_{{ loop.index }}">{{ val }}</div>
      </div>
      {% endfor %}
      <button type="submit" name="set_thresholds" class="btn-update">Update Thresholds & Clear Images</button>
    </form>
  </div>

  <div class="panel">
    <h2>Upload Images for Detection</h2>
    <form method="POST" enctype="multipart/form-data" class="upload-form" action="/">
      {% for product, val in thresholds.items() %}
        <input type="hidden" name="threshold_{{ product }}" value="{{ val }}">
      {% endfor %}
      <input type="file" name="images" accept="image/*" multiple required />
      <input type="submit" value="Upload and Detect" />
    </form>

    <h2>Upload Video for Detection</h2>
    <form method="POST" enctype="multipart/form-data" class="upload-form" action="/upload_video">
      <input type="file" name="video" accept="video/*" required />
      <input type="submit" value="Upload and Process Video" />
    </form>
  </div>
</div>

{% if images_results %}
<div class="container">
  <div class="panel" style="flex: 1 1 100%;">
    <h2>Image Detection Results</h2>
    <div class="gallery">
      {% for img_res in images_results %}
      <div class="image-card">
        <img src="{{ img_res.img_url }}" alt="Result image {{ loop.index }}" />
        <table>
          <thead>
            <tr><th>Product</th><th>Count</th><th>Avg Confidence</th></tr>
          </thead>
          <tbody>
            {% for item in img_res.detections %}
            <tr>
              <td>{{ item.product }}</td>
              <td>{{ item.count }}</td>
              <td>{{ "%.2f"|format(item.avg_confidence) }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
      {% endfor %}
    </div>

    <h2>Consolidated Shelf Status Report</h2>
    <table class="shelf-report">
      <thead>
        <tr>
          <th>Product</th>
          <th>Total Count</th>
          <th>Average Confidence</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        {% for row in consolidated_report %}
        <tr>
          <td>{{ row.product }}</td>
          <td>{{ row.total_count }}</td>
          <td>{{ "%.2f"|format(row.avg_confidence) }}</td>
          <td class="
            {% if row.status == 'OK' %}status-ok
            {% elif row.status == 'Below Threshold' %}status-below
            {% else %}status-out
            {% endif %}
          ">{{ row.status }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</div>
{% endif %}

{% if video_url %}
<div class="container">
  <div class="panel" style="flex: 1 1 100%;">
    <h2>Processed Video Result</h2>
    <video width="720" height="480" controls>
      <source src="{{ video_url }}" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </div>
</div>
{% endif %}

</body>
</html>
"""


def create_app(model_path: str = CFG.APP_MODEL_PATH) -> Flask:
    app = Flask(__name__)

    model = YOLO(model_path)
    class_names_map = model.model.names  # YOLO names mapping

    def _classify_stock_status(product_count: int, total_count_all_products: int, threshold: int) -> str:
        return classify_stock_status(product_count, total_count_all_products, threshold)

    @app.route("/", methods=["GET", "POST"])
    def upload_image():
        thresholds = DEFAULT_STOCK_THRESHOLDS.copy()
        images_results = []
        consolidated_counts: Dict[str, int] = {}
        consolidated_conf_sums: Dict[str, float] = {}

        if request.method == "POST" and "set_thresholds" in request.form:
            new_thresholds: Dict[str, int] = {}
            for product in DEFAULT_STOCK_THRESHOLDS:
                key = f"threshold_{product}"
                val = request.form.get(key)
                try:
                    val_int = int(val)
                    if val_int < 0:
                        val_int = 0
                    new_thresholds[product] = val_int
                except Exception:
                    new_thresholds[product] = DEFAULT_STOCK_THRESHOLDS[product]
            thresholds = new_thresholds
            return render_template_string(HTML_TEMPLATE, thresholds=thresholds)

        if request.method == "POST" and "images" in request.files:
            for product in DEFAULT_STOCK_THRESHOLDS:
                key = f"threshold_{product}"
                val = request.form.get(key)
                if val:
                    try:
                        thresholds[product] = max(0, int(val))
                    except Exception:
                        pass

            files = request.files.getlist("images")
            if not files or len(files) == 0:
                return render_template_string(HTML_TEMPLATE, thresholds=thresholds)

            project_folder = os.path.join(app.root_path, "static", "predictions")
            if os.path.exists(project_folder):
                shutil.rmtree(project_folder)
            os.makedirs(project_folder, exist_ok=True)

            uploads_folder = os.path.join(app.root_path, "static", "image_results")
            os.makedirs(uploads_folder, exist_ok=True)

            detections_list: List[np.ndarray] = []
            saved_image_paths: List[str] = []

            for file in files:
                if file and file.filename != "":
                    with tempfile.TemporaryDirectory() as temp_dir:
                        file_path = os.path.join(temp_dir, file.filename)
                        file.save(file_path)

                        results = model.predict(
                            source=file_path,
                            conf=0.4,
                            save=True,
                            project=project_folder,
                            name="predict",
                            exist_ok=True,
                        )
                        detections = results[0].detach().cpu().numpy() if len(results) > 0 else []
                        detections_list.append(detections)

                        saved_imgs_dir = os.path.join(project_folder, "predict")
                        saved_files = sorted(os.listdir(saved_imgs_dir))
                        saved_img_filename = saved_files[-1]

                        src_path = os.path.join(saved_imgs_dir, saved_img_filename)
                        dst_path = os.path.join(uploads_folder, saved_img_filename)
                        shutil.copy(src_path, dst_path)

                        saved_image_paths.append(f"image_results/{saved_img_filename}")

            shutil.rmtree(project_folder)

            for idx, detections in enumerate(detections_list):
                counts: Dict[str, int] = {}
                conf_sums: Dict[str, float] = {}

                for det in detections:
                    _, _, _, _, conf, cls_id = det
                    cls_id = int(cls_id)
                    cls_name = class_names_map.get(cls_id, str(cls_id))
                    counts[cls_name] = counts.get(cls_name, 0) + 1
                    conf_sums[cls_name] = conf_sums.get(cls_name, 0.0) + conf

                img_detections = []
                for product in thresholds:
                    count = counts.get(product, 0)
                    avg_conf = conf_sums.get(product, 0.0) / count if count > 0 else 0.0
                    img_detections.append({"product": product, "count": count, "avg_confidence": avg_conf})
                    consolidated_counts[product] = consolidated_counts.get(product, 0) + count
                    consolidated_conf_sums[product] = consolidated_conf_sums.get(product, 0.0) + conf_sums.get(product, 0.0)

                images_results.append({
                    "img_url": url_for('static', filename=saved_image_paths[idx].replace(os.path.sep, "/")),
                    "detections": img_detections,
                })

            total_count_all_products = sum(consolidated_counts.values())
            consolidated_report = []
            for product in thresholds:
                total_count = consolidated_counts.get(product, 0)
                total_conf = consolidated_conf_sums.get(product, 0.0)
                avg_conf = (total_conf / total_count) if total_count > 0 else 0.0
                threshold = thresholds[product]
                status = _classify_stock_status(total_count, total_count_all_products, threshold)
                consolidated_report.append({
                    "product": product,
                    "total_count": total_count,
                    "avg_confidence": avg_conf,
                    "status": status,
                })

            return render_template_string(
                HTML_TEMPLATE,
                thresholds=thresholds,
                images_results=images_results,
                consolidated_report=consolidated_report,
            )

        return render_template_string(HTML_TEMPLATE, thresholds=thresholds)

    @app.route('/video/<filename>')
    def serve_video(filename):
        static_folder = os.path.join(app.root_path, "static")
        return send_from_directory(static_folder, filename)

    @app.route("/upload_video", methods=["GET", "POST"])
    def upload_video():
        if request.method == "POST":
            video_file = request.files.get("video")
            if not video_file or video_file.filename == "":
                return "No video file uploaded", 400

            with tempfile.TemporaryDirectory() as temp_dir:
                input_video_path = os.path.join(temp_dir, video_file.filename)
                video_file.save(input_video_path)

                static_folder = os.path.join(app.root_path, "static")
                os.makedirs(static_folder, exist_ok=True)

                output_filename = f"Result_{uuid.uuid4().hex}.mp4"
                output_path = os.path.join(static_folder, output_filename)

                frame_dir = os.path.join(temp_dir, "frames")
                os.makedirs(frame_dir, exist_ok=True)

                cap = cv2.VideoCapture(input_video_path)
                if not cap.isOpened():
                    return "Failed to open uploaded video", 400

                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    frame_path = os.path.join(frame_dir, f"frame_{frame_count:05d}.jpg")
                    cv2.imwrite(frame_path, frame)
                cap.release()

                model.predict(source=frame_dir, conf=0.25, save=True, project=frame_dir, name="preds", exist_ok=True)

                pred_dir = os.path.join(frame_dir, "preds")
                saved_frames = sorted([f for f in os.listdir(pred_dir) if f.lower().endswith('.jpg')])

                cap = cv2.VideoCapture(input_video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                fourcc = cv2.VideoWriter_fourcc(*"X264")
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                for frame_file in saved_frames:
                    img = cv2.imread(os.path.join(pred_dir, frame_file))
                    if img.shape[1] != width or img.shape[0] != height:
                        img = cv2.resize(img, (width, height))
                    out.write(img)
                out.release()

                print(f"Video saved at: {output_path}")
                video_url = url_for("static", filename=output_filename)
                print(f"Video URL: {video_url}")

                shutil.rmtree(frame_dir)

            return render_template_string(HTML_TEMPLATE, thresholds=DEFAULT_STOCK_THRESHOLDS, video_url=video_url)

        return """
        <h1>Upload Video for Detection</h1>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="video" accept="video/*" required />
            <input type="submit" value="Upload and Process Video" />
        </form>
        """

    return app

# ==== Dockerfile Writer =======================================================

DOCKERFILE_CONTENT = """\
# Use Python
FROM python:3.8-slim-bullseye

# Prevent Python output buffering
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install static build of ffmpeg with libx264 included
RUN wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
    && tar -xJf ffmpeg-release-amd64-static.tar.xz \
    && cp ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/ffmpeg \
    && cp ffmpeg-*-amd64-static/ffprobe /usr/local/bin/ffprobe \
    && rm -rf ffmpeg-release-amd64-static.tar.xz ffmpeg-*-amd64-static


# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir torch==1.10.2+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project
COPY . .

# Expose port
EXPOSE 5000

# Run directly with Python
CMD ["python", "run.py"]
"""


def write_dockerfile(path: str = CFG.DOCKERFILE_NAME) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(DOCKERFILE_CONTENT)
    print("Dockerfile created successfully.")

# ==== Stage Orchestration =====================================================

def stage_all(dataset) -> None:
    # Counts + plots
    df_counts = count_labels_per_class(LABEL_DIRS, CLASS_NAMES)
    plot_class_counts(df_counts)

    # Example grid
    examples = gather_example_images_per_class(LABEL_DIRS, IMAGE_DIRS, CLASS_NAMES)
    if examples:
        show_examples_grid(examples, CLASS_NAMES)

    # Train
    train_yolo_models(
        dataset_yaml=os.path.join(dataset.location, "data.yaml"),
        model_variants=CFG.MODEL_VARIANTS,
        epochs=CFG.EPOCHS,
        imgsz=CFG.IMGSZ,
        batch=CFG.BATCH,
        patience=CFG.PATIENCE,
    )

    # Benchmark
    df, model = benchmark_yolo_models(list(CFG.MODEL_FILES), data_yaml=os.path.join(dataset.location, "data.yaml"))
    if model is not None:
        print("\n🏆 Best Model Summary 🏆")
        print("="*40)
        best_model_row = df.loc[df['mAP@0.5:0.95'].idxmax()]
        print(f"Model: {best_model_row['Model']}")
        print(f"mAP@0.5: {best_model_row['mAP@0.5']:.4f}")
        print(f"mAP@0.5:0.95: {best_model_row['mAP@0.5:0.95']:.4f}")
        print(f"Precision: {best_model_row['Precision']:.4f}")
        print(f"Recall: {best_model_row['Recall']:.4f}")
        print(f"Reason: Highest mAP@0.5:0.95 (better overall detection performance)")
        print("="*40)
        print(df.head())

        # Inference on test images (saves annotated outputs)
        _ = model.predict(source=os.path.join(dataset.location, "test/images"), conf=CFG.CONF_THRESHOLD, save=True, save_txt=True)

        # Display a few predictions from runs/detect/predict
        image_paths = glob.glob('runs/detect/predict/*.jpg')
        num_images = 6
        selected_paths = random.sample(image_paths, min(num_images, len(image_paths)))
        cols = 3
        rows = (len(selected_paths) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten()
        for ax, img_path in zip(axes, selected_paths):
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

        # Post-prediction report for test set
        class_names = model.model.names
        test_image_paths = sorted(glob.glob(os.path.join(dataset.location, "test/images", "*.*")))
        results = model.predict(source=os.path.join(dataset.location, "test/images"), conf=CFG.CONF_THRESHOLD)
        df_report = generate_shelf_status_report(results, class_names, DEFAULT_STOCK_THRESHOLDS, test_image_paths)
        print(df_report.head())


def stage_train(dataset) -> None:
    train_yolo_models(
        dataset_yaml=os.path.join(dataset.location, "data.yaml"),
        model_variants=CFG.MODEL_VARIANTS,
        epochs=CFG.EPOCHS,
        imgsz=CFG.IMGSZ,
        batch=CFG.BATCH,
        patience=CFG.PATIENCE,
    )


def stage_benchmark(dataset) -> None:
    df, model = benchmark_yolo_models(list(CFG.MODEL_FILES), data_yaml=os.path.join(dataset.location, "data.yaml"))
    if model is not None and not df.empty:
        best_model_row = df.loc[df['mAP@0.5:0.95'].idxmax()]
        print("\n🏆 Best Model Summary 🏆")
        print("="*40)
        print(f"Model: {best_model_row['Model']}")
        print(f"mAP@0.5: {best_model_row['mAP@0.5']:.4f}")
        print(f"mAP@0.5:0.95: {best_model_row['mAP@0.5:0.95']:.4f}")
        print(f"Precision: {best_model_row['Precision']:.4f}")
        print(f"Recall: {best_model_row['Recall']:.4f}")
        print(f"Reason: Highest mAP@0.5:0.95 (better overall detection performance)")
        print("="*40)
        print(df.head())


def stage_infer_test(dataset) -> None:
    # Requires a trained/best model path (user must update CFG.APP_MODEL_PATH or provide via CLI)
    model = YOLO(CFG.APP_MODEL_PATH)
    _ = model.predict(source=os.path.join(dataset.location, "test/images"), conf=CFG.CONF_THRESHOLD, save=True, save_txt=True)

    image_paths = glob.glob('runs/detect/predict/*.jpg')
    if image_paths:
        num_images = 6
        selected_paths = random.sample(image_paths, min(num_images, len(image_paths)))
        cols = 3
        rows = (len(selected_paths) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten()
        for ax, img_path in zip(axes, selected_paths):
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.show()


def stage_video(input_video: str, output_video: str | None) -> None:
    if output_video is None:
        os.makedirs(CFG.OUTPUT_VIDEO_DIR, exist_ok=True)
        output_video = os.path.join(CFG.OUTPUT_VIDEO_DIR, "output_detected.mp4")
    detect_and_save_video(
        input_path=input_video,
        output_path=output_video,
        model_path=CFG.APP_MODEL_PATH,
    )


def stage_app() -> None:
    app = create_app(CFG.APP_MODEL_PATH)
    app.run(debug=True, use_reloader=False)


def stage_dockerfile() -> None:
    write_dockerfile(CFG.DOCKERFILE_NAME)

# ==== CLI ====================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retail Shelf Monitoring Pipeline")
    p.add_argument("--stage", choices=["all", "train", "benchmark", "infer_test", "video", "app", "dockerfile"], default="app")
    p.add_argument("--input_video", type=str, default=None, help="Path to input video for --stage video")
    p.add_argument("--output_video", type=str, default=None, help="Output video path for --stage video")
    return p.parse_args()


def main():
    args = parse_args()

    if args.stage in {"all", "train", "benchmark", "infer_test"}:
        dataset = get_or_download_dataset(CFG)

    if args.stage == "all":
        stage_all(dataset)
    elif args.stage == "train":
        stage_train(dataset)
    elif args.stage == "benchmark":
        stage_benchmark(dataset)
    elif args.stage == "infer_test":
        stage_infer_test(dataset)
    elif args.stage == "video":
        if not args.input_video:
            raise SystemExit("--input_video is required for --stage video")
        stage_video(args.input_video, args.output_video)
    elif args.stage == "app":
        stage_app()
    elif args.stage == "dockerfile":
        stage_dockerfile()


if __name__ == "__main__":
    main()