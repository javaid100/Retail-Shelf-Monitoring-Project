import os
import shutil
import tempfile
import subprocess
import uuid
import cv2
from ultralytics import YOLO
from flask import current_app, url_for
from app.config import MODEL_PATH, PREDICTIONS_FOLDER, IMAGE_RESULTS_FOLDER
from app.utils import classify_stock_status


# Load YOLO model
model = YOLO(MODEL_PATH)
class_names = model.model.names


def detect_images(files, thresholds):
    if os.path.exists(PREDICTIONS_FOLDER):
        shutil.rmtree(PREDICTIONS_FOLDER)
    os.makedirs(PREDICTIONS_FOLDER, exist_ok=True)
    os.makedirs(IMAGE_RESULTS_FOLDER, exist_ok=True)

    detections_list = []
    saved_image_paths = []
    consolidated_counts = {}
    consolidated_conf_sums = {}

    for file in files:
        if file and file.filename != "":
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, file.filename)
                file.save(file_path)

                results = model.predict(
                    source=file_path,
                    conf=0.4,
                    save=True,
                    project=PREDICTIONS_FOLDER,
                    name="predict",
                    exist_ok=True
                )
                detections = results[0].detach().cpu().numpy() if len(results) > 0 else []
                detections_list.append(detections)

                saved_imgs_dir = os.path.join(PREDICTIONS_FOLDER, "predict")
                saved_files = sorted(os.listdir(saved_imgs_dir))
                saved_img_filename = saved_files[-1]

                src_path = os.path.join(saved_imgs_dir, saved_img_filename)
                dst_path = os.path.join(IMAGE_RESULTS_FOLDER, saved_img_filename)
                shutil.copy(src_path, dst_path)

                saved_image_paths.append(f"image_results/{saved_img_filename}")

    shutil.rmtree(PREDICTIONS_FOLDER)

    images_results = []
    for idx, detections in enumerate(detections_list):
        counts = {}
        conf_sums = {}
        for det in detections:
            _, _, _, _, conf, cls_id = det
            cls_id = int(cls_id)
            cls_name = class_names.get(cls_id, str(cls_id))
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
            "detections": img_detections
        })

    total_count_all_products = sum(consolidated_counts.values())
    consolidated_report = []
    for product in thresholds:
        total_count = consolidated_counts.get(product, 0)
        total_conf = consolidated_conf_sums.get(product, 0.0)
        avg_conf = (total_conf / total_count) if total_count > 0 else 0.0
        threshold = thresholds[product]
        status = classify_stock_status(total_count, total_count_all_products, threshold)
        consolidated_report.append({
            "product": product,
            "total_count": total_count,
            "avg_confidence": avg_conf,
            "status": status
        })

    return images_results, consolidated_report


def detect_video(video_file):
    """Process uploaded video with YOLO, return relative URL to processed MP4."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded file
        input_video_path = os.path.join(temp_dir, video_file.filename)
        video_file.save(input_video_path)

        # Prepare output paths
        static_videos = os.path.join(current_app.root_path, "static", "videos")
        os.makedirs(static_videos, exist_ok=True)
        output_filename = f"Result_{uuid.uuid4().hex}.mp4"
        output_path = os.path.join(static_videos, output_filename)

        # Extract frames
        frame_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frame_dir, exist_ok=True)
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError("Failed to open uploaded video")

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            frame_path = os.path.join(frame_dir, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(frame_path, frame)
        cap.release()

        # Run YOLO on frames
        model.predict(source=frame_dir, conf=0.25, save=True, project=frame_dir, name="preds", exist_ok=True)
        pred_dir = os.path.join(frame_dir, "preds")
        saved_frames = sorted([f for f in os.listdir(pred_dir) if f.lower().endswith('.jpg')])

        # Video parameters
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Step 1: Write MJPG AVI
        avi_path = os.path.join(temp_dir, "temp_output.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(avi_path, fourcc, fps, (width, height))
        for frame_file in saved_frames:
            img = cv2.imread(os.path.join(pred_dir, frame_file))
            if img.shape[1] != width or img.shape[0] != height:
                img = cv2.resize(img, (width, height))
            out.write(img)
        out.release()

        # Step 2: Convert AVI → MP4 with libx264
        subprocess.run([
            "ffmpeg", "-y", "-i", avi_path,
            "-vcodec", "libx264", "-crf", "23", "-preset", "fast",
            output_path
        ], check=True)

        # Cleanup
        shutil.rmtree(frame_dir)

        return url_for("static", filename=f"videos/{output_filename}")