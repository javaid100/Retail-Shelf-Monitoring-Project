import os

# Base paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_FOLDER = os.path.join(BASE_DIR, "static")
IMAGE_RESULTS_FOLDER = os.path.join(STATIC_FOLDER, "image_results")
PREDICTIONS_FOLDER = os.path.join(STATIC_FOLDER, "predictions")
VIDEOS_FOLDER = os.path.join(STATIC_FOLDER, "videos")

# Default stock thresholds
DEFAULT_STOCK_THRESHOLDS = {
    "4D_medical_face-mask": 3,
    "Let-green_alcohol_wipes": 2,
    "X-men": 1,
    "aquafina": 4,
    "cart": 1,
    "life-buoy": 2,
    "luong_kho": 1,
    "milo": 3,
    "teppy_orange_juice": 2
}

# YOLO model path
MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "models", "yolov8m_best.pt")
