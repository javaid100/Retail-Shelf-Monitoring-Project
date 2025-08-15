from flask import Blueprint, render_template, request
from app.config import DEFAULT_STOCK_THRESHOLDS
from app.detection import detect_images, detect_video

main_bp = Blueprint("main", __name__)

@main_bp.route("/", methods=["GET", "POST"])
def index():
    thresholds = DEFAULT_STOCK_THRESHOLDS.copy()
    images_results = []
    consolidated_report = []
    video_url = None

    if request.method == "POST":
        if "set_thresholds" in request.form:
            for product in thresholds:
                val = request.form.get(f"threshold_{product}")
                thresholds[product] = max(0, int(val)) if val else thresholds[product]
            return render_template("index.html", thresholds=thresholds)

        if "images" in request.files:
            files = request.files.getlist("images")
            images_results, consolidated_report = detect_images(files, thresholds)
            return render_template("index.html", thresholds=thresholds, images_results=images_results, consolidated_report=consolidated_report)

    return render_template("index.html", thresholds=thresholds)


@main_bp.route("/upload_video", methods=["GET", "POST"])
def upload_video():
    if request.method == "POST":
        video_file = request.files.get("video")
        if not video_file or video_file.filename == "":
            return "No video file uploaded", 400
        try:
            video_url = detect_video(video_file)
            return render_template("index.html", thresholds=DEFAULT_STOCK_THRESHOLDS, video_url=video_url)
        except Exception as e:
            return f"Error processing video: {e}", 500

    return render_template("index.html", thresholds=DEFAULT_STOCK_THRESHOLDS)