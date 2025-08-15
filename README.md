# Retail Shelf Monitoring Project
## Overview

The Retail Shelf Monitoring Project is a full-stack AI solution for automated shelf monitoring in retail stores. It uses computer vision and deep learning models to detect stock levels, misplaced items, and empty shelves in real-time. This project helps retailers reduce stockouts, improve operational efficiency, and enhance customer satisfaction.

## Directory Structure
Retail-Shelf-Monitoring-Project/
│
├── app/                         # Application source code (API, backend, or scripts)
│   └── ...
├── data/                        # Dataset
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
├── models/                      # Pre-trained YOLO models
│   ├── yolov8n_best.pt
│   ├── yolov8s_best.pt
│   └── yolov8m_best.pt
├── venv/                        # Python virtual environment (ignored in Git)
├── Dockerfile                    # Docker configuration for containerization
├── run.py                        # Main script to run the AI monitoring system
├── requirements.txt              # Python dependencies
├── retail_shelf_monitor.tar      # Large dataset/file (ignored in Git)
├── task-definition.json          # Cloud deployment task definitions
└── trust-policy.json             # Cloud IAM/trust policies

## Features

Automated shelf monitoring using YOLOv8 models.

Detects stock levels, misplaced products, and empty shelves.

Supports training, validation, and testing datasets.

Easily deployable with Docker.

Scalable for large retail environments.

Setup Instructions

Clone the repository:

git clone https://github.com/javaid100/Retail-Shelf-Monitoring-Project.git
cd Retail-Shelf-Monitoring-Project


Create and activate virtual environment:

python -m venv venv
venv\Scripts\activate       # On Windows


Install dependencies:

pip install -r requirements.txt


Run the main script:

python run.py

Notes

The venv/ folder and large files like retail_shelf_monitor.tar are ignored in Git.

Use your own dataset if needed. The structure under data/ should remain consistent.

Docker can be used to containerize and deploy the system for production.
