# 🛒 Retail Shelf Monitoring Project

## Overview
The Retail Shelf Monitoring Project is a full-stack AI solution for automated shelf monitoring in retail stores. Using computer vision and deep learning models, it can detect stock levels, misplaced items, and empty shelves.  

**Benefits:**
- Reduce stockouts
- Improve operational efficiency
- Enhance customer satisfaction

## Directory Structure
Retail-Shelf-Monitoring-Project/
├── app/  
│   └── ...  
├── data/  
│   ├── train/  
│   │   ├── images/  
│   │   └── labels/  
│   ├── valid/  
│   │   ├── images/  
│   │   └── labels/  
│   └── test/  
│       ├── images/  
│       └── labels/  
├── models/  
│   ├── yolov8n_best.pt  
│   ├── yolov8s_best.pt  
│   └── yolov8m_best.pt  
├── venv/  
├── Dockerfile  
├── run.py  
├── requirements.txt  
├── task-definition.json  
└── trust-policy.json  

## Features
- Automated shelf monitoring using YOLOv8 models
- Detect stock levels, and empty shelves
- Supports training, validation, and testing datasets
- Easily deployable with Docker
- Scalable for large retail environments

## Setup Instructions
1. Clone the repository:
   git clone https://github.com/javaid100/Retail-Shelf-Monitoring-Project.git
   cd Retail-Shelf-Monitoring-Project
2. Create and activate virtual environment:
   python -m venv venv
3. Install dependencies:
   pip install -r requirements.txt
4. Run the main script:
   python run.py

## Docker Deployment
- Build Docker image:
  docker build -t retail-shelf-monitor .
- Run Docker container:
  docker run -it --rm retail-shelf-monitor

## Key Scripts
- run.py — Main script to start AI monitoring
- app/ — Contains backend APIs or scripts for processing
- models/ — Pre-trained YOLOv8 weights for inference

## Best Practices
- Keep datasets structured under data/ for training and evaluation
- Regularly update YOLOv8 models for improved detection accuracy
- Use cloud deployment with task definitions in task-definition.json for scalability
- Manage permissions using trust-policy.json for secure access

> Built with ❤️ using Python, YOLOv8, and Docker
