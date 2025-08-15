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
