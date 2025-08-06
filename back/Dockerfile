FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# System-level dependencies including wget for downloading models
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libgtk-3-0 \
    libavcodec58 \
    libavformat58 \
    libswscale5 \
    ffmpeg \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy your app code first
COPY . .

# Create models directory
RUN mkdir -p /app/models

# Download Places365 model and categories
RUN wget -O /app/models/resnet18_places365.pth.tar \
    "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"

RUN wget -O /app/models/categories_places365.txt \
    "https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt"

# Create model download script
COPY <<EOF /app/download_models.py
import sys
import os

print("Starting model downloads...")

# Download YOLO model
try:
    from ultralytics import YOLO
    print("Pre-downloading YOLOv8 model...")
    model = YOLO('yolov8n.pt')
    print("YOLOv8 model downloaded successfully")
except Exception as e:
    print(f"Warning: Could not pre-download YOLO model: {e}")

# Download SentenceTransformer model
try:
    from sentence_transformers import SentenceTransformer
    print("Pre-downloading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("SentenceTransformer model downloaded successfully")
except Exception as e:
    print(f"Warning: Could not pre-download SentenceTransformer model: {e}")

# Download MiDaS model
try:
    import torch
    print("Pre-downloading MiDaS model...")
    model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', pretrained=True)
    print("MiDaS model downloaded successfully")
except Exception as e:
    print(f"Warning: Could not pre-download MiDaS model: {e}")

print("Model download script completed")
EOF

# Run the model download script
RUN python3 /app/download_models.py

# Set proper permissions
RUN chmod -R 755 /app

# Cloud Run expects app on port 8080
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

# Run using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]