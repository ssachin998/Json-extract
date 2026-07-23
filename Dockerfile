# QBank PDF → JSON Extraction Pipeline Dockerfile
# ==============================================
# Deploys as a self-configuring Web Dashboard on Railway, Render, or any Docker-based platform

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY qbank_pipeline.py .
COPY app.py .

# Create output directories
RUN mkdir -p data assets/questions assets/options assets/solutions assets/tables

# Expose Web Interface Port
EXPOSE 8080

# Default command starts the web dashboard
CMD ["python", "app.py"]
