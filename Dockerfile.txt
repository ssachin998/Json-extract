# QBank PDF → JSON Extraction Pipeline Dockerfile
# ==============================================
# Use this to deploy to Railway, Render, or any Docker-based platform

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Tesseract language data (English)
RUN apt-get install -y tesseract-ocr-eng

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the pipeline script
COPY qbank_pipeline.py .

# Create output directories
RUN mkdir -p data assets/questions assets/options assets/solutions assets/tables

# Set working directory
WORKDIR /app

# Default command (override with your PDF and subject code)
# Example: docker run -v $(pwd):/app qbank-pipeline Psychology_QBank.pdf PSY
CMD ["python", "qbank_pipeline.py", "input.pdf", "SUB"]
