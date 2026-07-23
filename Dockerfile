FROM python:3.11-slim

# poppler-utils gives us pdftoppm, pdfimages, pdftotext
RUN apt-get update && apt-get install -y poppler-utils && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY qbank_pipeline.py app.py ./
# PDFs get uploaded via the dashboard now, so no need to bake them into the image.
# (If you'd rather pre-load them at build time, uncomment the next line and
#  add a pdfs/ folder next to this Dockerfile.)
# COPY pdfs/ ./pdfs/

# /data is where the Railway Volume will be mounted (persistent across restarts)
ENV OUTPUT_DIR=/data/qbank_output
ENV STATE_PATH=/data/qbank_output/state.json

EXPOSE 8080
CMD ["python3", "app.py"]
