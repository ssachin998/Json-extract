FROM python:3.11-slim

# poppler-utils gives us pdftoppm, pdfimages, pdftotext
RUN apt-get update && apt-get install -y poppler-utils && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY qbank_pipeline.py qbank_validator.py fix_output.py app.py test_v2_chapter.py ./
# PDFs get uploaded via the dashboard now, so no need to bake them into the image.
# (If you'd rather pre-load them at build time, uncomment the next line and
#  add a pdfs/ folder next to this Dockerfile.)
# COPY pdfs/ ./pdfs/

# V2: SEPARATE output root on the Railway Volume -- v1 data (/data/qbank_output)
# stays untouched while v2 is being proven.
ENV OUTPUT_DIR=/data/qbank_output_v2
# Flush every print() immediately, otherwise Docker block-buffers stdout and
# pipeline progress never shows up in Railway's Deploy Logs in real time.
ENV PYTHONUNBUFFERED=1

EXPOSE 8080
CMD ["python3", "app.py"]
