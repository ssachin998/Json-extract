# QBank PDF → JSON Extraction Pipeline

## 🎯 Overview

This is a **fully automated pipeline** that converts MCQ PDFs (even with broken text layers) into structured JSON format with extracted images.

**Follows the exact specification** from `qbank_extraction_pipeline.md`

---

## ✨ Features

- ✅ **Handles broken text layers** (common in pirated PDFs)
- ✅ **Extracts embedded images** (excluding watermarks)
- ✅ **Uses OCR** (tesseract) for text extraction
- ✅ **Follows exact JSON schema** from specification
- ✅ **Proper folder structure** with subject subfolders
- ✅ **Image naming convention**: `{SUBJECT}/{id}_{TYPE}_01.webp`
- ✅ **Validation** of all outputs
- ✅ **Works on Railway/Render** (Docker-based deployment)

---

## 📦 Quick Start

### Local Installation

```bash
# 1. Clone or download these files:
#    - qbank_pipeline.py
#    - Dockerfile
#    - requirements.txt
#    - DEPLOYMENT_GUIDE.md

# 2. Install system dependencies
# Ubuntu/Debian:
sudo apt-get install poppler-utils tesseract-ocr libtesseract-dev

# Mac:
brew install poppler tesseract

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Run the pipeline
python qbank_pipeline.py your_file.pdf SUBJECT_CODE
```

### Docker (Recommended)

```bash
# Build the image
docker build -t qbank-pipeline .

# Run with your PDF (replace paths)
docker run -v $(pwd):/app qbank-pipeline Psychology_QBank.pdf PSY

# Or with specific output directory
docker run -v $(pwd)/input:/app -v $(pwd)/output:/app/data qbank-pipeline input.pdf PSY
```

---

## 🚀 Usage

### Basic Command

```bash
python qbank_pipeline.py <pdf_path> <subject_code>
```

### Examples

```bash
# Process Psychology QBank
python qbank_pipeline.py Psychology_QBank.pdf PSY

# Process Biology QBank
python qbank_pipeline.py Biology_QBank.pdf BIO

# Process with custom DPI (higher quality, slower)
# Edit DPI = 300 in qbank_pipeline.py
```

### Subject Codes

Use **3 uppercase letters** for subject codes:
- `PSY` - Psychology
- `BIO` - Biology
- `CHE` - Chemistry
- `PHY` - Physics
- `ANA` - Anatomy
- `PHM` - Pharmacology
- etc.

---

## 📁 Output Structure

```
.
├── data/
│   ├── chapters.json          # Chapter metadata
│   └── questions.jsonl        # All questions (JSONL format)
│
└── assets/
    ├── questions/
    │   └── {SUBJECT}/
    │       ├── {SUBJECT}-001-001_Q_01.webp
    │       └── ...
    ├── options/
    │   └── {SUBJECT}/
    │       └── {SUBJECT}-001-001_OPT_A_01.webp
    ├── solutions/
    │   └── {SUBJECT}/
    │       └── {SUBJECT}-001-001_SOL_01.webp
    └── tables/
        └── {SUBJECT}/
            └── {SUBJECT}-001-007_TABLE_01.webp
```

---

## 🔧 JSON Schema

### chapters.json

```json
{
  "chapter_id": "PSY-001",
  "subject": "PSY",
  "chapter_no": 1,
  "chapter_title": "Theories of Personality & Defense Mechanisms"
}
```

### questions.jsonl (one JSON object per line)

```json
{
  "id": "PSY-001-001",
  "subject": "PSY",
  "chapter_id": "PSY-001",
  "question": {
    "text": "Which of the following is NOT a Freudian defense mechanism?",
    "images": [{"type": "portrait", "file": "PSY/PSY-001-001_Q_01.webp"}]
  },
  "options": [
    {"id": "A", "text": "Repression", "images": []},
    {"id": "B", "text": "Sublimation", "images": []},
    {"id": "C", "text": "Projection", "images": []},
    {"id": "D", "text": "Photosynthesis", "images": []}
  ],
  "correct_options": ["D"],
  "solution": {
    "text": "Photosynthesis is a biological process, not a defense mechanism. Freud identified repression, sublimation, and projection as defense mechanisms.",
    "images": [],
    "tables": [
      {
        "type": "psychosexual_stages",
        "markdown": "| Phase | Age | Focus |\n|---|---|---|\n| Oral | 0-1 | Mouth |\n| Anal | 1-3 | Anus |\n| Phallic | 3-6 | Genitals |\n| Latency | 6-puberty | Social skills |\n| Genital | Puberty+ | Sexual maturity |",
        "file": null
      }
    ]
  },
  "tags": ["freud", "psychoanalysis", "defense-mechanisms"]
}
```

---

## 🛠️ Tools Used

| Tool | Purpose | Installation |
|------|---------|--------------|
| `poppler-utils` | PDF analysis, rasterization, text extraction | `apt-get install poppler-utils` |
| `pypdf` | PDF resource inspection, image detection | `pip install pypdf` |
| `tesseract` | OCR text extraction | `apt-get install tesseract-ocr` |
| `Pillow` | Image conversion (PNG → WEBP) | `pip install Pillow` |
| `pytesseract` | Python wrapper for tesseract | `pip install pytesseract` |

---

## 📊 Pipeline Steps

The script automatically performs all 7 steps from the specification:

1. **Map real page numbers** - Analyzes PDF structure, checks for offsets
2. **Rasterize pages** - Converts PDF pages to images at 150 DPI
3. **Find embedded images** - Inspects resource dictionaries per page
4. **Extract real images** - Extracts images, excluding reused watermarks
5. **Read content** - Uses OCR to extract text from rasterized pages
6. **Assemble JSON** - Creates structured `chapters.json` and `questions.jsonl`
7. **Validate** - Checks all JSON format and image references

---

## ⚠️ Limitations

### OCR Quality
- **Tesseract OCR** works well for:
  - Paragraph text
  - Simple layouts
  - Clean, high-resolution PDFs

- **Tesseract struggles with:**
  - Complex multi-column layouts
  - Tables (structure is lost)
  - Mathematical equations
  - Very low-quality scans

### For Best Accuracy
For complex tables and layouts, **use a vision model**:
1. The script will generate rasterized page images
2. Feed these images to a vision-capable AI (like me!)
3. Manually update the JSON with the structured data

---

## 🌐 Cloud Deployment

### Railway
- Free tier available
- Easy file upload/download
- See `DEPLOYMENT_GUIDE.md` for detailed instructions

### Render
- Free tier available
- Background worker support
- See `DEPLOYMENT_GUIDE.md` for detailed instructions

### Other Platforms
Works on any platform that supports:
- Docker containers
- Python 3.8+
- poppler-utils and tesseract

---

## 📚 Documentation

- **Pipeline Specification**: `qbank_extraction_pipeline.md`
- **Deployment Guide**: `DEPLOYMENT_GUIDE.md`
- **Script**: `qbank_pipeline.py`

---

## 🎓 Example Workflow

### Processing a Single PDF

```bash
# 1. Install dependencies
pip install -r requirements.txt
sudo apt-get install poppler-utils tesseract-ocr

# 2. Run pipeline
python qbank_pipeline.py Psychology_QBank.pdf PSY

# 3. Check output
tail -n 5 data/questions.jsonl
ls assets/questions/PSY/

# 4. Validate
python -c "
import json
with open('data/questions.jsonl') as f:
    for line in f:
        json.loads(line)  # Will error on invalid JSON
print('All JSON is valid!')
"
```

### Processing Multiple Subjects

```bash
#!/bin/bash

# Process all subjects
python qbank_pipeline.py Psychology.pdf PSY
python qbank_pipeline.py Biology.pdf BIO
python qbank_pipeline.py Chemistry.pdf CHE

# Combine all questions into one file
cat data/questions.jsonl >> all_questions.jsonl
```

---

## 🔍 Troubleshooting

### Common Errors

**Error: `pdftoppm not found`**
```bash
sudo apt-get install poppler-utils
```

**Error: `tesseract not found`**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng
```

**Error: `No module named 'pypdf'`**
```bash
pip install pypdf
```

**Error: `pytesseract not found`**
```bash
pip install pytesseract
```

### Performance Issues

**Slow processing for large PDFs**
- Increase DPI in script: `DPI = 300` → better quality but slower
- Process in smaller batches
- Use more powerful hardware

**Memory issues**
- Process one chapter at a time
- Use Docker with more memory allocation

---

## 💡 Tips

1. **Start small**: Test with a 10-page PDF first
2. **Verify output**: Check the first few questions manually
3. **Backup PDFs**: The pipeline doesn't modify originals
4. **Use subject folders**: Prevents filename collisions
5. **Validate always**: Run validation before considering a batch done

---

## 📞 Support

For issues with:
- **Dependencies**: Check installation guides for your OS
- **Pipeline logic**: Refer to `qbank_extraction_pipeline.md`
- **Deployment**: See `DEPLOYMENT_GUIDE.md`
- **OCR accuracy**: Consider using vision model for complex pages

---

## 🎯 ID Format

All IDs follow this pattern:
```
{SUBJECT}-{CHAPTER:03d}-{QUESTION:03d}
```

Examples:
- `PSY-001-001` - Psychology, Chapter 1, Question 1
- `PSY-001-042` - Psychology, Chapter 1, Question 42
- `BIO-005-001` - Biology, Chapter 5, Question 1

---

## 🖼️ Image Naming Convention

All image paths follow this pattern:
```
{SUBJECT}/{ID}_{TYPE}_01.webp
```

| Type | Prefix | Example |
|------|--------|---------|
| Question image | `Q` | `PSY/PSY-001-001_Q_01.webp` |
| Option image | `OPT_{ID}` | `PSY/PSY-001-001_OPT_A_01.webp` |
| Solution image | `SOL` | `PSY/PSY-001-001_SOL_01.webp` |
| Table image | `TABLE` | `PSY/PSY-001-007_TABLE_01.webp` |

---

## ✅ Validation Rules

The script automatically validates:
1. ✅ All JSON is valid
2. ✅ All IDs match pattern `{SUBJECT}-\d{3}-\d{3}`
3. ✅ All image paths match pattern `{SUBJECT}/{ID}_{TYPE}_\d{2}.webp`
4. ✅ All referenced image files exist
5. ✅ No orphaned image files (files not referenced in JSON)

---

## 📈 Scaling to Multiple PDFs

For processing 20+ PDFs:

1. **Create a processing script**:
```bash
#!/bin/bash
SUBJECTS=("PSY" "BIO" "CHE" "PHY" "ANA" "PHM")
PDFs=("Psychology.pdf" "Biology.pdf" "Chemistry.pdf" "Physics.pdf" "Anatomy.pdf" "Pharmacology.pdf")

for i in {0..5}; do
    echo "Processing ${PDFs[$i]} as ${SUBJECTS[$i]}..."
    python qbank_pipeline.py "${PDFs[$i]}" "${SUBJECTS[$i]}"
done
```

2. **Or use parallel processing**:
```bash
# Process 4 PDFs in parallel
python qbank_pipeline.py Psychology.pdf PSY &
python qbank_pipeline.py Biology.pdf BIO &
python qbank_pipeline.py Chemistry.pdf CHE &
python qbank_pipeline.py Physics.pdf PHY &
wait
```

3. **Spot-check**: Always manually verify 5-10 random questions from each PDF

---

## 🏆 Best Practices

1. **Quality first**: Verify first chapter before batch processing
2. **Consistency**: Use same subject codes across all PDFs
3. **Organization**: Keep each subject in its own folder
4. **Backup**: Always backup original PDFs
5. **Documentation**: Keep notes on any manual corrections made

---

## 🎓 Understanding the Pipeline

For a deep dive into how the pipeline works, see:
- `qbank_extraction_pipeline.md` - The original specification
- Comments in `qbank_pipeline.py` - Detailed implementation notes

---

## 📝 License

This pipeline is provided as-is for your use. No warranty is given.

---

## 🙏 Credits

- **Pipeline Design**: Based on the specification in `qbank_extraction_pipeline.md`
- **Tools**: poppler-utils, tesseract, pypdf, Pillow
- **OCR**: Tesseract OCR engine
