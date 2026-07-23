# QBank Pipeline - Deployment Guide

## 📦 Quick Start (Local Machine)

### Prerequisites
- Python 3.8+
- Docker (optional, for containerized deployment)

### 1. Install Dependencies

#### On Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y poppler-utils tesseract-ocr libtesseract-dev libleptonica-dev
```

#### On Mac (Homebrew):
```bash
brew install poppler tesseract
```

#### Python Packages:
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
# Basic usage
python qbank_pipeline.py your_file.pdf SUBJECT_CODE

# Example: Process Psychology QBank
python qbank_pipeline.py Psychology_QBank.pdf PSY

# Example: Process Biology QBank
python qbank_pipeline.py Biology_QBank.pdf BIO
```

### 3. Output Files

After processing, you'll have:
```
data/
├── chapters.json      # Chapter metadata
└── questions.jsonl    # All questions (one JSON object per line)

assets/
├── questions/
│   └── PSY/           # Subject folder
│       ├── PSY-001-001_Q_01.webp
│       └── ...
├── options/
│   └── PSY/
├── solutions/
│   └── PSY/
└── tables/
    └── PSY/
```

---

## 🚀 Deploy to Railway

### Step 1: Create a Railway Account
- Go to [https://railway.app](https://railway.app)
- Sign up (free tier available)

### Step 2: Create a New Project
1. Click "New Project"
2. Select "Deploy from GitHub repo" or "New from empty"
3. If from GitHub: Upload these files to a repo
4. If empty: Upload files manually

### Step 3: Upload Files
Upload these files to Railway:
- `qbank_pipeline.py`
- `Dockerfile`
- `requirements.txt`
- Your PDF file (e.g., `Psychology_QBank.pdf`)

### Step 4: Configure Variables
1. Go to project settings
2. Add environment variable if needed (none required for basic usage)

### Step 5: Deploy
1. Railway will automatically detect the Dockerfile
2. Click "Deploy"
3. Wait for build to complete (5-10 minutes)

### Step 6: Run the Pipeline
After deployment:
1. Go to the "Runtime" tab
2. Open the shell/terminal
3. Run:
```bash
python qbank_pipeline.py Psychology_QBank.pdf PSY
```

### Step 7: Download Results
1. In the Railway file browser, navigate to `data/` and `assets/`
2. Download the generated files

---

## 🚀 Deploy to Render

### Step 1: Create a Render Account
- Go to [https://render.com](https://render.com)
- Sign up (free tier available)

### Step 2: Create a New Background Worker
1. Click "New +" → "Background Worker"
2. Connect your GitHub repo or upload files
3. Select the repository containing your files

### Step 3: Configure the Worker
1. **Name**: QBank Pipeline
2. **Root Directory**: (leave blank or set to `/`)
3. **Build Command**: `docker build .`
4. **Start Command**: `python qbank_pipeline.py your_file.pdf SUBJECT_CODE`
   
   Example: `python qbank_pipeline.py Psychology_QBank.pdf PSY`

5. **Auto-Deploy**: Enable if you want automatic redeploys

### Step 4: Add Environment Variables
No environment variables are required for basic usage.

### Step 5: Deploy
Click "Create Background Worker"

### Step 6: Check Logs
- Go to the "Logs" tab to see pipeline progress
- Wait for completion (may take 10-30 minutes for large PDFs)

### Step 7: Download Results
1. Go to the "Files" tab
2. Download the `data/` and `assets/` directories

---

## 🔧 Customizing the Pipeline

### Processing Multiple PDFs
Create a shell script to process multiple PDFs:

```bash
#!/bin/bash
python qbank_pipeline.py Psychology_QBank.pdf PSY
python qbank_pipeline.py Biology_QBank.pdf BIO
python qbank_pipeline.py Chemistry_QBank.pdf CHE
```

### Using Vision Model (Manual Step)
For highest accuracy on complex tables:

1. The script will generate rasterized page images in the temp directory
2. Manually review pages with tables
3. Use a vision model (like me!) to extract table data
4. Update the `questions.jsonl` file with the corrected table markdown

### Adjusting OCR Quality
Edit the `TESSERACT_CONFIG` variable in `qbank_pipeline.py`:

```python
# For better accuracy (slower)
TESSERACT_CONFIG = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-+()[]{}!?:;"\/'

# For faster processing (less accurate)
TESSERACT_CONFIG = r'--oem 1 --psm 6'
```

---

## ⚠️ Troubleshooting

### Common Issues

**1. "pdftoppm not found"**
- Solution: Install poppler-utils
- Ubuntu: `sudo apt-get install poppler-utils`
- Mac: `brew install poppler`

**2. "tesseract not found"**
- Solution: Install tesseract-ocr
- Ubuntu: `sudo apt-get install tesseract-ocr tesseract-ocr-eng`
- Mac: `brew install tesseract`

**3. "pytesseract not found"**
- Solution: Install Python package
- `pip install pytesseract`

**4. "No modules named pypdf"**
- Solution: Install Python package
- `pip install pypdf`

**5. Pipeline is slow**
- Large PDFs (200+ pages) may take 10-30 minutes
- Consider processing in smaller batches
- Use Railway/Render with more CPU resources

**6. OCR accuracy is poor**
- Ensure your PDF is high quality
- Try adjusting the DPI (change `DPI = 150` to `DPI = 300` in the script)
- For complex layouts, use the vision model approach

---

## 📊 Pipeline Steps Explained

The script follows all 7 steps from your specification:

1. **Map real page numbers** - Uses `pdfinfo`, `pdffonts`, `pdftotext`
2. **Rasterize pages** - Uses `pdftoppm` at 150 DPI
3. **Find embedded images** - Uses `pypdf` to inspect resource dictionaries
4. **Extract real images** - Uses `pdfimages` + watermark detection
5. **Read content** - Uses `tesseract` OCR (vision model recommended for tables)
6. **Assemble JSON** - Creates `chapters.json` + `questions.jsonl`
7. **Validate** - Checks all JSON format and image references

---

## 🎯 Folder Structure (Output)

```
.
├── data/
│   ├── chapters.json          # Array of chapter objects
│   └── questions.jsonl        # JSONL with one question per line
│
└── assets/
    ├── questions/
    │   └── {SUBJECT}/
    │       ├── {SUBJECT}-001-001_Q_01.webp
    │       ├── {SUBJECT}-001-001_Q_02.webp
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

## 💡 Tips for Best Results

1. **Start with a small PDF** (10-20 pages) to test the pipeline
2. **Check the output** of the first chapter before processing everything
3. **For complex tables**: Use the vision model approach (feed page images to me)
4. **For large PDFs**: Process one subject at a time
5. **Backup your PDF**: The pipeline doesn't modify the original file

---

## 📞 Need Help?

If you encounter issues:
1. Check the logs for error messages
2. Verify all dependencies are installed
3. Try with a smaller PDF first
4. The script includes validation - check the error messages

For questions about the pipeline logic, refer to `qbank_extraction_pipeline.md`
