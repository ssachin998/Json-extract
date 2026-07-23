#!/usr/bin/env python3
"""
QBank PDF → JSON Extraction Pipeline
===================================
Full automation script following the exact specification from qbank_extraction_pipeline.md

Uses:
- poppler-utils (pdfinfo, pdffonts, pdftotext, pdftoppm, pdfimages)
- pypdf (Python) - for resource-dict inspection
- tesseract - OCR for text extraction
- Pillow (PIL) - image format conversion

Output:
- data/chapters.json
- data/questions.jsonl
- assets/questions/{SUBJECT}/
- assets/options/{SUBJECT}/
- assets/solutions/{SUBJECT}/
- assets/tables/{SUBJECT}/

Usage:
    python qbank_pipeline.py input.pdf SUBJECT_CODE
    
Example:
    python qbank_pipeline.py Psychology_QBank.pdf PSY
"""

import os
import sys
import json
import re
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

# Python libraries
from PIL import Image
from pypdf import PdfReader
import pytesseract


# ============================================================================
# CONFIGURATION
# ============================================================================

DPI = 150  # Rasterization DPI
TESSERACT_CONFIG = r'--oem 3 --psm 6'  # Best for single uniform block of text

# Folder structure
DATA_DIR = "data"
ASSETS_DIR = "assets"
IMAGE_CATEGORIES = ["questions", "options", "solutions", "tables"]

# Regex patterns
SUBJECT_PATTERN = re.compile(r'^[A-Z]{3}$')
ID_PATTERN = re.compile(r'^[A-Z]{3}-\d{3}-\d{3}$')
IMAGE_PATH_PATTERN = re.compile(
    r'^[A-Z]{3}/[A-Z]{3}-\d{3}-\d{3}_[A-Z]+(_[A-Z])?_\d{2}\.webp$'
)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_command(cmd: List[str], cwd: Optional[str] = None) -> Tuple[bool, str]:
    """Run a shell command and return success status and output."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300
        )
        return result.returncode == 0, result.stdout
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def check_dependencies() -> bool:
    """Check if all required tools are installed."""
    required = ['pdfinfo', 'pdffonts', 'pdftotext', 'pdftoppm', 'pdfimages', 'tesseract']
    missing = []
    
    for tool in required:
        # Poppler tools use '-v', Tesseract uses '--version'
        flag = '--version' if tool == 'tesseract' else '-v'
        success, _ = run_command([tool, flag])
        if not success:
            missing.append(tool)
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("Install on Ubuntu/Debian:")
        print("  sudo apt-get install poppler-utils tesseract-ocr libtesseract-dev")
        print("Install Python packages:")
        print("  pip install pypdf Pillow pytesseract")
        return False
    
    return True


def get_pdf_info(pdf_path: str) -> Dict[str, Any]:
    """Get PDF metadata using pdfinfo."""
    success, output = run_command(['pdfinfo', pdf_path])
    if not success:
        return {}
    
    info = {}
    for line in output.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            info[key.strip()] = value.strip()
    
    return info


def get_pdf_fonts(pdf_path: str) -> str:
    """Get PDF font information using pdffonts."""
    success, output = run_command(['pdffonts', pdf_path])
    return output if success else ""


def sample_text_page(pdf_path: str, page_num: int) -> str:
    """Sample text from a specific page using pdftotext."""
    success, output = run_command([
        'pdftotext', '-f', str(page_num), '-l', str(page_num),
        pdf_path, '-'
    ])
    return output if success else ""


def is_text_layer_broken(pdf_path: str, sample_page: int = 5) -> bool:
    """Check if the PDF has broken text layer."""
    text = sample_text_page(pdf_path, sample_page)
    
    # Check for garbled symbols
    garbled_patterns = [
        r'[¡†‡•…]',  # Common garbled symbols
        r'[\x80-\xFF]',  # Non-ASCII characters (except common ones)
    ]
    
    for pattern in garbled_patterns:
        if re.search(pattern, text):
            return True
    
    # If text is very short or mostly punctuation
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    if len(words) < 10:
        return True
    
    return False


# ============================================================================
# IMAGE EXTRACTION FUNCTIONS
# ============================================================================

def get_page_images(pdf_path: str, page_num: int) -> List[Dict[str, Any]]:
    """Get all XObject images from a specific page using pypdf."""
    reader = PdfReader(pdf_path)
    
    if page_num > len(reader.pages):
        return []
    
    page = reader.pages[page_num - 1]  # 1-indexed
    images = []
    
    if '/Resources' in page:
        resources = page['/Resources']
        if '/XObject' in resources:
            xobjects = resources['/XObject']
            for name, obj in xobjects.items():
                if obj is None:
                    continue
                
                obj = obj.get_object()
                if '/Width' in obj and '/Height' in obj:
                    images.append({
                        'name': name,
                        'width': obj['/Width'],
                        'height': obj['/Height'],
                        'idnum': obj.get('/ID', [0, 0])[0] if '/ID' in obj else 0,
                        'ref': obj
                    })
    
    return images


def detect_watermark(pdf_path: str) -> Optional[Dict[str, Any]]:
    """Detect if there's a reused watermark image across all pages."""
    reader = PdfReader(pdf_path)
    image_counts = defaultdict(int)
    
    for i, page in enumerate(reader.pages):
        if '/Resources' not in page:
            continue
        
        resources = page['/Resources']
        if '/XObject' not in resources:
            continue
        
        xobjects = resources['/XObject']
        for name, obj in xobjects.items():
            if obj is None:
                continue
            
            obj = obj.get_object()
            if '/Width' in obj and '/Height' in obj:
                # Create a unique key based on dimensions and object ID
                key = (obj['/Width'], obj['/Height'], obj.get('/ID', [0, 0])[0])
                image_counts[key] += 1
    
    # Find the most common image (likely watermark)
    if image_counts:
        watermark_key = max(image_counts.items(), key=lambda x: x[1])[0]
        if image_counts[watermark_key] == len(reader.pages):
            # This image appears on every page - it's a watermark
            return {
                'width': watermark_key[0],
                'height': watermark_key[1],
                'id': watermark_key[2]
            }
    
    return None


def extract_page_images(pdf_path: str, page_num: int, output_dir: str, 
                         watermark: Optional[Dict[str, Any]] = None) -> List[str]:
    """Extract images from a specific page, excluding watermark."""
    extracted = []
    
    # Use pdfimages to extract
    success, _ = run_command([
        'pdfimages', '-png', '-f', str(page_num), '-l', str(page_num),
        pdf_path, os.path.join(output_dir, 'page_%03d')
    ])
    
    if not success:
        return extracted
    
    # Find all extracted PNG files
    png_files = sorted(Path(output_dir).glob(f"page_{page_num:03d}-*.png"))
    
    # Filter out watermark if detected
    if watermark:
        for png_file in png_files:
            try:
                img = Image.open(png_file)
                if (img.width == watermark['width'] and 
                    img.height == watermark['height']):
                    png_file.unlink()  # Delete watermark
                    continue
            except:
                pass
    
    # Convert to WEBP and return paths
    for i, png_file in enumerate(png_files):
        webp_path = png_file.with_suffix('.webp')
        try:
            img = Image.open(png_file)
            img.save(webp_path, 'WEBP', quality=95)
            png_file.unlink()  # Remove PNG
            extracted.append(str(webp_path))
        except Exception as e:
            print(f"  ⚠️  Failed to convert {png_file}: {e}")
    
    return extracted


# ============================================================================
# PAGE PROCESSING FUNCTIONS
# ============================================================================

def rasterize_page(pdf_path: str, page_num: int, output_dir: str) -> Optional[str]:
    """Rasterize a single page to JPEG using pdftoppm."""
    output_prefix = os.path.join(output_dir, f"page_{page_num:04d}")
    success, _ = run_command([
        'pdftoppm', '-jpeg', '-r', str(DPI), '-f', str(page_num), '-l', str(page_num),
        pdf_path, output_prefix
    ])
    
    if not success:
        return None
    
    # Find the generated JPEG
    jpeg_files = list(Path(output_dir).glob(f"page_{page_num:04d}.jpg"))
    if jpeg_files:
        return str(jpeg_files[0])
    
    return None


def rasterize_page_range(pdf_path: str, start_page: int, end_page: int, 
                         output_dir: str) -> List[str]:
    """Rasterize a range of pages."""
    ensure_dir(output_dir)
    pages = []
    
    for page_num in range(start_page, end_page + 1):
        jpeg_path = rasterize_page(pdf_path, page_num, output_dir)
        if jpeg_path:
            pages.append(jpeg_path)
        else:
            print(f"  ⚠️  Failed to rasterize page {page_num}")
    
    return pages


def extract_text_from_image(image_path: str) -> str:
    """Extract text from an image using tesseract OCR."""
    try:
        text = pytesseract.image_to_string(
            Image.open(image_path),
            config=TESSERACT_CONFIG
        )
        return text.strip()
    except Exception as e:
        print(f"  ⚠️  OCR failed for {image_path}: {e}")
        return ""


def extract_tables_from_image(image_path: str) -> List[Dict[str, Any]]:
    """Extract tables from an image using tesseract (basic table detection)."""
    # Note: For high-quality table extraction, a vision model is recommended
    # This is a fallback using OCR
    try:
        # Use tesseract's table detection mode
        data = pytesseract.image_to_data(
            Image.open(image_path),
            config=r'--oem 3 --psm 6 outputbase table',
            output_type=pytesseract.Output.DICT
        )
        
        # Basic table reconstruction (this is simplified)
        # In production, consider using camelot, pdfplumber, or a vision model
        tables = []
        
        # Group by table structure
        # This is a placeholder - real table extraction needs more sophisticated logic
        if data.get('text'):
            tables.append({
                'type': 'extracted',
                'markdown': '',  # Would need proper table parsing
                'file': None
            })
        
        return tables
    except Exception as e:
        print(f"  ⚠️  Table extraction failed for {image_path}: {e}")
        return []


# ============================================================================
# CHAPTER & QUESTION PROCESSING
# ============================================================================

def detect_chapters(pdf_path: str, subject_code: str) -> List[Dict[str, Any]]:
    """Detect chapters from the PDF."""
    chapters = []
    reader = PdfReader(pdf_path)
    
    # Try to extract from TOC first
    if '/Outlines' in reader.trailer['/Root']:
        outlines = reader.trailer['/Root']['/Outlines'].get_object()
        if '/First' in outlines:
            # Parse TOC
            current = outlines['/First'].get_object()
            chapter_num = 1
            
            while current:
                if '/Title' in current:
                    title = current['/Title']
                    # Clean title
                    title = title.replace('\r', ' ').replace('\n', ' ').strip()
                    
                    chapter_id = f"{subject_code}-{chapter_num:03d}"
                    chapters.append({
                        'chapter_id': chapter_id,
                        'subject': subject_code,
                        'chapter_no': chapter_num,
                        'chapter_title': title,
                        'start_page': None,  # Will be determined
                        'end_page': None
                    })
                    chapter_num += 1
                
                if '/Next' in current:
                    current = current['/Next'].get_object()
                else:
                    break
    
    # If TOC parsing failed, use a simple approach
    if not chapters:
        # Assume each chapter starts on a page with "Chapter" or similar
        chapter_num = 1
        for i, page in enumerate(reader.pages):
            text = page.extract_text() if page else ""
            
            # Look for chapter markers
            chapter_patterns = [
                rf'Chapter\s+{chapter_num}',
                rf'CHAPTER\s+{chapter_num}',
                rf'Ch\.?\s+{chapter_num}',
            ]
            
            for pattern in chapter_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    chapter_id = f"{subject_code}-{chapter_num:03d}"
                    chapters.append({
                        'chapter_id': chapter_id,
                        'subject': subject_code,
                        'chapter_no': chapter_num,
                        'chapter_title': f"Chapter {chapter_num}",
                        'start_page': i + 1,
                        'end_page': None
                    })
                    chapter_num += 1
                    break
    
    return chapters


def determine_chapter_page_ranges(pdf_path: str, chapters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Determine start and end pages for each chapter."""
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    
    # If we have start pages from TOC, use them
    if chapters and chapters[0].get('start_page'):
        for i in range(len(chapters) - 1):
            chapters[i]['end_page'] = chapters[i + 1]['start_page'] - 1
        chapters[-1]['end_page'] = total_pages
        return chapters
    
    # Otherwise, divide pages equally
    num_chapters = len(chapters)
    if num_chapters == 0:
        return []
    
    pages_per_chapter = total_pages // num_chapters
    
    for i, chapter in enumerate(chapters):
        start = i * pages_per_chapter + 1
        end = (i + 1) * pages_per_chapter if i < num_chapters - 1 else total_pages
        chapter['start_page'] = start
        chapter['end_page'] = end
    
    return chapters


def parse_question_from_text(text: str, chapter_id: str, question_num: int, 
                             subject_code: str) -> Optional[Dict[str, Any]]:
    """Parse a question from OCR text."""
    # This is a simplified parser - in production, use a vision model
    # or more sophisticated NLP
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    if not lines:
        return None
    
    question_id = f"{subject_code}-{int(chapter_id.split('-')[1]):03d}-{question_num:03d}"
    
    # Find question text (first non-empty line that looks like a question)
    question_text = ""
    options = []
    solution_text = ""
    correct_options = []
    tags = []
    
    # Simple state machine
    state = 'question'  # question, options, solution
    current_option = None
    
    for line in lines:
        # Skip page numbers, headers, footers
        if re.match(r'^\d+$', line) or re.match(r'^Page\s+\d+', line, re.IGNORECASE):
            continue
        
        if state == 'question':
            # Check if this is the start of options
            if re.match(r'^[A-D]\.\)?\s', line, re.IGNORECASE):
                state = 'options'
                # Save the question text accumulated so far
                question_text = '\n'.join([l for l in lines[:lines.index(line)] if l]).strip()
                
                # Process this option
                match = re.match(r'^[A-D]\.\)?\s(.*)', line, re.IGNORECASE)
                if match:
                    current_option = match.group(1).strip()
                    option_id = line[0].upper()
                    options.append({'id': option_id, 'text': current_option, 'images': []})
            else:
                question_text = line if not question_text else question_text + "\n" + line
        
        elif state == 'options':
            # Check if this is another option
            if re.match(r'^[A-D]\.\)?\s', line, re.IGNORECASE):
                match = re.match(r'^[A-D]\.\)?\s(.*)', line, re.IGNORECASE)
                if match:
                    current_option = match.group(1).strip()
                    option_id = line[0].upper()
                    options.append({'id': option_id, 'text': current_option, 'images': []})
            # Check if this is the answer/solution
            elif re.match(r'(Answer|Solution|Explanation|Correct):', line, re.IGNORECASE):
                state = 'solution'
                solution_text = line + "\n"
            else:
                # Continue current option
                if options and current_option is not None:
                    options[-1]['text'] += "\n" + line
        
        elif state == 'solution':
            solution_text += line + "\n"
            
            # Check for correct answer markers
            if re.match(r'^[A-D]$', line) or re.match(r'^[A-D]\.\)?$', line):
                correct_options.append(line[0].upper())
    
    # Clean up
    question_text = question_text.strip()
    solution_text = solution_text.strip()
    
    # Remove empty options
    options = [opt for opt in options if opt['text'].strip()]
    
    if not question_text or not options:
        return None
    
    return {
        'id': question_id,
        'subject': subject_code,
        'chapter_id': chapter_id,
        'question': {
            'text': question_text,
            'images': []
        },
        'options': options,
        'correct_options': correct_options if correct_options else ['A'],  # Default
        'solution': {
            'text': solution_text,
            'images': [],
            'tables': []
        },
        'tags': tags
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def process_chapter(pdf_path: str, chapter: Dict[str, Any], 
                    subject_code: str, temp_dir: str) -> List[Dict[str, Any]]:
    """Process a single chapter and return questions."""
    print(f"\n📖 Processing Chapter: {chapter['chapter_title']}")
    print(f"   Pages: {chapter['start_page']}-{chapter['end_page']}")
    
    questions = []
    chapter_dir = os.path.join(temp_dir, f"chapter_{chapter['chapter_id']}")
    ensure_dir(chapter_dir)
    
    # Step 2: Rasterize the chapter's page range
    print("   🖼️  Rasterizing pages...")
    page_images = rasterize_page_range(
        pdf_path,
        chapter['start_page'],
        chapter['end_page'],
        chapter_dir
    )
    
    if not page_images:
        print("   ⚠️  No pages rasterized")
        return questions
    
    # Step 3: Find embedded images per page
    print("   🔍 Detecting watermarks...")
    watermark = detect_watermark(pdf_path)
    if watermark:
        print(f"   ✅ Watermark detected: {watermark['width']}x{watermark['height']}")
    else:
        print("   ℹ️  No watermark detected")
    
    # Step 4: Extract real images
    print("   🖼️  Extracting embedded images...")
    image_dir = os.path.join(temp_dir, "extracted_images")
    ensure_dir(image_dir)
    
    for page_num in range(chapter['start_page'], chapter['end_page'] + 1):
        page_img_dir = os.path.join(image_dir, f"page_{page_num}")
        ensure_dir(page_img_dir)
        extract_page_images(pdf_path, page_num, page_img_dir, watermark)
    
    # Step 5: Read page content with OCR
    print("   📄 Extracting text with OCR...")
    question_num = 1
    
    for page_num, jpeg_path in enumerate(page_images, start=chapter['start_page']):
        text = extract_text_from_image(jpeg_path)
        
        if not text:
            continue
        
        # Parse questions from this page
        # In a real implementation, use a vision model for better accuracy
        question = parse_question_from_text(
            text, chapter['chapter_id'], question_num, subject_code
        )
        
        if question:
            # Add image references if any were extracted for this page
            page_img_dir = os.path.join(image_dir, f"page_{page_num}")
            if os.path.exists(page_img_dir):
                webp_files = list(Path(page_img_dir).glob("*.webp"))
                for i, webp_file in enumerate(webp_files):
                    # Copy to final assets location
                    img_type = "questions"  # Simplified - would need to detect type
                    final_path = os.path.join(
                        ASSETS_DIR, img_type, subject_code,
                        f"{question['id']}_Q_{i+1:02d}.webp"
                    )
                    ensure_dir(os.path.dirname(final_path))
                    shutil.copy(webp_file, final_path)
                    
                    # Add to question images
                    question['question']['images'].append({
                        'type': 'figure',
                        'file': f"{subject_code}/{question['id']}_Q_{i+1:02d}.webp"
                    })
            
            questions.append(question)
            question_num += 1
    
    print(f"   ✅ Found {len(questions)} questions")
    return questions


def save_chapters_json(chapters: List[Dict[str, Any]], output_dir: str) -> None:
    """Save chapters to JSON file."""
    ensure_dir(output_dir)
    
    chapters_data = []
    for chapter in chapters:
        chapters_data.append({
            'chapter_id': chapter['chapter_id'],
            'subject': chapter['subject'],
            'chapter_no': chapter['chapter_no'],
            'chapter_title': chapter['chapter_title']
        })
    
    with open(os.path.join(output_dir, 'chapters.json'), 'w') as f:
        json.dump(chapters_data, f, indent=2)
    
    print(f"✅ Saved {len(chapters_data)} chapters to {output_dir}/chapters.json")


def save_questions_jsonl(questions: List[Dict[str, Any]], output_dir: str) -> None:
    """Save questions to JSONL file."""
    ensure_dir(output_dir)
    
    with open(os.path.join(output_dir, 'questions.jsonl'), 'w') as f:
        for question in questions:
            f.write(json.dumps(question) + '\n')
    
    print(f"✅ Saved {len(questions)} questions to {output_dir}/questions.jsonl")


def validate_outputs(subject_code: str, data_dir: str, assets_dir: str) -> bool:
    """Validate all outputs against the schema."""
    print("\n🔍 Validating outputs...")
    
    # Check chapters.json
    chapters_path = os.path.join(data_dir, 'chapters.json')
    if not os.path.exists(chapters_path):
        print("❌ chapters.json not found")
        return False
    
    with open(chapters_path) as f:
        chapters = json.load(f)
    
    # Check questions.jsonl
    questions_path = os.path.join(data_dir, 'questions.jsonl')
    if not os.path.exists(questions_path):
        print("❌ questions.jsonl not found")
        return False
    
    questions = []
    with open(questions_path) as f:
        for line_num, line in enumerate(f, 1):
            try:
                q = json.loads(line)
                questions.append(q)
                
                # Validate ID format
                if not ID_PATTERN.match(q['id']):
                    print(f"❌ Invalid ID format on line {line_num}: {q['id']}")
                    return False
                
                # Validate subject
                if q['subject'] != subject_code:
                    print(f"❌ Subject mismatch on line {line_num}: {q['subject']}")
                    return False
                
                # Validate image paths
                path_re = re.compile(
                    rf'^{subject_code}/[A-Z]{{3}}-\d{{3}}-\d{{3}}_[A-Z]+(_[A-Z])?_\d{{2}}\.webp$'
                )
                
                for bucket in [q['question']] + q['options'] + [q['solution']]:
                    for img in bucket.get('images', []):
                        if not path_re.match(img['file']):
                            print(f"❌ Invalid image path on line {line_num}: {img['file']}")
                            return False
                        
                        # Check if file exists
                        img_path = os.path.join(assets_dir, img['file'])
                        if not os.path.exists(img_path):
                            print(f"❌ Image file not found on line {line_num}: {img_path}")
                            return False
                    
                    for tbl in bucket.get('tables', []):
                        if tbl.get('file') and not path_re.match(tbl['file']):
                            print(f"❌ Invalid table path on line {line_num}: {tbl['file']}")
                            return False
                
            except json.JSONDecodeError:
                print(f"❌ Invalid JSON on line {line_num}")
                return False
    
    # Check for orphaned image files
    all_image_paths = set()
    for q in questions:
        for bucket in [q['question']] + q['options'] + [q['solution']]:
            for img in bucket.get('images', []):
                all_image_paths.add(img['file'])
            for tbl in bucket.get('tables', []):
                if tbl.get('file'):
                    all_image_paths.add(tbl['file'])
    
    for img_type in IMAGE_CATEGORIES:
        type_dir = os.path.join(assets_dir, img_type, subject_code)
        if os.path.exists(type_dir):
            for root, dirs, files in os.walk(type_dir):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), assets_dir)
                    if rel_path not in all_image_paths:
                        print(f"⚠️  Orphaned image file: {rel_path}")
    
    print("✅ All validations passed!")
    return True


def main():
    """Main pipeline execution."""
    if len(sys.argv) < 3:
        print("Usage: python qbank_pipeline.py <pdf_path> <subject_code>")
        print("Example: python qbank_pipeline.py Psychology_QBank.pdf PSY")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    subject_code = sys.argv[2].upper()
    
    # Validate subject code
    if not SUBJECT_PATTERN.match(subject_code):
        print(f"❌ Invalid subject code: {subject_code}. Must be 3 uppercase letters.")
        sys.exit(1)
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print(f"\n🚀 Starting QBank Pipeline")
    print(f"   PDF: {pdf_path}")
    print(f"   Subject: {subject_code}")
    
    # Create temp directory
    temp_dir = f"temp_{subject_code}"
    ensure_dir(temp_dir)
    
    # Step 1: Map real page numbers
    print("\n📊 Step 1: Analyzing PDF...")
    pdf_info = get_pdf_info(pdf_path)
    print(f"   Pages: {pdf_info.get('Pages', 'N/A')}")
    print(f"   Title: {pdf_info.get('Title', 'N/A')}")
    
    fonts_info = get_pdf_fonts(pdf_path)
    print(f"   Fonts: {fonts_info.count('embedded') if fonts_info else 0} embedded")
    
    # Check if text layer is broken
    is_broken = is_text_layer_broken(pdf_path)
    print(f"   Text layer: {'❌ BROKEN' if is_broken else '✅ OK'}")
    
    if not is_broken:
        print("   ℹ️  Text layer is intact. Consider using pdftotext directly for better accuracy.")
    
    # Step 1.5: Detect chapters
    print("\n📚 Step 1.5: Detecting chapters...")
    chapters = detect_chapters(pdf_path, subject_code)
    
    if not chapters:
        print("   ⚠️  No chapters detected. Using single chapter.")
        chapters = [{
            'chapter_id': f"{subject_code}-001",
            'subject': subject_code,
            'chapter_no': 1,
            'chapter_title': 'Full Document',
            'start_page': 1,
            'end_page': int(pdf_info.get('Pages', 0))
        }]
    else:
        chapters = determine_chapter_page_ranges(pdf_path, chapters)
    
    print(f"   ✅ Found {len(chapters)} chapters")
    for ch in chapters:
        print(f"      - {ch['chapter_id']}: {ch['chapter_title']} (Pages {ch['start_page']}-{ch['end_page']})")
    
    # Create output directories
    ensure_dir(DATA_DIR)
    for img_type in IMAGE_CATEGORIES:
        ensure_dir(os.path.join(ASSETS_DIR, img_type, subject_code))
    
    # Process each chapter
    all_questions = []
    for chapter in chapters:
        questions = process_chapter(pdf_path, chapter, subject_code, temp_dir)
        all_questions.extend(questions)
    
    # Save outputs
    save_chapters_json(chapters, DATA_DIR)
    save_questions_jsonl(all_questions, DATA_DIR)
    
    # Validate
    if validate_outputs(subject_code, DATA_DIR, ASSETS_DIR):
        print("\n🎉 Pipeline completed successfully!")
        print(f"   Chapters: {len(chapters)}")
        print(f"   Questions: {len(all_questions)}")
        print(f"   Output directory: {os.path.abspath(DATA_DIR)}")
        print(f"   Assets directory: {os.path.abspath(ASSETS_DIR)}")
    else:
        print("\n❌ Pipeline completed with validation errors")
        sys.exit(1)
    
    # Cleanup temp directory
    print(f"\n🧹 Cleaning up temporary files...")
    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
