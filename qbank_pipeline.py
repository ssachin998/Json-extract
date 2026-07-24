#!/usr/bin/env python3
"""
QBank PDF -> JSON extraction pipeline.
Run this on your own machine / Railway (needs: poppler-utils, pypdf, Pillow,
google-generativeai, requests). Designed to survive a 100-req/day Gemini
free-tier limit by checkpointing progress and resuming across multiple runs
(e.g. via a daily cron job / Railway scheduled task).

SETUP
-----
pip install pypdf pillow google-generativeai
apt-get install poppler-utils        # gives you pdftoppm, pdfimages, pdftotext

Set your key:
    export GEMINI_API_KEY="your-key-here"

CONFIGURE
---------
Edit PDFS below: one entry per subject PDF. `page_offset` = (PDF file page
number) - (printed page number shown at the bottom of the page). Find this
ONCE per PDF manually:
    pdftoppm -jpeg -r 150 -f 4 -l 4 yourbook.pdf /tmp/check
    # open /tmp/check-004.jpg, look at chapter 1's printed page number vs "4"
    # offset = 4 - printed_page_number_seen

RUN
---
python3 qbank_pipeline.py
(re-run it daily / whenever you hit the rate limit message; it resumes
automatically from state.json)
"""

import base64
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import google.generativeai as genai
from PIL import Image
from pypdf import PdfReader

# ============================================================
# CONFIG — edit this section for each new subject PDF
# ============================================================

PDFS = [
    {
        "subject": "PSY",
        "path": "./pdfs/Psychiatry_ed8.pdf",
        "page_offset": -1,
    },
    # add the other 19 here, same shape — path is relative to /app/pdfs/
    # since that's where the Dockerfile copies them
]

# OUTPUT_ROOT points into the Railway Volume mount (/data) so progress
# and output survive restarts/redeploys. Falls back to a local folder
# if you're running this outside Railway (e.g. Colab) without a volume.
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_DIR", "./qbank_output"))
DATA_DIR = OUTPUT_ROOT / "data"
ASSETS_DIR = OUTPUT_ROOT / "assets"
STATE_FILE = OUTPUT_ROOT / "state.json"

MAX_CALLS_PER_DAY = 95          # safety buffer under your 100/day cap
PAGES_PER_GEMINI_CALL = 6       # tune this: more pages/call = fewer calls,
                                 # but keep it small enough that Gemini can
                                 # read every question accurately
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"   # confirmed working model from your bot's config

IMG_PATH_RE = re.compile(r"^[A-Z]{3}/[A-Z]{3}-\d{3}-\d{3}_[A-Z]+(_[A-Z])?_\d{2}\.webp$")

# ============================================================
# STATE (checkpoint / resume)
# ============================================================

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"calls_today": 0, "day_stamp": "", "pdf_progress": {}}

def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))

def today_stamp():
    return time.strftime("%Y-%m-%d")

def reset_daily_counter_if_needed(state):
    if state.get("day_stamp") != today_stamp():
        state["day_stamp"] = today_stamp()
        state["calls_today"] = 0

# ============================================================
# STEP 1: parse the TOC to auto-discover chapters + page ranges
# (TOC pages have clean, non-garbled text -- confirmed reliable to
# pdftotext even on PDFs where body-page text is broken/garbled)
# ============================================================

def extract_toc_chapters(pdf_path, toc_page_range=(2, 4)):
    """
    Returns [{"chapter_no": int, "chapter_title": str, "start_printed_page": int}, ...]
    Adjust toc_page_range per PDF if the contents table spans more/fewer pages.
    """
    text = subprocess.run(
        ["pdftotext", "-f", str(toc_page_range[0]), "-l", str(toc_page_range[1]),
         "-layout", pdf_path, "-"],
        capture_output=True, text=True
    ).stdout

    chapters = []
    # Matches lines like: "12   Bipolar and Related Disorders   160"
    for line in text.splitlines():
        m = re.match(r"^\s*(\d{1,3})\s+(.*?)\s+(\d{1,4})\s*$", line)
        if m:
            no, title, page = m.groups()
            title = title.strip()
            if len(title) < 3:
                continue
            chapters.append({
                "chapter_no": int(no),
                "chapter_title": title,
                "start_printed_page": int(page),
            })
    return chapters

def compute_page_ranges(chapters, page_offset, last_page_file):
    """Turns a flat chapter list into (file_start, file_end) ranges."""
    for i, ch in enumerate(chapters):
        file_start = ch["start_printed_page"] + page_offset
        if i + 1 < len(chapters):
            next_start = chapters[i + 1]["start_printed_page"] + page_offset
            file_end = next_start - 1
        else:
            file_end = last_page_file
        ch["file_start"] = file_start
        ch["file_end"] = file_end
    return chapters

# ============================================================
# STEP 2: watermark auto-detection
# (the shared background image reused on every page -- must be
# excluded, or you'll extract the watermark instead of real figures)
# ============================================================

def find_watermark_object_id(pdf_path, sample_pages=30):
    reader = PdfReader(pdf_path)
    counts = {}
    n = min(sample_pages, len(reader.pages))
    for i in range(n):
        res = reader.pages[i].get("/Resources")
        xobjs = res.get("/XObject") if res else None
        if not xobjs:
            continue
        for name, ref in xobjs.items():
            obj = ref.get_object()
            if obj.get("/Subtype") == "/Image":
                counts[ref.idnum] = counts.get(ref.idnum, 0) + 1
    if not counts:
        return None
    # whichever object ID appears on (almost) every sampled page = watermark
    watermark_id = max(counts, key=counts.get)
    if counts[watermark_id] < n * 0.5:
        return None  # no dominant repeated image -> no watermark to exclude
    return watermark_id

def extract_real_images(pdf_path, file_page, watermark_id, subject, out_dir):
    """
    Extracts every embedded image on file_page EXCEPT the watermark object.
    Returns a list of saved relative paths ("SUBJECT/filename.webp").
    Caller is responsible for deciding which question/option/solution each
    belongs to (Gemini's response should say which figure goes where).
    """
    reader = PdfReader(pdf_path)
    if not (1 <= file_page <= len(reader.pages)):
        print(f"  [WARN] extract_real_images: page {file_page} out of range "
              f"(pdf has {len(reader.pages)} pages) -- skipping")
        return []
    page = reader.pages[file_page - 1]
    res = page.get("/Resources")
    xobjs = res.get("/XObject") if res else None
    saved = []
    if not xobjs:
        return saved
    (out_dir / subject).mkdir(parents=True, exist_ok=True)
    for name, ref in xobjs.items():
        obj = ref.get_object()
        if obj.get("/Subtype") != "/Image" or ref.idnum == watermark_id:
            continue
        try:
            img = page.images[name].image if hasattr(page, "images") else None
        except Exception:
            img = None
        # fall back to pdfimages CLI extraction (handles CMYK/odd encodings better)
        tmp_prefix = f"/tmp/{subject}_p{file_page}_{ref.idnum}"
        subprocess.run(
            ["pdfimages", "-png", "-f", str(file_page), "-l", str(file_page), pdf_path, tmp_prefix],
            capture_output=True
        )
        for f in sorted(Path("/tmp").glob(f"{Path(tmp_prefix).name}*.png")):
            im = Image.open(f)
            if im.size[0] * im.size[1] < 5000:
                continue  # skip tiny noise images
            fname = f"{subject}-p{file_page}-{ref.idnum}.webp"
            rel_path = f"{subject}/{fname}"
            im.convert("RGB").save(out_dir / subject / fname, "WEBP", quality=95)
            saved.append(rel_path)
    return saved

# ============================================================
# STEP 3: Gemini call — page images in, structured JSON out
# ============================================================

SCHEMA_PROMPT = """You are extracting MCQ questions from scanned textbook pages into strict JSON.

Return a JSON array. Each element is one question:
{
  "q_no": <question number as printed>,
  "question_text": "...",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "correct_option": "A" | "B" | "C" | "D" | null,   // null if answer key not on these pages
  "solution_text": "..." | null,                     // null if solution not on these pages
  "tables": [{"type": "short_label", "markdown": "| col | col |\\n|---|---|\\n..."}],
  "has_figure_in_question": true|false,
  "has_figure_in_solution": true|false
}

Rules:
- Preserve every word verbatim. Do NOT summarize or paraphrase.
- If a page only contains an answer-key table (Q.No -> correct option) with
  no other question text, still return entries for those q_no with only
  "correct_option" filled and everything else null.
- If a page only contains solutions, return entries with only "solution_text"
  (and "tables" if present) filled, matched to the right q_no.
- Any table in the solution (e.g. stage/phase comparison tables) must be
  converted to a markdown table string in "tables", not skipped.
- Output ONLY the JSON array, no commentary, no markdown code fences.
"""

SAFETY_SETTINGS = [
    {"category": c, "threshold": "BLOCK_ONLY_HIGH"}
    for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
              "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
]
# Medical/psychiatry textbook content routinely covers violence, self-harm,
# sexual assault etc. in a clinical context (e.g. "which defense mechanism
# explains this rape survivor's amnesia") -- BLOCK_ONLY_HIGH keeps obviously
# harmful content blocked while allowing legitimate clinical material through.

def call_gemini_on_pages(model, image_paths):
    parts = [SCHEMA_PROMPT]
    for p in image_paths:
        parts.append(Image.open(p))
    resp = model.generate_content(
        parts,
        safety_settings=SAFETY_SETTINGS,
        request_options={"retry": None},
    )

    if not resp.candidates:
        raise RuntimeError(f"Empty response (prompt blocked?). prompt_feedback={resp.prompt_feedback}")

    candidate = resp.candidates[0]
    finish_reason = getattr(candidate, "finish_reason", None)
    if finish_reason and str(finish_reason) not in ("1", "STOP"):
        raise RuntimeError(f"Response did not finish normally (finish_reason={finish_reason}). "
                            f"Likely safety-blocked or hit token limit -- try fewer pages per call.")

    text = resp.text.strip()
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(text)

# ============================================================
# STEP 4: merge partial results (a question's text might be on one
# page and its answer/solution on a later page) into final records
# ============================================================

def merge_question_records(existing, new_items):
    """existing: dict keyed by q_no -> record (in progress for current chapter)"""
    for item in new_items:
        qn = item["q_no"]
        rec = existing.setdefault(qn, {
            "q_no": qn, "question_text": None, "options": None,
            "correct_option": None, "solution_text": None, "tables": [],
            "has_figure_in_question": False, "has_figure_in_solution": False,
        })
        for k in ["question_text", "options", "correct_option", "solution_text"]:
            if item.get(k):
                rec[k] = item[k]
        if item.get("tables"):
            rec["tables"].extend(item["tables"])
        rec["has_figure_in_question"] = rec["has_figure_in_question"] or item.get("has_figure_in_question", False)
        rec["has_figure_in_solution"] = rec["has_figure_in_solution"] or item.get("has_figure_in_solution", False)
    return existing

def build_final_question(subject, chapter_id, chapter_no, q_no, rec, image_files):
    qid = f"{subject}-{chapter_no:03d}-{q_no:03d}"
    q_images = [{"type": "figure", "file": f} for f in image_files.get("question", [])]
    sol_images = [{"type": "figure", "file": f} for f in image_files.get("solution", [])]
    tables = [{"type": t.get("type", "table"), "markdown": t["markdown"], "file": None}
              for t in rec.get("tables", [])]

    for img in q_images + sol_images:
        assert IMG_PATH_RE.match(img["file"]), f"Bad image path generated: {img['file']}"

    return {
        "id": qid,
        "subject": subject,
        "chapter_id": chapter_id,
        "question": {"text": rec["question_text"], "images": q_images},
        "options": [{"id": k, "text": v, "images": []} for k, v in (rec["options"] or {}).items()],
        "correct_options": [rec["correct_option"]] if rec["correct_option"] else [],
        "solution": {"text": rec["solution_text"], "images": sol_images, "tables": tables},
        "tags": [],
    }

# ============================================================
# MAIN DRIVER
# ============================================================

def process_pdf(pdf_cfg, state, genai_model, chapters_out, questions_fh):
    subject = pdf_cfg["subject"]
    pdf_path = pdf_cfg["path"]
    progress = state["pdf_progress"].setdefault(subject, {"chapters_done": [], "current": None})

    watermark_id = find_watermark_object_id(pdf_path)
    print(f"[{subject}] watermark object id: {watermark_id}")

    total_pages = len(PdfReader(pdf_path).pages)
    toc = extract_toc_chapters(pdf_path)
    chapters = compute_page_ranges(toc, pdf_cfg["page_offset"], total_pages)

    for ch in chapters:
        chapter_id = f"{subject}-{ch['chapter_no']:03d}"
        if chapter_id in progress["chapters_done"]:
            continue

        chapters_out.append({
            "chapter_id": chapter_id, "subject": subject,
            "chapter_no": ch["chapter_no"], "chapter_title": ch["chapter_title"],
        })

        page_dir = Path(f"/tmp/{subject}_ch{ch['chapter_no']:03d}")
        page_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "pdftoppm", "-jpeg", "-r", "150",
            "-f", str(ch["file_start"]), "-l", str(ch["file_end"]),
            pdf_path, str(page_dir / "page")
        ])
        page_files = sorted(page_dir.glob("page-*.jpg"))

        chapter_records = {}
        image_files_by_q = {}  # not tracked per-q here for simplicity; see NOTE below

        for batch_start in range(0, len(page_files), PAGES_PER_GEMINI_CALL):
            reset_daily_counter_if_needed(state)
            if state["calls_today"] >= MAX_CALLS_PER_DAY:
                print("Daily Gemini call limit reached. Saving progress, exiting.")
                save_state(state)
                sys.exit(0)

            batch = page_files[batch_start:batch_start + PAGES_PER_GEMINI_CALL]
            try:
                items = call_gemini_on_pages(genai_model, batch)
            except Exception as e:
                err_text = str(e)
                if "429" in err_text or "quota" in err_text.lower():
                    print(f"  [QUOTA] Gemini quota exhausted -- stopping run for now: {e}")
                    save_state(state)
                    sys.exit(0)
                print(f"  [WARN] Gemini call failed on {subject} ch{ch['chapter_no']} batch {batch_start}: {e}")
                continue
            state["calls_today"] += 1
            save_state(state)

            chapter_records = merge_question_records(chapter_records, items)

            # extract real (non-watermark) images from this batch's pages.
            # pdftoppm names output files using the ACTUAL pdf page number
            # (e.g. page-005.jpg for real page 5) -- read it directly from
            # the filename, don't recompute it relative to ch["file_start"].
            for pf in batch:
                file_page_num = int(pf.stem.split("-")[-1])
                imgs = extract_real_images(pdf_path, file_page_num, watermark_id, subject, ASSETS_DIR / "questions")
                # NOTE: simple version -- attaches any image found on a page to
                # whichever q_no Gemini flagged has_figure_in_question True and
                # doesn't have an image yet. Review this mapping manually for
                # pages with multiple figures.
                for qn, rec in chapter_records.items():
                    if rec.get("has_figure_in_question") and qn not in image_files_by_q:
                        if imgs:
                            image_files_by_q[qn] = {"question": imgs, "solution": []}

        for qn, rec in sorted(chapter_records.items()):
            final_q = build_final_question(
                subject, chapter_id, ch["chapter_no"], qn, rec,
                image_files_by_q.get(qn, {"question": [], "solution": []})
            )
            questions_fh.write(json.dumps(final_q, ensure_ascii=False) + "\n")
            questions_fh.flush()

        progress["chapters_done"].append(chapter_id)
        save_state(state)
        print(f"[{subject}] chapter {ch['chapter_no']} ({ch['chapter_title']}) done -> {len(chapter_records)} questions")

def main():
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(GEMINI_MODEL)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    state = load_state()
    reset_daily_counter_if_needed(state)

    chapters_path = DATA_DIR / "chapters.json"
    chapters_out = json.loads(chapters_path.read_text()) if chapters_path.exists() else []

    questions_path = DATA_DIR / "questions.jsonl"
    with open(questions_path, "a", encoding="utf-8") as questions_fh:
        for pdf_cfg in PDFS:
            process_pdf(pdf_cfg, state, model, chapters_out, questions_fh)

    chapters_path.write_text(json.dumps(chapters_out, indent=2, ensure_ascii=False))
    save_state(state)
    print("All done (or paused at daily limit -- just re-run this script to resume).")

if __name__ == "__main__":
    main()
