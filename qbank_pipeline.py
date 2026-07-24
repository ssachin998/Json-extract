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

MAX_CALLS_PER_DAY = 950         # safety buffer under your 1000/day cap
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

def extract_toc_chapters(pdf_path, toc_page_range=(1, 3)):
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

def _resolve(obj):
    """Follow an IndirectObject reference; pass through anything else."""
    return obj.get_object() if hasattr(obj, "get_object") else obj

def _page_xobjects(page):
    """Return the page's /Resources /XObject dict (resolved), or {}."""
    res = _resolve(page.get("/Resources"))
    if not res:
        return {}
    xobjs = _resolve(res.get("/XObject"))
    return xobjs if xobjs else {}

def find_watermark_object_id(pdf_path, sample_pages=30):
    reader = PdfReader(pdf_path)
    counts = {}
    n = min(sample_pages, len(reader.pages))
    for i in range(n):
        for name, ref in _page_xobjects(reader.pages[i]).items():
            obj = _resolve(ref)
            if obj.get("/Subtype") != "/Image":
                continue
            obj_id = getattr(ref, "idnum", None)
            if obj_id is None:
                continue  # inline/direct image -- can't track by object id
            counts[obj_id] = counts.get(obj_id, 0) + 1
    if not counts:
        return None
    # whichever object ID appears on (almost) every sampled page = watermark
    watermark_id = max(counts, key=counts.get)
    if counts[watermark_id] < n * 0.5:
        return None  # no dominant repeated image -> no watermark to exclude
    return watermark_id

def _decode_image_fallback(obj):
    """Best-effort decode of an image XObject using its raw stream, for
    cases pypdf's own page.images accessor can't handle. Covers the common
    FlateDecode DeviceRGB/DeviceGray case; returns None otherwise."""
    try:
        w, h = int(obj["/Width"]), int(obj["/Height"])
        mode = {"/DeviceRGB": "RGB", "/DeviceGray": "L"}.get(str(obj.get("/ColorSpace")))
        if mode is None or w <= 0 or h <= 0:
            return None
        data = obj.get_data()  # pypdf applies filters (Flate etc.)
        need = w * h * len(mode)
        if isinstance(data, bytes) and len(data) >= need:
            return Image.frombytes(mode, (w, h), data[:need])
        return None
    except Exception:
        return None

def extract_real_images(pdf_path, file_page, watermark_id, subject, out_dir):
    """
    Extracts every embedded image on file_page EXCEPT the watermark object.
    Returns a list of saved relative paths ("SUBJECT/filename.webp") --
    exactly one entry per saved file (no duplicates, no watermarks).
    Caller is responsible for deciding which question/option/solution each
    belongs to (Gemini's response should say which figure goes where).
    """
    reader = PdfReader(pdf_path)
    if not (1 <= file_page <= len(reader.pages)):
        print(f"  [WARN] extract_real_images: page {file_page} out of range "
              f"(pdf has {len(reader.pages)} pages) -- skipping")
        return []
    page = reader.pages[file_page - 1]
    saved = []
    (out_dir / subject).mkdir(parents=True, exist_ok=True)
    for name, ref in _page_xobjects(page).items():
        obj = _resolve(ref)
        if obj.get("/Subtype") != "/Image":
            continue
        obj_id = getattr(ref, "idnum", None)
        if watermark_id is not None and obj_id == watermark_id:
            continue  # the watermark -- never save it as a question figure
        # Save exactly THIS image object. NOTE: don't shell out to
        # `pdfimages -f P -l P` per object here -- it dumps EVERY image on
        # the page (watermark included) under one prefix each time, so
        # looping over N real images re-extracts the whole page N times
        # and overwrites/duplicates output files. Decode this one object
        # directly instead.
        try:
            im = page.images[name].image
        except Exception:
            im = _decode_image_fallback(obj)
        if im is None:
            print(f"  [WARN] could not decode image {name} (obj {obj_id}) on "
                  f"page {file_page} -- skipping")
            continue
        if im.size[0] * im.size[1] < 5000:
            continue  # skip tiny noise images
        stem = obj_id if obj_id is not None else str(name).strip("/")
        fname = f"{subject}-p{file_page}-{stem}.webp"
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
- ANSWER KEY TABLES ARE CRITICAL -- READ THIS CAREFULLY: any table you see
  with a "Question No." / "Q.No" column and a "Correct Option" / "Answer"
  column, however many rows it has, MUST produce one JSON entry PER ROW.
  Do not skip rows, do not summarize the table, do not describe it in prose.
  Example: if the table shows
      5 -> b
      6 -> c
      7 -> a
  you must output all three as separate entries:
      {"q_no": 5, "question_text": null, "options": null, "correct_option": "b", "solution_text": null, "tables": [], "has_figure_in_question": false, "has_figure_in_solution": false}
      {"q_no": 6, ..., "correct_option": "c", ...}
      {"q_no": 7, ..., "correct_option": "a", ...}
  A table with 20 rows means 20 separate JSON entries, not one summary entry.
- If a page only contains solutions, return entries with only "solution_text"
  (and "tables" if present) filled, matched to the right q_no.
- If a question's options are split across two pages (e.g. A/B on this page,
  C/D on the next), only include the options actually visible on THIS batch
  of pages -- do not guess or invent the missing ones. They will be merged
  with the other batch's output automatically.
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
        raw_qn = item.get("q_no")
        if raw_qn is None:
            print(f"  [WARN] Gemini returned an item with no q_no, skipping: {str(item)[:200]}")
            continue
        try:
            qn = int(raw_qn)  # Gemini's JSON sometimes returns q_no as a
                               # string ("7") and sometimes as a number (7);
                               # force a consistent type so later sorting
                               # never compares int against str.
        except (TypeError, ValueError):
            print(f"  [WARN] Gemini returned a non-numeric q_no ({raw_qn!r}), skipping")
            continue
        rec = existing.setdefault(qn, {
            "q_no": qn, "question_text": None, "options": None,
            "correct_option": None, "solution_text": None, "tables": [],
            "has_figure_in_question": False, "has_figure_in_solution": False,
        })
        for k in ["question_text", "solution_text"]:
            if item.get(k):
                rec[k] = item[k]

        # Options can arrive across TWO different batches when a question
        # straddles a page break (e.g. options A/B on one page, C/D on the
        # next). Merge by option letter instead of overwriting the whole
        # dict, or the earlier batch's options get silently discarded.
        # Also normalize every option letter to uppercase here, since Gemini
        # (and the source PDF itself) mixes "a)" and "A." lettering -- if we
        # don't normalize once, centrally, correct_options ("D") will fail
        # to match options[].id ("d") later and the answer will look wrong
        # in the app even though the data is technically all there.
        if item.get("options"):
            if rec["options"] is None:
                rec["options"] = {}
            for opt_id, opt_text in item["options"].items():
                rec["options"][str(opt_id).strip().upper()] = opt_text

        if item.get("correct_option"):
            rec["correct_option"] = str(item["correct_option"]).strip().upper()

        if item.get("tables"):
            rec["tables"].extend(item["tables"])
        rec["has_figure_in_question"] = rec["has_figure_in_question"] or item.get("has_figure_in_question", False)
        rec["has_figure_in_solution"] = rec["has_figure_in_solution"] or item.get("has_figure_in_solution", False)
    return existing

def build_final_question(subject, chapter_id, chapter_no, q_no, rec, image_files):
    qid = f"{subject}-{chapter_no:03d}-{q_no:03d}"

    def valid_images(imgs, kind):
        out = []
        for f in imgs:
            if IMG_PATH_RE.match(f):
                out.append({"type": "figure", "file": f})
            else:
                print(f"  [WARN] Dropping malformed {kind} image path for {qid}: {f}")
        return out

    q_images = valid_images(image_files.get("question", []), "question")
    sol_images = valid_images(image_files.get("solution", []), "solution")
    tables = [{"type": t.get("type", "table"), "markdown": t["markdown"], "file": None}
              for t in rec.get("tables", [])]

    return {
        "id": qid,
        "subject": subject,
        "chapter_id": chapter_id,
        "question": {"text": rec["question_text"], "images": q_images},
        "options": [{"id": str(k).strip().upper(), "text": v, "images": []} for k, v in (rec["options"] or {}).items()],
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
                if not imgs:
                    continue
                # NOTE: simple version -- attaches any image found on a page to
                # whichever q_no Gemini flagged as needing one (question OR
                # solution figure) and doesn't have one of that kind yet.
                # Review this mapping manually for pages with multiple figures.
                assigned = False
                for qn, rec in chapter_records.items():
                    entry = image_files_by_q.setdefault(qn, {"question": [], "solution": []})
                    needs_q_img = rec.get("has_figure_in_question") and not entry["question"]
                    needs_sol_img = rec.get("has_figure_in_solution") and not entry["solution"]
                    if not (needs_q_img or needs_sol_img):
                        continue
                    kind = "Q" if needs_q_img else "SOL"
                    # LOCKED structure: ALL figure types live together in
                    # assets/questions/{SUBJECT}/ -- one folder per subject,
                    # type is told only by the filename suffix:
                    #   {id}_Q_01.webp  {id}_SOL_01.webp
                    #   {id}_OPT_A_01.webp  {id}_TABLE_01.webp
                    # Do NOT create assets/solutions/, assets/options/ or
                    # assets/tables/ -- the app relies on this convention.
                    qid = f"{subject}-{ch['chapter_no']:03d}-{qn:03d}"
                    renamed = []
                    for idx, old_rel in enumerate(imgs, start=1):
                        old_path = ASSETS_DIR / "questions" / old_rel
                        new_name = f"{qid}_{kind}_{idx:02d}.webp"
                        new_rel = f"{subject}/{new_name}"
                        new_path = ASSETS_DIR / "questions" / subject / new_name
                        old_path.rename(new_path)
                        renamed.append(new_rel)
                    entry["question" if kind == "Q" else "solution"] = renamed
                    assigned = True
                    break  # this page's image(s) assigned; don't also hand it to a later question
                if not assigned:
                    print(f"  [WARN] Page {file_page_num}: extracted image(s) {imgs} but no "
                          f"question/solution in this chapter claimed one -- left unmatched "
                          f"under its temp filename for manual review.")

        for qn, rec in sorted(chapter_records.items(), key=lambda x: x[0]):
            final_q = build_final_question(
                subject, chapter_id, ch["chapter_no"], qn, rec,
                image_files_by_q.get(qn, {"question": [], "solution": []})
            )
            questions_fh.write(json.dumps(final_q, ensure_ascii=False) + "\n")
            questions_fh.flush()

        progress["chapters_done"].append(chapter_id)
        save_state(state)
        # Persist chapters.json incrementally too. main() also writes it at the
        # end, but if we exit early (daily Gemini limit -> sys.exit, crash,
        # redeploy) that final write never happens -- and since completed
        # chapters are in chapters_done, the next run would skip them and
        # they'd be permanently missing from chapters.json.
        chapters_path = DATA_DIR / "chapters.json"
        chapters_path.write_text(json.dumps(chapters_out, indent=2, ensure_ascii=False))
        n_no_answer = sum(1 for r in chapter_records.values() if not r.get("correct_option"))
        n_no_solution = sum(1 for r in chapter_records.values() if not r.get("solution_text"))
        print(f"[{subject}] chapter {ch['chapter_no']} ({ch['chapter_title']}) done -> "
              f"{len(chapter_records)} questions ({n_no_answer} missing answer, {n_no_solution} missing solution)")

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
