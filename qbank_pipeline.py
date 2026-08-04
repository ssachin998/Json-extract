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

import difflib
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
import pytesseract

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

MAX_CALLS_PER_DAY = 1400        # self-imposed brake with ~7% buffer under the
                                 # free tier's 1500 requests/day (per project;
                                 # buffer covers shared-key use by other bots,
                                 # page-by-page retries, and quota-window vs
                                 # server-date misalignment). NOT Google's limit.
PAGES_PER_GEMINI_CALL = 6       # tune this: more pages/call = fewer calls,
                                 # but keep it small enough that Gemini can
                                 # read every question accurately
BATCH_OVERLAP_PAGES = 2         # consecutive batches share 2 pages, so a
                                 # question/solution split across a batch
                                 # boundary is seen WHOLE (with its q_no) in at
                                 # least one call -- see ROOT_CAUSE_ANALYSIS.md
                                 # RC-3. Step size = 6-2 = 4 new pages/call.
                                 # Merge by q_no makes re-extraction idempotent.
TARGETED_RETRY_MAX_ROUNDS = 2   # after a chapter's normal pass, up to this many
                                 # small focused re-asks for answer/options fields
                                 # still missing (merged from target_retry_patch.py)
SOLUTION_GATE_MIN_SHARE = 0.6   # if >=60% of a chapter's questions already have
                                 # solution text, the book DOES print explanations
                                 # here -> remaining solution gaps are extraction
                                 # losses and become retry-eligible. Replaces the
                                 # blanket solution-exclusion (RC-4), which the
                                 # 2026-07-25 run-2 log REFUTED: run-1's "answer-key
                                 # only" chapters (ch4/ch6) came back 0-missing on
                                 # the same pages, and ch11's count changed 5->8
                                 # between runs -- proof of nondeterministic model
                                 # drops, not absent print.
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"   # confirmed working model from your bot's config

MIN_SECONDS_BETWEEN_CALLS = 5   # free tier = ~15 requests/minute (1 per 4s).
                                 # Without pacing, back-to-back calls (v2's Q/A/S
                                 # passes, single-page retries, crop ladders) bust
                                 # the RPM window instantly -> 429 bursts. 5s
                                 # spacing caps a run at 12 RPM: bursts disappear
                                 # and the 65s backoff ladder stops firing.
_last_call_ts = 0.0

# --- SECTION-AWARE BATCHING (run-6 user ask: "pehle questions ek saath, phir
# answer table, phir solutions") ------------------------------------------
# Instead of walking the chapter in fixed 6-page windows (33% of pages re-sent
# as overlap), the Solutions-section start is detected ONCE from the text
# layer and the chapter is sent in section-sized windows: the whole
# questions+answers stretch in LARGE windows (1-2 calls -> every question
# shares one context, so boundary splits and cross-window option drops
# disappear, and fewer calls = less 15-RPM pressure + the daily quota lasts),
# and the Solutions section in recitation-safe chunks (long verbatim spans
# are what trigger finish_reason=4, page 218 class). Pass activation stays
# probe-based -- the section labels only SIZE the windows, they never skip a
# pass, so a mislabeled page can't lose a question.
QUESTIONS_CHUNK_PAGES = 10     # a chapter's question section usually fits in
                               # 1-2 calls; all questions share one context
SOLUTIONS_CHUNK_PAGES = 5      # smaller spans = recitation-safe (page 218
                               # class: a whole-section S-pass fails as a unit)
SECTION_OVERLAP_PAGES = 1      # tiny intra-section overlap (a question split
                               # across a chunk boundary is still seen whole);
                               # overlap drops from 2/6 (33%) to 1/10 (10%) --
                               # the token waste the old fixed windows had

def _pace_gemini_call():
    """Sleep just enough that consecutive Gemini requests stay
    MIN_SECONDS_BETWEEN_CALLS apart. Called at the two choke points EVERY
    request flows through: call_gemini_on_pages and
    gemini_json_call_splitting's one_call."""
    global _last_call_ts
    gap = time.time() - _last_call_ts
    if gap < MIN_SECONDS_BETWEEN_CALLS:
        time.sleep(MIN_SECONDS_BETWEEN_CALLS - gap)
    _last_call_ts = time.time()

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

def write_chapters(path, chapters_out):
    """Write chapters.json deduplicated by chapter_id (last entry wins).
    Dedup matters when a chapter is re-processed after manual state surgery
    (removing its id from chapters_done to force re-extraction): the id gets
    appended again while an older entry is already in chapters.json."""
    uniq = {}
    for c in chapters_out:
        uniq[c["chapter_id"]] = c
    path.write_text(json.dumps(list(uniq.values()), indent=2, ensure_ascii=False))


def write_chapter_file(subject, chapter_id, chapter_rows):
    """Per-chapter output file, written the moment ONE chapter FULLY completes
    (batches, orphans, image ladder, drain, sweep, targeted retry -- every
    process) and BEFORE the next chapter starts. Proves per-chapter closure
    at a glance and lets the consuming app load chapters individually."""
    d = DATA_DIR / "by_chapter"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{chapter_id}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chapter_rows),
        encoding="utf-8")


def build_subject_bundle(subject, chapters_out):
    """All chapters done -> bundle everything under subjects/{SUBJECT_NAME}/:
    chapters.json (this subject only), questions.jsonl (concat of the
    per-chapter files, in chapter order) and chapters/{CH}.jsonl copies.
    Additive convenience layer -- data/questions.jsonl stays the master."""
    src = DATA_DIR / "by_chapter"
    ch_files = sorted(src.glob(f"{subject}-*.jsonl")) if src.exists() else []
    root = OUTPUT_ROOT / "subjects" / subject
    (root / "chapters").mkdir(parents=True, exist_ok=True)
    combined = []
    for f in ch_files:
        txt = f.read_text(encoding="utf-8")
        (root / "chapters" / f.name).write_text(txt, encoding="utf-8")
        combined.append(txt)
    (root / "questions.jsonl").write_text("".join(combined), encoding="utf-8")
    mine = [c for c in chapters_out if c.get("subject") == subject]
    (root / "chapters.json").write_text(json.dumps(mine, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    print(f"[{subject}] bundle ready -> subjects/{subject}/ "
          f"({len(ch_files)} chapter file(s) + chapters.json + questions.jsonl)")

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
    seen_ids = set()  # some PDFs alias the SAME image object under two XObject
                       # names on one page -> would return the same path twice,
                       # and the second rename in process_pdf crashes with
                       # FileNotFoundError (observed in prod on PSY p264)
    for name, ref in _page_xobjects(page).items():
        obj = _resolve(ref)
        if obj.get("/Subtype") != "/Image":
            continue
        obj_id = getattr(ref, "idnum", None)
        dedupe_key = obj_id if obj_id is not None else str(name)
        if dedupe_key in seen_ids:
            continue  # alias of an image already saved from this page
        seen_ids.add(dedupe_key)
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
- If the text at the top of the FIRST page is clearly the continuation of a
  question, options or a solution from BEFORE these pages (starts mid-sentence
  and no question number is visible), STILL return it as one item with
  "q_no": null and the visible fragment under "solution_text"/"question_text".
  Never invent a question number -- the pipeline salvages these fragments for
  review instead of guessing.
- CONTEXT HANDLING: a "CONTEXT FROM PREVIOUS BATCH" text block may precede
  the page images (the Gemini API is stateless, so continuity context is
  injected manually into every request). Use it ONLY to continue the
  referenced item under its original q_no -- never output that context text
  as a new item. Some leading page-images may be OVERLAP from the previous
  batch, provided purely as continuity context: extract normally, and if an
  item visibly SPANS from an overlap page into the new pages, combine both
  sides into ONE complete item under its printed q_no.
- BATCH META (required): after the last question object, append ONE extra
  control object describing how the LAST page of this batch ends:
  {"_batch_meta": {"last_q_no": <int or null>,
                   "ends_mid_content": true|false,
                   "cut_part": "question"|"options"|"solution"|null,
                   "tail_text": "<verbatim last ~25 words at the bottom of
                                 the last page, else empty string>"}}
  ends_mid_content = true ONLY when the last question's text, options or
  solution is visibly cut off at the bottom of the last page (must continue
  on the following page).
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

# ============================================================
# V2 — MULTI-PHASE (3-PASS) ARCHITECTURE
# Same pages, three small focused asks instead of one big ask.
# Why: the v1 single-pass prompt made the model juggle stems+options,
# answer keys AND solutions in one response -- its per-call attention
# budget split three ways, causing context bleeding (wrong-owner stems),
# glued solution blobs and dropped key rows. Small focused asks are
# measurably more accurate (the same principle as targeted_retry).
#
# QUOTA NOTE: this is NOT "3x every call". A zero-token pdftotext probe
# decides per batch which passes are even worth a call:
#   * questions-section batch -> Q-pass only (A/S skipped)
#   * solutions-section batch -> S-pass only (Q skipped, sticky)
#   * batch whose pages print an answer-key table -> +A-pass
# On the trial book this lands very close to v1's total call count while
# each call is narrower and cleaner.
# ============================================================

PIPELINE_TAG = "v2-3pass"

_BATCH_META_BLOCK = """- BATCH META (required): after the last object, append ONE extra control
  object describing how the LAST page of this batch ends:
  {"_batch_meta": {"last_q_no": <int or null>,
                   "ends_mid_content": true|false,
                   "cut_part": "question"|"options"|"solution"|null,
                   "tail_text": "<verbatim last ~25 words at the bottom of
                                 the last page, else empty string>"}}
- FIGURE MAP (required whenever ANY figure, photo, diagram or chart is
  visible on these pages): append ONE more control object
  {"_figure_map": [{"q_no": <int|null>, "slot": "question"|"solution"|null}, ...]}
  with EXACTLY ONE entry per figure, in top-to-bottom reading order page by
  page. q_no = the question the figure belongs to (null if it is
  decorative/unrelated/watermark). slot = "question" if the figure appears
  with or above the question stem, "solution" if it appears inside that
  question's explanation region (null when q_no is null). Every visible
  figure MUST have an entry -- the pipeline uses this map to attach each
  extracted image to its question. If no figures at all: {"_figure_map": []}
- Output ONLY the JSON array, no commentary, no markdown code fences.
"""

SCHEMA_PROMPT_Q = """You are extracting MCQ QUESTIONS from scanned textbook pages into strict JSON.
This is the QUESTION-ONLY pass. Extract question stems, options and any table
that is part of a QUESTION. DO NOT extract answers or solutions/explanations
in this pass -- a separate pass handles those; never copy solution prose here.

Return a JSON array. Each element is one question:
{
  "q_no": <question number as printed>,
  "question_text": "..." | null,
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."} | null,
  "correct_option": null,
  "solution_text": null,
  "tables": [{"type": "short_label", "markdown": "| col | col |\\n|---|---|\\n..."}],
  "has_figure_in_question": true|false,
  "has_figure_in_solution": false
}

Rules:
- Preserve every word verbatim. Do NOT summarize or paraphrase.
- NEVER invent a question number. If text at the top of the FIRST page is a
  continuation from BEFORE these pages (starts mid-sentence, no number
  visible), FIRST use the OVERLAP/CONTEXT pages to determine which question
  it continues (the preceding overlap page usually shows that question's
  number) and return it under that q_no; ONLY when ownership cannot be
  established from the context, return it with "q_no": null and the visible
  fragment under "question_text"/"options".
- If a question's options are split across two pages, only include the
  options actually visible on THIS batch -- they merge automatically.
- If a visible line is clearly an answer-letter line or explanation prose
  (not a stem/option), SKIP it -- do not force it into an item.
- CONTEXT HANDLING: a "CONTEXT FROM PREVIOUS BATCH" text block may precede
  the page images. Use it ONLY to continue the referenced item under its
  original q_no -- never output that context text as a new item. Leading
  page-images may be OVERLAP from the previous batch (continuity only):
  if a stem/options visibly SPAN from an overlap page into new pages,
  combine both sides into ONE complete item under its printed q_no.
""" + _BATCH_META_BLOCK

SCHEMA_PROMPT_A = """You are reading ANSWER KEYS from scanned textbook pages into strict JSON.
This is the ANSWER-KEY-ONLY pass. Your ONLY job: extract the mapping from
question number to correct option letter, wherever it is printed on these
pages (dedicated key tables, or answer lines printed beside questions).

Return a JSON array with ONE entry PER ROW you can see:
{"q_no": <int>, "question_text": null, "options": null,
 "correct_option": "A" | "B" | "C" | "D",
 "solution_text": null, "tables": [],
 "has_figure_in_question": false, "has_figure_in_solution": false}

Rules:
- READ THIS CAREFULLY: any table with a "Question No." / "Q.No" column and a
  "Correct Option" / "Answer" column -- however many rows -- MUST produce one
  JSON entry PER ROW. A 20-row table = 20 entries, not one summary entry.
  Do not skip rows, do not summarize the table, do not describe it in prose.
- Normalise the letter to UPPERCASE A/B/C/D. If a row's letter is illegible,
  SKIP that row entirely -- never guess.
- Preserve row order and letters exactly as printed (verbatim accuracy).
- Return ONLY the rows you can actually see on THESE pages. If no key/answer
  is printed here, return an empty array [].
""" + _BATCH_META_BLOCK.replace(
    '"cut_part": "question"|"options"|"solution"|null', '"cut_part": null')

SCHEMA_PROMPT_S = """You are extracting printed SOLUTIONS / EXPLANATIONS from scanned textbook
pages into strict JSON. This is the SOLUTION-ONLY pass. DO NOT extract
question stems or options in this pass -- a separate pass handles those.

Return a JSON array. Each element is one question's solution:
{
  "q_no": <question number the solution is printed for>,
  "question_text": null,
  "options": null,
  "correct_option": "A" | "B" | "C" | "D" | null,   // only if "Ans: B" style is printed
  "solution_text": "..." ,
  "tables": [{"type": "short_label", "markdown": "| col | col |\\n|---|---|\\n..."}],
  "has_figure_in_question": false,
  "has_figure_in_solution": true|false
}

Rules:
- Preserve every word verbatim. Do NOT summarize or paraphrase.
- NEVER invent a question number -- use ONLY numbers explicitly printed with
  the solution (e.g. "Solution to Question 4:" -> q_no 4) or PROVEN by the
  OVERLAP/CONTEXT pages (e.g. the "Solution to Question 4:" header visible
  at the bottom of the preceding overlap page). If the top of the FIRST
  page continues a solution from BEFORE these pages with no number visible,
  FIRST use the OVERLAP/CONTEXT pages to determine which question it
  continues and return it under that q_no; ONLY when ownership cannot be
  established, return it with "q_no": null under "solution_text".
- ONE ENTRY PER QUESTION. The text of EACH question's solution goes ONLY into
  that question's own entry. Text printed after a "Solution to Question N:"
  header belongs to q_no N, never to an earlier entry.
- Any table inside a solution must become a markdown table string in "tables".
- CONTEXT HANDLING: a "CONTEXT FROM PREVIOUS BATCH" text block may precede
  the page images. Use it ONLY to continue the referenced solution under its
  original q_no -- never output that context text as a new item. Leading
  page-images may be OVERLAP (continuity only): if a solution visibly SPANS
  from an overlap page into new pages, combine both sides into ONE complete
  entry under its printed q_no.
""" + _BATCH_META_BLOCK

# Zero-token pdftotext probes that decide which passes a batch even needs.
KEY_TABLE_PROBE_RE = re.compile(
    r"(question\s*no|q\.?\s*no)[^\n]{0,40}(correct\s*option|answer)"
    r"|answer\s*key", re.IGNORECASE)
SOLUTION_PROBE_RE = re.compile(
    r"solution\s+to\s+question\s+\d{1,3}\s*:", re.IGNORECASE)

# Claude's mandated Task-4 marker (kept verbatim for audit parity) -- the
# SAFE clipping built around it lives in clip_pass_solutions(); the naive
# "cut at first match" version is NOT used anywhere because it can delete
# unique neighbour content when the model fails to emit sibling items.
SOLUTION_MARKER_RE = re.compile(r'(?i)solution\s+to\s+question\s+(\d+)\s*:', re.MULTILINE)


def clip_pass_solutions(items):
    """V2 S-pass response parser guard (Task 4, hardened).

    For every item, clip a foreign "Solution to Question N:" tail ONLY when
    the tail is provably redundant -- i.e. the numbered question appears as
    its OWN sibling item in the same response (the model emitted both, so
    nothing unique is lost). Steps per item:
      1. strip LEADING "Solution to Question N:" furniture headers (all of
         them -- never clip-to-empty, which the naive version would do);
      2. scan for an embedded header naming a DIFFERENT q_no that exists as a
         sibling item -> hard-cut before it;
      3. an embedded header naming a q_no with NO sibling item is LEFT
         INTACT (possibly unique neighbour content) -- the chapter-level
         integrity sweep trims it later with a chapter-wide donor proof.
    Returns (items, n_clipped)."""
    def _qn(it):
        try:
            q = it.get("q_no")
            return int(q) if q is not None and not isinstance(q, bool) else None
        except (TypeError, ValueError):
            return None

    sibling_qns = {q for q in (_qn(it) for it in items) if q is not None}
    n_clipped = 0
    for it in items:
        s = it.get("solution_text") or ""
        if not s:
            continue
        orig = s
        own = _qn(it)
        # 1. leading furniture headers
        while True:
            m = re.match(r"\s*Solution\s+to\s+Question\s+\d{1,3}\s*[:.\-]?\s*",
                         s, re.IGNORECASE)
            if not m:
                break
            s = s[m.end():]
        # 2. provably-redundant foreign tail
        for m in SOLUTION_MARKER_RE.finditer(s):
            if m.start() == 0:
                continue
            n = int(m.group(1))
            if own is not None and n == own:
                continue
            if n in sibling_qns:
                s = s[:m.start()].rstrip()
                n_clipped += 1
            break  # 3. donor-less tails are left for the chapter sweep
        if s != orig:
            it["solution_text"] = s
    return items, n_clipped

def parse_gemini_json_array(text):
    """Parse Gemini's structured response without discarding valid records.

    Gemini occasionally emits two adjacent JSON arrays despite the prompt's
    "one array only" instruction. ``json.loads`` then raises ``Extra data``;
    previously that made a healthy six-page batch fall back to six expensive
    single-page requests, and an individual page could still be lost. Accept
    consecutive complete arrays (or objects) while rejecting malformed tails.
    """
    clean = re.sub(r"```(?:json)?", "", (text or "").strip(),
                   flags=re.IGNORECASE).strip()
    if not clean:
        raise ValueError("Gemini returned an empty JSON response")

    # Some otherwise-valid answers start with prose such as "Here is the
    # JSON:". Recover the first JSON container rather than discarding a full
    # batch and retrying every page individually.
    starts = [i for i in (clean.find("["), clean.find("{")) if i >= 0]
    if starts:
        clean = clean[min(starts):]

    decoder = json.JSONDecoder()
    values, pos = [], 0
    length = len(clean)
    while pos < length:
        while pos < length and clean[pos].isspace():
            pos += 1
        if pos >= length:
            break
        try:
            value, end = decoder.raw_decode(clean, pos)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid Gemini JSON near character {pos}: {exc.msg}") from exc
        if isinstance(value, list):
            values.extend(value)
        elif isinstance(value, dict):
            # Tolerate newline-delimited objects from a model that ignored the
            # array wrapper. Downstream validation still handles every item.
            values.append(value)
        else:
            raise ValueError("Gemini JSON must contain an array or object")
        pos = end
    return values


def _ocr_content_owner(item, chapter_records):
    """Prefer q_no, then identify OCR scraps from option-content evidence."""
    try:
        qn = int(item.get("q_no"))
        if qn in chapter_records:
            return qn
    except (TypeError, ValueError):
        pass
    values = " ".join(str(v or "") for v in (item.get("options") or {}).values()).lower()
    tokens = {t for t in re.findall(r"\w+", values) if len(t) > 3}
    if not tokens:
        return None
    best_qn, best_score = None, 0.0
    for qn, rec in chapter_records.items():
        opt_text = " ".join(str(v or "") for v in (rec.get("options") or {}).values()).lower()
        opt_tokens = {t for t in re.findall(r"\w+", opt_text) if len(t) > 3}
        score = len(tokens & opt_tokens) / max(1, min(len(tokens), len(opt_tokens)))
        if score > best_score:
            best_qn, best_score = qn, score
    return best_qn if best_score >= 0.5 else None


def _novel_solution_tail(existing, incoming):
    """Return only the unseen suffix of OCR text; never append a re-read body."""
    if not incoming or _frag_mostly_present(incoming, existing, 0.85):
        return ""
    # Select the match that consumes the FURTHEST part of incoming text, not
    # merely its largest isolated block. This removes an entire repeated
    # solution before retaining its continuation after the truncation point.
    blocks = difflib.SequenceMatcher(None, existing.lower(), incoming.lower()).get_matching_blocks()
    usable = [b for b in blocks if b.size >= 30 and b.b < len(incoming) * 0.75]
    if usable:
        end = max(b.b + b.size for b in usable)
        tail = incoming[end:].lstrip(" \n,.;:")
        if tail:
            return tail
    return ""  # uncertain overlap is safer than duplicating a full solution


def is_recitation_risk_solution_page(pdf_path, page_no):
    """Route printed sensitive solution pages away from vision generation."""
    text = pdftotext_page(pdf_path, page_no)
    if not re.search(r"Solution\s+to\s+Question\s+\d{1,3}", text, re.I):
        return False
    risk = r"sexual|rape|genital|vulva|penis|assault|suicide|homicide|abuse|forensic|injury"
    return bool(re.search(risk, text, re.I))


def _recover_ocr_solution_headers(raw_text, chapter_records):
    """Use Tesseract text directly when printed Solution-to-Question headers
    exist; this avoids asking Gemini to regenerate a blocked page at all."""
    hits = list(re.finditer(r"Solution\s+to\s+Question\s+(\d{1,3})\s*[:.]?", raw_text, re.I))
    recovered = 0
    for i, hit in enumerate(hits):
        qn = int(hit.group(1)); rec = chapter_records.get(qn)
        if not rec:
            continue
        end = hits[i + 1].start() if i + 1 < len(hits) else len(raw_text)
        segment = raw_text[hit.end():end].strip()
        tail = _novel_solution_tail(rec.get("solution_text") or "", segment)
        if tail:
            rec["solution_text"] = (rec.get("solution_text") or "").rstrip() + "\n" + tail
            recovered += 1
            print(f"  [OCR_FALLBACK] header-spliced continuation to q{qn}")
    return recovered


def normalize_ocr_fallback_item(raw_item):
    """Map OCR-structurer aliases into the merge schema before orphan logic."""
    return {
        "q_no": raw_item.get("q_no") or raw_item.get("question_number"),
        "question_text": raw_item.get("question_text") or raw_item.get("stem") or raw_item.get("topic"),
        "solution_text": raw_item.get("solution_text") or raw_item.get("explanation"),
        "options": raw_item.get("options"),
        "correct_option": raw_item.get("correct_option") or raw_item.get("correct_options"),
        "tables": raw_item.get("tables") or [],
        "has_figure_in_question": bool(raw_item.get("has_figure_in_question")),
        "has_figure_in_solution": bool(raw_item.get("has_figure_in_solution")),
    }


# ---------------------------------------------------------------------------
# Cross-field contamination hardening (run-7 audit): OCR text can carry page
# numbers / watermarks / footers, and a recovered SOLUTION fragment can
# contaminate question_text. Every recovery path below is field-scoped and
# provenance-tagged so a solution recovery can never populate a stem.
# ---------------------------------------------------------------------------
CONTAMINATION_TOKEN_SHARE = 0.8   # >=80% of a stem's tokens in its own
                                  # solution = the "stem" is really solution
                                  # prose (cross-field contamination class)

_OCR_NOISE_LINE_RES = [
    re.compile(r"^\s*[-–—.·]?\s*\d{1,4}\s*[-–—.·]?\s*$"),          # 12 / -12- / 12.
    re.compile(r"^\s*page\s*\d{1,4}\s*(of\s*\d{1,4})?\s*$", re.I),  # Page 12 of 300
    re.compile(r"^\s*(https?://|www\.)\S+\s*$", re.I),              # urls
    re.compile(r"^\s*(©|\(c\)|copyright).*$", re.I),                # copyright
    re.compile(r"^\s*(\[?\s*no\.?\s*\]?\s*)?\d{1,4}\s*$", re.I),    # bare "12"
]

# Explanation-style OPENERS that can never start a real question stem
# (mirrors/extends SOLUTION_STYLE_STEM_RE -- kept here for the contamination
# validator so the two modules stay independent).
_EXPLANATION_START_RE = re.compile(
    r"^\s*(?:option\s+[a-d]\s*[:.)\-]|ans(?:wer)?\s*[:.)\-]|the\s+correct\s+(?:answer|option)\b|"
    r"(?:hence|thus|therefore|so)\s*,\s*(?:the\s+)?(?:correct\s+)?option\b|"
    r"correct\s+answer\s+is\b|the\s+(?:correct\s+)?answer\s+is\b|"
    r"solution\s*[:.)\-]|explanation\s*[:.)\-]|answer\s*[:.)\-]|"
    r"solution\s+to\s+question\s+\d+|explanation\s+of\s+question\s+\d+)",
    re.IGNORECASE)


def _clean_ocr_text(text):
    """Strip page-level noise from OCR text BEFORE it is merged or spliced
    (run-7 hardening #5). Conservative: removes whole lines only, never
    rewrites prose. Detected:
      * standalone page numbers ("12", "- 12 -", "12.")
      * "Page 12 of 300" footers
      * urls, copyright lines, ISBNs
      * a short line repeated >=3 times in the block (running header/footer)
    Medical wording is preserved verbatim."""
    if not text:
        return text
    lines = text.splitlines()
    counts = {}
    for ln in lines:
        s = ln.strip()
        if s and len(s) <= 40:
            counts[s] = counts.get(s, 0) + 1
    out = []
    n_stripped = 0
    for ln in lines:
        s = ln.strip()
        if not s:
            out.append(ln)
            continue
        if any(r.match(s) for r in _OCR_NOISE_LINE_RES):
            n_stripped += 1
            continue
        if counts.get(s, 0) >= 3 and len(s) <= 40:
            # repeated short line (running header/footer) -- but never strip
            # a "Solution to Question N:" header the recovery relies on
            if not re.match(r"Solution\s+to\s+Question\s+\d", s, re.I):
                n_stripped += 1
                continue
        out.append(ln)
    if n_stripped:
        print(f"  [OCR_CLEAN] stripped {n_stripped} page-noise line(s) "
              f"(page numbers / footers / watermarks)")
    return "\n".join(out)


def _stem_reject_reason(qtext, rec=None):
    """Cross-field contamination proof for a would-be question stem
    (run-7 hardening #3/#6). Returns a short reason string, or None when the
    text plausibly IS a stem. A stem is rejected when it:
      1. opens with explanation-style language ("Option A:", "Ans. is B",
         "The correct answer is", "Solution to Question N:" ...);
      2. is substantially contained in the record's OWN solution text
         (>=CONTAMINATION_TOKEN_SHARE of its tokens appear there) -- a real
         stem shares clinical vocabulary but never ~80% of its tokens with
         its own explanation.
    Conservative by design: short/ambiguous text is never rejected here (the
    validator + find_incomplete treat 'missing' as retry-eligible, so a false
    rejection only costs a re-ask, while a false ACCEPT ships corruption)."""
    t = (qtext or "").strip()
    if not t:
        return None
    if _EXPLANATION_START_RE.match(t):
        return "opens with explanation-style language"
    if rec and len(t) >= 60:
        sol = (rec.get("solution_text") or "").strip()
        if sol and _frag_mostly_present(t, sol, CONTAMINATION_TOKEN_SHARE):
            return "stem text substantially contained in this record's own solution"
    return None


# field scopes per recovery pass (run-7 hardening #2: patch-only recovery).
# A recovery response may ONLY modify the fields its pass was invoked to
# recover -- everything else is dropped at the merge boundary.
_RECOVERY_SCOPE = {
    "Q": {"question_text", "options"},
    "A": {"correct_option"},
    # S includes correct_option pragmatically: the printed "Ans: B" line sits
    # INSIDE the solution block and no other pass may ever see this page
    # (recitation-blocked); question_text/options are NEVER touched.
    "S": {"solution_text", "tables", "correct_option"},
}


def _apply_recovery_scope(item, scope, prov):
    """Null every field of a recovered item that its recovery pass is NOT
    allowed to produce (run-7 hardening #2), and tag the item with its
    provenance. scope: None = unrestricted (normal batches)."""
    if scope is not None:
        for f in list(item.keys()):
            if f not in scope and f not in ("q_no", "_prov",
                                            "has_figure_in_question",
                                            "has_figure_in_solution"):
                item[f] = None
    item["_prov"] = prov
    return item


def ocr_fallback_text(image_path):
    """Non-generative final fallback for recitation-blocked page imagery.
    Output is cleaned of page-level noise (page numbers, watermarks,
    footers) before anything merges it (run-7 hardening #5)."""
    raw = pytesseract.image_to_string(Image.open(image_path))
    return _clean_ocr_text(raw)


def call_gemini_text_only(model, prompt):
    """Structure OCR text without resending the recitation-triggering image."""
    _pace_gemini_call()
    resp = model.generate_content([prompt], safety_settings=SAFETY_SETTINGS,
                                  request_options={"retry": None})
    if not getattr(resp, "candidates", None) or not (resp.text or "").strip():
        raise RuntimeError("Empty Gemini text-only OCR restructuring response")
    return parse_gemini_json_array(resp.text)


def call_gemini_on_pages(model, image_paths, context="", prompt=None):
    parts = [prompt or SCHEMA_PROMPT,
             "These are medical/psychiatric educational pages. Clinical references to violence, "
             "sexuality, self-harm, abuse, or forensic scenarios are quoted textbook content; "
             "transcribe them faithfully for educational extraction, without adding advice."]
    if context:
        parts.append(context)  # carry-forward / overlap context (stateless API)
    for p in image_paths:
        parts.append(Image.open(p))
    page_label = ",".join(Path(p).name for p in image_paths)
    _pace_gemini_call()
    try:
        resp = model.generate_content(
            parts,
            safety_settings=SAFETY_SETTINGS,
            request_options={"retry": None},
        )
    except Exception as exc:
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None) or "unknown"
        print(f"  [GEMINI_ERROR] {page_label}: status={status} reason={str(exc)[:240]}")
        raise RuntimeError(f"Gemini API error status={status}: {exc}") from exc

    candidates = getattr(resp, "candidates", None) or []
    feedback = getattr(resp, "prompt_feedback", None)
    if not candidates:
        print(f"  [GEMINI_ERROR] {page_label}: status=ok candidates=0 block_reason={feedback}")
        raise RuntimeError(f"Empty Gemini response; block_reason={feedback}")

    candidate = candidates[0]
    finish_reason = getattr(candidate, "finish_reason", None)
    if finish_reason and str(finish_reason) not in ("1", "STOP"):
        kind = "SAFETY_BLOCKED" if str(finish_reason) in ("8", "PROHIBITED_CONTENT") else "GEMINI_ERROR"
        print(f"  [{kind}] {page_label}: status=ok finish_reason={finish_reason} block_reason={feedback}", flush=True)
        raise RuntimeError(f"Gemini response did not finish normally (finish_reason={finish_reason})")
    try:
        text = resp.text
    except Exception as exc:
        print(f"  [GEMINI_ERROR] {page_label}: status=ok finish_reason={finish_reason} text_unavailable={exc}")
        raise RuntimeError(f"Gemini response text unavailable: {exc}") from exc
    if not (text or "").strip():
        print(f"  [GEMINI_ERROR] {page_label}: status=ok finish_reason={finish_reason} empty_body=true")
        raise RuntimeError("Empty Gemini response body")
    return parse_gemini_json_array(text)

def retry_batch_page_by_page(model, batch, state, ctx=None, prompt=None):
    """A whole-batch failure (RECITATION/safety finish_reason, token limit)
    is usually caused by just ONE page in the batch. Retrying each page
    alone isolates the bad page instead of losing the whole batch's worth
    of questions/answers/solutions (seen in prod: finish_reason=4 killed a
    6-page batch, wiping one chapter's answers and another's solutions).
    Respects the same daily quota and exits cleanly if it's hit.

    A page that fails EVEN ALONE (usually recitation-sensitive content) is
    NO LONGER silently dropped: it is persisted to state["failed_pages"]
    and gets a second-chance call at chapter end (drain_failed_pages) with
    a different prompt framing -- run-2: page 217's skip cost 5 solutions."""
    print(f"  [INFO] retrying {len(batch)} pages one-by-one to isolate the failing page...")
    items = []
    recovered = 0
    for pf in batch:
        reset_daily_counter_if_needed(state)
        if state["calls_today"] >= MAX_CALLS_PER_DAY:
            print("Daily Gemini call limit reached during single-page retry. Saving progress, exiting.")
            save_state(state)
            sys.exit(0)
        try:
            items.extend(call_gemini_on_pages(model, [pf], prompt=prompt))
            state["calls_today"] += 1
            recovered += 1
        except Exception as e2:
            t2 = str(e2)
            if "429" in t2 or "quota" in t2.lower():
                print(f"  [QUOTA] Gemini quota exhausted during retry -- stopping run for now: {e2}")
                save_state(state)
                sys.exit(0)
            print(f"  [WARN] page {pf.name} failed even alone ({e2}) -- queued for second-chance drain")
            entry = {"page_file": pf.name, "true_page": int(pf.stem.split("-")[-1]),
                     "reason": t2[:200], "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
            if ctx:
                # persist WHICH pass failed (run-7 hardening #2): the drain
                # must recover only the fields that pass was supposed to
                # produce (Q->question/options, A->answer, S->solution).
                entry.update({"subject": ctx.get("subject"), "chapter_no": ctx.get("chapter_no"),
                              "chapter_id": ctx.get("chapter_id"),
                              "pass": ctx.get("pass")})
            failed = state.setdefault("failed_pages", [])
            # Q and S passes can fail on the same recitation-blocked page.
            # Drain it once; duplicate entries caused duplicate OCR splices.
            if not any(e.get("chapter_id") == entry.get("chapter_id") and
                       e.get("true_page") == entry.get("true_page") for e in failed):
                failed.append(entry)
    save_state(state)
    print(f"  [INFO] single-page retry: {recovered}/{len(batch)} pages recovered")
    return items

# ============================================================
# FEATURE: targeted gap-retry (run AFTER normal batches + orphan
# recovery, BEFORE writing the chapter's questions to disk)
#
# WHY: even with full page context in one call, Gemini sometimes drops a
# few fields out of a large batch (proven: a 17-row answer-key table fully
# visible in a single call still came back missing rows 9-13; a single
# question's 4 options fully visible on one page came back with only 3).
# This isn't a batching/context bug -- it's the model's own per-call error
# rate on dense extraction tasks. The fix: after the normal pass, check
# what's STILL missing and ask again with a MUCH smaller, narrowly-scoped
# prompt (just the specific gaps) -- small focused asks are consistently
# more accurate than "extract everything on these 6 pages at once".
#
# (Merged in from target_retry_patch.py: config TARGETED_RETRY_MAX_ROUNDS
# sits beside the other constants; the call site is in process_pdf right
# after orphan recovery and before the chapter write loop.)
# ============================================================

def chapter_integrity_sweep(chapter_records, image_files_by_q, subject, chapter_no, stats):
    """Zero-token deterministic pre-retry sweep (run-4 audit RCA classes).
    Runs BEFORE targeted retry so provably-wrong/provably-incomplete fields
    are re-asked in the SAME run instead of shipping. Never destroys content
    without a deterministic proof; every action is written to
    data/integrity_flags.jsonl. Returns the set of q_nos whose solutions
    look truncated (handed to the retry as forced solution re-asks)."""
    forced_solution, flags = set(), []

    def iflag(kind, qn, detail, matched=True, **extra):
        entry = {"kind": kind, "q_no": qn, "chapter_id": f"{subject}-{chapter_no:03d}",
                 "detail": detail, "matched": matched}
        entry.update(extra)
        flags.append(entry)
        _append_jsonl(DATA_DIR / "integrity_flags.jsonl", entry)

    # 1. duplicate-stem pairs (012-001 class): identical/near-identical stems
    #    on two records of one chapter. The record whose stem does NOT cohere
    #    with its own payload is the wrong-owner copy -- strip its stem so
    #    the retry's Gap-1 anchor refills it from the pages.
    qns = sorted(chapter_records)
    stems = {qn: (chapter_records[qn].get("question_text") or "").strip() for qn in qns}
    for i, qa in enumerate(qns):
        for qb in qns[i + 1:]:
            ta, tb = stems.get(qa) or "", stems.get(qb) or ""
            if not ta or not tb or min(len(ta), len(tb)) < 80:
                continue
            sim = difflib.SequenceMatcher(None, ta[:400], tb[:400]).ratio()
            if sim < 0.95:
                continue
            ca = _stem_payload_coherence(ta, chapter_records[qa])
            cb = _stem_payload_coherence(tb, chapter_records[qb])
            if abs(ca - cb) >= STEM_COHERENCE_MARGIN:
                loser, winner = (qa, qb) if ca < cb else (qb, qa)
                chapter_records[loser]["question_text"] = None
                stems[loser] = ""
                stats["dup_stems_stripped"] = stats.get("dup_stems_stripped", 0) + 1
                iflag("duplicate_stem_stripped", loser,
                      f"stem duplicated q{winner} (sim {sim:.2f}) but coherence "
                      f"{min(ca, cb):.2f} vs winner {max(ca, cb):.2f} -- stripped, retry refills",
                      winner=winner, similarity=round(sim, 3))
                print(f"  [SWEEP] q{loser}: stem duplicated q{winner} with worse "
                      f"payload coherence -- stripped for same-run retry")
            else:
                iflag("duplicate_stem_review", qb,
                      f"stem near-duplicates q{qa} (sim {sim:.2f}); coherence tie "
                      f"({ca:.2f} vs {cb:.2f}) -- needs review", matched=False)
                print(f"  [WARN] [SWEEP] q{qa}~q{qb}: near-duplicate stems, coherence "
                      f"undecidable -- logged for review, no data touched")

    # 2. foreign 'Option X:' line glued at a solution's head (009-007 class):
    #    strip it ONLY when the same line already exists verbatim on another
    #    record of this chapter (proves it is a stray duplicate, not content
    #    this question alone owns). Otherwise flag, keep text, retry nothing.
    for qn in qns:
        sol = (chapter_records[qn].get("solution_text") or "").strip()
        if not _foreign_option_line(sol, chapter_records[qn]):
            continue
        head_line = sol.splitlines()[0].strip()
        dup_elsewhere = any(other != qn and head_line
                            and head_line in (chapter_records[other].get("solution_text") or "")
                            for other in qns)
        if dup_elsewhere:
            chapter_records[qn]["solution_text"] = sol[len(sol.splitlines()[0]):].lstrip("\n ")
            stats["foreign_heads_stripped"] = stats.get("foreign_heads_stripped", 0) + 1
            iflag("foreign_option_head_stripped", qn,
                  f"solution began with a foreign 'Option' line that exists verbatim "
                  f"on another record -- stripped: {head_line[:120]!r}")
            print(f"  [SWEEP] q{qn}: stripped foreign 'Option' head (verbatim dup elsewhere)")
        else:
            iflag("foreign_option_head_review", qn,
                  f"solution begins with an 'Option' line its own options cannot own "
                  f"-- kept (unique), needs review: {head_line[:120]!r}", matched=False)
            print(f"  [WARN] [SWEEP] q{qn}: foreign 'Option' head but no verbatim donor "
                  f"-- kept, logged for review")

    # 2b. foreign "Solution to Question N:" dump TAIL (external-audit class,
    #     2026-07-27: on a dense solutions page the model sometimes returns
    #     the FIRST record's item with its own correct solution PLUS the
    #     verbatim solutions of every later question on that page concatenated
    #     after 'Solution to Question 2:' headers; e.g. ch11 q1 carried
    #     q1+q2+...+q8 in one 5689-char blob while q2..q8 ALSO owned their
    #     own correct copies). sanitize_solution_text only trims such a tail
    #     when it duplicates THIS record's own text; a tail holding the
    #     neighbour's UNIQUE solution is kept there by caution. Here the
    #     cross-record proof exists: if the header names a record of THIS
    #     chapter that already owns a non-empty solution, the tail is provably
    #     redundant -> trim at the FIRST such header. Donor-less headers are
    #     left intact (never delete possibly-unique content) and flagged.
    for qn in qns:
        sol = chapter_records[qn].get("solution_text") or ""
        if not sol:
            continue
        for m in SOLUTION_DUMP_HDR_RE.finditer(sol):
            if m.start() <= 2:
                continue  # leading header: sanitize_solution_text strips it at build
            n = int(m.group(1))
            if n == qn:
                continue
            donor_sol = chapter_records.get(n, {}).get("solution_text") or ""
            if donor_sol.strip():
                trimmed = sol[:m.start()].rstrip()
                chapter_records[qn]["solution_text"] = trimmed
                stats["solution_dumps_trimmed"] = stats.get("solution_dumps_trimmed", 0) + 1
                iflag("foreign_solution_dump_trimmed", qn,
                      f"embedded 'Solution to Question {n}:' header at char {m.start()} -- "
                      f"tail trimmed ({len(sol) - len(trimmed)} chars); donor q{n} already owns "
                      f"its solution ({len(donor_sol)} chars) -- redundancy proven")
                print(f"  [SWEEP] q{qn}: trimmed foreign 'Solution to Question {n}:' dump tail "
                      f"({len(trimmed)} chars kept; donor q{n} owns its own solution)")
                if looks_truncated_solution(trimmed, has_tables=bool(chapter_records[qn].get("tables"))):
                    forced_solution.add(qn)
                break
            iflag("foreign_solution_dump_review", qn,
                  f"embedded 'Solution to Question {n}:' header but donor q{n} owns NO "
                  f"solution -- tail kept (may be unique content), needs review",
                  matched=False)
            break

    # 3. truncated-solution suspects (023-007/006-009 class): deterministic
    #    dangling-end / mid-flow-cut patterns -- re-ask the FULL solution.
    for qn in qns:
        rec = chapter_records[qn]
        sol = (rec.get("solution_text") or "")
        if not sol.strip():
            continue
        entry = image_files_by_q.get(qn, {"question": [], "solution": []})
        if looks_truncated_solution(sol, has_tables=bool(rec.get("tables")),
                                    has_images=bool(entry.get("solution"))):
            forced_solution.add(qn)
            iflag("truncated_solution_retry", qn,
                  f"solution looks truncated (...{sol.rstrip()[-50:]!r}) -- forced re-ask")
            print(f"  [SWEEP] q{qn}: solution looks truncated -- targeted retry will re-ask it")

    # 4. over-attributed question images (022-003 class, rows healed by the
    #    recovery path where the rename-time cap never ran).
    for qn in qns:
        entry = image_files_by_q.get(qn)
        if not entry or len(entry.get("question") or []) <= MAX_QUESTION_IMAGES:
            continue
        extras = entry["question"][MAX_QUESTION_IMAGES:]
        del entry["question"][MAX_QUESTION_IMAGES:]
        _append_jsonl(DATA_DIR / "unmatched_images.jsonl",
                      {"subject": subject, "chapter_id": f"{subject}-{chapter_no:03d}",
                       "page": None, "files": extras,
                       "reason": f"over-attribution sweep (> {MAX_QUESTION_IMAGES} question "
                                 f"images on one question) -- de-referenced for review"})
        iflag("question_images_trimmed", qn,
              f"had {len(extras) + MAX_QUESTION_IMAGES} question images; kept first "
              f"{MAX_QUESTION_IMAGES}, de-referenced {len(extras)}")
        print(f"  [SWEEP] q{qn}: de-referenced {len(extras)} over-attributed question "
              f"image(s) -- logged to unmatched_images.jsonl")

    # 4b. over-attributed SOLUTION images (same class on the solution side:
    #     user report -- 7 figures on one solutions page collapsed into 2
    #     solutions; the sweep also heals rows from runs before the cap).
    for qn in qns:
        entry = image_files_by_q.get(qn)
        if not entry or len(entry.get("solution") or []) <= MAX_SOLUTION_IMAGES:
            continue
        extras = entry["solution"][MAX_SOLUTION_IMAGES:]
        del entry["solution"][MAX_SOLUTION_IMAGES:]
        _append_jsonl(DATA_DIR / "unmatched_images.jsonl",
                      {"subject": subject, "chapter_id": f"{subject}-{chapter_no:03d}",
                       "page": None, "files": extras,
                       "reason": f"over-attribution sweep (> {MAX_SOLUTION_IMAGES} solution "
                                 f"images on one question) -- de-referenced for review"})
        iflag("solution_images_trimmed", qn,
              f"had {len(extras) + MAX_SOLUTION_IMAGES} solution images; kept first "
              f"{MAX_SOLUTION_IMAGES}, de-referenced {len(extras)}")
        print(f"  [SWEEP] q{qn}: de-referenced {len(extras)} over-attributed solution "
              f"image(s) -- logged to unmatched_images.jsonl")

    # 5. contaminated stems (run-7 cross-field contamination class): a
    #    question_text that OPENS with explanation language or is
    #    substantially contained in its own solution is solution prose, not a
    #    stem (the audit's pattern: "question_text contains a paragraph from
    #    that question's or a neighbor's solution"). Strip it so the targeted
    #    retry REFILLS the stem from the pages (Gap-1 anchor: the solution
    #    names its question) instead of shipping a populated-but-wrong field.
    #    "Field is populated" is NOT treated as "field is valid".
    for qn in qns:
        rec = chapter_records[qn]
        qt = (rec.get("question_text") or "").strip()
        if not qt:
            continue
        reason = _stem_reject_reason(qt, rec)
        if not reason:
            continue
        chapter_records[qn]["question_text"] = None
        stats["contaminated_stems_stripped"] = stats.get("contaminated_stems_stripped", 0) + 1
        iflag("contaminated_stem_stripped", qn,
              f"question_text was solution prose ({reason}; prov="
              f"{rec.get('_prov', {}).get('question_text')}) -- stripped, "
              f"retry refills the real stem")
        print(f"  [SWEEP] q{qn}: stripped contaminated stem ({reason}) -- "
              f"targeted retry will refill it")

    if flags:
        stats["integrity_flags"] = stats.get("integrity_flags", 0) + len(flags)
    return forced_solution


def find_incomplete_records(chapter_records, force_solution_qns=(), printed_solution_qns=()):
    """
    Returns [(q_no, missing_fields), ...] for records worth retrying.

    force_solution_qns: q_nos whose non-empty solutions the integrity sweep
    judged truncated -- re-asked like a missing solution regardless of the
    60% gate (the book provably printed SOMETHING here; we hold a fragment).

    printed_solution_qns: q_nos whose "Solution to Question N:" header was
    found in the chapter's text layer (chapter_printed_solution_qns). A
    header PROVES the book prints an explanation for that q_no, so a missing
    solution for it is an extraction loss -- retry-eligible even when the
    chapter as a whole sits below the 60% gate (ch25 class: 7/12 = 58%,
    gate suppressed 5 real solutions).

    "answer" and "options" gaps are always retry-worthy: every real MCQ has
    4 options and one marked answer somewhere in the book.

    "solution" gaps are retry-worthy only when chapter-internal evidence
    says the book PRINTS explanations here (>=60% of questions already have
    solution text -- SOLUTION_GATE_MIN_SHARE, or a printed header for that
    specific q_no). Chapters where the book genuinely prints no explanations
    (answer-key-only sections, RC-4) show ~0% coverage and stay protected:
    no quota is wasted chasing content that was never printed.
    """
    incomplete = []
    for qn, rec in chapter_records.items():
        # SEMANTIC COMPLETENESS (run-7 hardening #6): a non-empty
        # question_text that is really solution prose is NOT a valid stem --
        # treat it as missing so the retry replaces it instead of the record
        # shipping a populated-but-wrong field. The sweep strips these before
        # retry; this check is the net for records the sweep never saw.
        qt = (rec.get("question_text") or "").strip()
        if qt and _stem_reject_reason(qt, rec):
            missing = ["question"]
            if not rec.get("correct_option"):
                missing.append("answer")
            options = rec.get("options") or {}
            if len(options) < 4 or any(not str(v or "").strip() for v in options.values()):
                missing.append("options")
            if (rec.get("solution_text") or "").strip() or rec.get("correct_option") or options:
                incomplete.append((qn, missing))
            continue
        if not qt:
            # Stem-less records USED to be skipped here ("nothing to anchor a
            # retry to") -- wrong: a present solution_text/correct_option IS
            # the anchor. The stem and its own solution never share lexical
            # overlap, but the solution names the question it explains, and
            # Gemini can walk back from it (Gap-1: PSY-001-003).
            if (rec.get("solution_text") or "").strip() or rec.get("correct_option"):
                incomplete.append((qn, ["question"]))
            continue  # truly anchorless scraps stay ineligible
        missing = []
        if not rec.get("correct_option"):
            missing.append("answer")
        options = rec.get("options") or {}
        if len(options) < 4 or any(not str(v or "").strip() for v in options.values()):
            missing.append("options")
        if missing:
            incomplete.append((qn, missing))

    n = len(chapter_records)
    n_with_sol = sum(1 for r in chapter_records.values() if (r.get("solution_text") or "").strip())
    book_prints_solutions = n > 0 and n_with_sol / n >= SOLUTION_GATE_MIN_SHARE
    printed = set(printed_solution_qns or ())
    forced = set(force_solution_qns or ())
    if book_prints_solutions or forced or printed:
        by_qn = {qn: missing for qn, missing in incomplete}
        for qn, rec in chapter_records.items():
            truncated = qn in forced and (rec.get("solution_text") or "").strip()
            printed_here = qn in printed
            if rec.get("question_text") and (
                    (not (rec.get("solution_text") or "").strip()
                     and (book_prints_solutions or printed_here))
                    or truncated):
                if qn in by_qn:
                    if "solution" not in by_qn[qn]:
                        by_qn[qn].append("solution")
                else:
                    incomplete.append((qn, ["solution"]))
    return incomplete


def build_targeted_retry_prompt(incomplete_items, chapter_records):
    """Focused retry schema.  Tables must never be returned inside prose."""
    lines = [
        "You already extracted most of this chapter from these SAME pages. Find ONLY the requested missing pieces.",
        "Return ONLY a valid JSON array, beginning with [ and ending with ]. No prose or markdown fences.",
        "Each element uses exactly this schema:",
        '{"q_no": <int>, "question_text": "..."|null, "correct_option": "A"|"B"|"C"|"D"|null, "options": {"A":"...","B":"...","C":"...","D":"..."}|null, "solution_text":"plain prose only"|null, "tables":[{"type":"short label","markdown":"| col | col |\\n|---|---|\\n..."}]}',
        "Every table MUST be in tables[] as markdown. NEVER put pipes, headers, or table rows in solution_text.",
        "Only fill requested fields; use null when not visible. Return [] if none are visible.",
        "MISSING PIECES TO FIND:",
    ]
    for qn, missing in incomplete_items:
        rec = chapter_records[qn]
        qtext = (rec.get("question_text") or "")[:120]
        lines.append(f"Question {qn} (stem begins: {qtext!r}):")
        if "question" in missing:
            lines.append(f"- Return full verbatim question stem and all four options A-D for q{qn}.")
        if "answer" in missing:
            lines.append(f"- Return the correct option letter for q{qn} from the printed answer key.")
        if "options" in missing:
            lines.append(f"- Return all four options A-D for q{qn}; captured letters: {sorted((rec.get('options') or {}).keys())}.")
        if "solution" in missing:
            original = (rec.get("solution_text") or "")[:500]
            lines.append(
                f"- Return only the missing/continuing part of q{qn}'s verbatim solution; put any table only in tables[]. "
                f"Preserve the source's line breaks and bullet-list structure. Existing text (do not repeat): {original!r}")
    return "\n".join(lines)


def _log_blocked_retry_fragment(chapter_id, qn, reason, fragment):
    """Ledger entry for a targeted-retry response that was provably another
    question's solution (wrong-owner guard). The fragment is never merged
    into the record; it stays visible here for review instead of silently
    blending two solutions (PSY-016/017-017 class)."""
    _append_jsonl(DATA_DIR / "integrity_flags.jsonl",
                  {"kind": "retry_foreign_fragment_blocked", "q_no": qn,
                   "chapter_id": chapter_id, "detail": reason,
                   "fragment": (fragment or "")[:600]})


def targeted_retry(model, page_files, chapter_records, state, max_rounds=2,
                   force_solution_qns=None, chapter_id=None, printed_solution_qns=None,
                   stats=None):
    """
    Up to `max_rounds` small, focused re-asks for whatever answer/option
    fields are still missing after normal processing. Sends the chapter's
    full page set again each round (simple and robust -- we don't track
    per-field page provenance) but with a MUCH smaller ask, which is what
    actually improves accuracy, not the page count. Stops early if a round
    makes no progress (no point burning quota repeating the same miss).
    force_solution_qns: integrity-sweep verdicts -- those records' non-empty
    solutions are REPLACED by a longer verbatim re-ask (truncated heal).
    printed_solution_qns: q_nos whose printed 'Solution to Question N:'
    header exists in the chapter -- bypasses the 60% solution-gate for them.
    stats: optional counter dict (chapter stats; used for the contaminated-
    stem block counter). Defaults to a throwaway dict when not passed.
    Returns the total number of fields filled.
    """
    if stats is None:
        stats = {}
    total_fixed = 0
    first_check = True
    forced = set(force_solution_qns or ())
    for round_no in range(1, max_rounds + 1):
        incomplete = find_incomplete_records(chapter_records, force_solution_qns=forced,
                                             printed_solution_qns=printed_solution_qns)
        if not incomplete:
            if first_check:
                # never exit silently again (run-2 learning: "retry skipped"
                # was really "nothing eligible") -- say WHY.
                n_sol_gaps = sum(1 for r in chapter_records.values()
                                 if r.get("question_text") and not (r.get("solution_text") or "").strip())
                if n_sol_gaps:
                    print(f"  [RETRY] nothing eligible: {n_sol_gaps} solution gap(s) suppressed by the "
                          f"60% source-evidence gate (treated as book-printed answer-key-only)")
            break
        first_check = False

        reset_daily_counter_if_needed(state)
        if state["calls_today"] >= MAX_CALLS_PER_DAY:
            print("  [RETRY] daily call limit reached -- stopping retries for now")
            break

        # Include the actual eligibility reason; a non-empty solution alone
        # does not say whether q13 is missing an answer, option, stem, or was
        # explicitly marked as a truncation suspect.
        preview = ", ".join(f"q{qn}[{','.join(missing)}]" for qn, missing in incomplete[:10])
        if len(incomplete) > 10:
            preview += ", ..."
        print(f"  [RETRY] round {round_no}: {len(incomplete)} question(s) still "
              f"incomplete ({preview}) -- sending targeted re-ask")

        prompt = build_targeted_retry_prompt(incomplete, chapter_records)
        # Resilient execution (run-4 lesson, ch9/ch16): never ONE heavy
        # whole-chapter call that fails as a unit -- back off on transient
        # 5xx, split halves->singles on any failure. A recitation-prone
        # single page now only costs that page, not the whole chapter's
        # retry.
        fix_arrays = gemini_json_call_splitting(
            model, prompt, page_files, state,
            label=f" (targeted retry round {round_no})", direct_page_fallback=True)
        if not fix_arrays:
            print("  [RETRY] every sub-call failed even after splitting -- skipping this round")
            continue
        fixes = []
        for arr in fix_arrays:
            if isinstance(arr, list):
                fixes.extend(arr)
        # A targeted response is a PATCH, never permission to replace fields
        # that were already complete. Ignore any extra model fields.
        requested_by_qn = {qn: set(missing) for qn, missing in incomplete}

        fixed_this_round = 0
        for fix in fixes:
            try:
                qn = int(fix.get("q_no"))
            except (TypeError, ValueError):
                continue
            rec = chapter_records.get(qn)
            requested = requested_by_qn.get(qn, set())
            if rec is None or not requested:
                continue
            # PATCH-ONLY + PROVENANCE (run-7 hardening #2/#4): a retry
            # response may only touch the fields that were requested, and
            # every patched field records its provenance (Q_RETRY / A_RETRY /
            # S_RETRY). A Q-retry's returned stem is additionally checked
            # for solution-prose contamination before it is accepted.
            req_prov = ("Q_RETRY" if "question" in requested
                        else ("A_RETRY" if "answer" in requested else "S_RETRY"))
            if "question" in requested and fix.get("question_text") and not (rec.get("question_text") or "").strip():
                incoming_q = str(fix["question_text"]).strip()
                stem_reason = _stem_reject_reason(incoming_q, rec)
                if stem_reason:
                    stats.setdefault("contaminated_stems_blocked", 0)
                    stats["contaminated_stems_blocked"] += 1
                    print(f"  [RETRY] blocked contaminated stem for q{qn} "
                          f"({stem_reason}) -- kept for review, still stem-missing")
                    _log_blocked_retry_fragment(chapter_id, qn, f"contaminated stem: {stem_reason}",
                                                incoming_q)
                else:
                    rec["question_text"] = incoming_q
                    rec["_prov"]["question_text"] = req_prov
                    fixed_this_round += 1
            if "answer" in requested and fix.get("correct_option") and not rec.get("correct_option"):
                rec["correct_option"] = str(fix["correct_option"]).strip().upper()
                rec["_prov"]["correct_option"] = req_prov
                fixed_this_round += 1
            sol_existing = (rec.get("solution_text") or "").strip()
            incoming_text, incoming_tables = _normalize_solution_payload(
                str(fix.get("solution_text") or ""), fix.get("tables") or [], qn)
            if "solution" in requested and incoming_text:
                if not sol_existing:
                    # A foreign fragment must not FILL an empty solution either
                    # (same audit class as the append below: the re-ask for
                    # q16 can come back carrying q17's block).
                    foreign = _solution_fragment_foreign(incoming_text, qn, rec, chapter_records)
                    if foreign:
                        _log_blocked_retry_fragment(chapter_id, qn, foreign, incoming_text)
                        print(f"  [RETRY] blocked foreign solution fragment for q{qn} "
                              f"(empty solution): {foreign}")
                    else:
                        rec["solution_text"] = incoming_text
                        rec["_prov"]["solution_text"] = req_prov
                        fixed_this_round += 1
                elif qn in forced:
                    # Targeted prompt asks for the missing continuation, so
                    # preserve the established prose/bullets instead of
                    # replacing it with a regenerated full solution.
                    if incoming_text.startswith(sol_existing) and len(incoming_text) > len(sol_existing):
                        rec["solution_text"] = incoming_text
                        rec["_prov"]["solution_text"] = req_prov
                        fixed_this_round += 1
                    elif not _frag_mostly_present(incoming_text, sol_existing, 0.9):
                        # Wrong-owner guard (external-audit 2026-08-02:
                        # q16's truncated re-ask returned q17's solution and
                        # the old code APPENDED it -- 'not mostly present'
                        # was misread as new continuation). Only append when
                        # the fragment passes the deterministic foreign
                        # proofs; a genuine continuation is kept.
                        foreign = _solution_fragment_foreign(incoming_text, qn, rec, chapter_records)
                        if foreign:
                            _log_blocked_retry_fragment(chapter_id, qn, foreign, incoming_text)
                            print(f"  [RETRY] blocked foreign solution fragment for q{qn}: {foreign} "
                                  f"(existing solution untouched; fragment logged)")
                        else:
                            rec["solution_text"] = sol_existing.rstrip() + "\n" + incoming_text
                            rec["_prov"]["solution_text"] = req_prov
                            fixed_this_round += 1
            if "solution" in requested and incoming_tables:
                before_tables = rec.get("tables") or []
                merged_tables = _dedupe_tables(list(before_tables) + incoming_tables)
                if merged_tables != before_tables:
                    rec["tables"] = merged_tables
                    # Only count a table patch if it adds information, not if
                    # normalization merely changes ordering of an identical set.
                    if len(merged_tables) > len(before_tables):
                        fixed_this_round += 1
            if "options" in requested and fix.get("options"):
                rec["options"] = rec.get("options") or {}
                before = len(rec["options"])
                for k, v in fix["options"].items():
                    if v:  # don't let a null/empty value overwrite nothing-useful
                        rec["options"].setdefault(str(k).strip().upper(), v)
                if len(rec["options"]) > before:
                    fixed_this_round += 1

        print(f"  [RETRY] round {round_no}: filled {fixed_this_round} field(s)")
        total_fixed += fixed_this_round
        if fixed_this_round == 0:
            print("  [RETRY] no progress this round -- stopping (remaining gaps "
                  "will be logged, not re-tried, to avoid wasting quota)")
            break

    # whatever's STILL missing after all rounds -- log it, don't hide it.
    # Forced (truncated) items are re-judged LIVE here: a healed solution
    # must not stay logged as missing just because the sweep's verdict came
    # before this round's fix landed.
    live_forced = {qn for qn in forced
                   if qn in chapter_records
                   and looks_truncated_solution(
                       (chapter_records[qn].get("solution_text") or ""),
                       has_tables=bool(chapter_records[qn].get("tables")))}
    still_incomplete = find_incomplete_records(chapter_records, force_solution_qns=live_forced,
                                               printed_solution_qns=printed_solution_qns)
    # ALWAYS rewrite this chapter's ledger entries to the outcome of THIS run
    # (possibly zero -- full heal must also clear stale rows), never blindly append.
    _prune_still_incomplete(chapter_id)
    if still_incomplete:
        path = DATA_DIR / "still_incomplete_after_retry.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)  # same guard as save_state/_append_jsonl (fresh volume)
        with open(path, "a", encoding="utf-8") as f:
            for qn, missing in still_incomplete:
                f.write(json.dumps({"q_no": qn, "missing": missing, "chapter_id": chapter_id},
                                   ensure_ascii=False) + "\n")
        print(f"  [RETRY] {len(still_incomplete)} question(s) still incomplete after "
              f"{max_rounds} round(s) -- logged to still_incomplete_after_retry.jsonl")

    return total_fixed

def _prune_still_incomplete(chapter_id):
    """Drop this chapter's OLD entries from still_incomplete_after_retry.jsonl.
    The ledger was append-only: rows healed later (next retry round, recovery,
    or the healer) left stale 'missing solution' entries behind (confirmed in
    the 2026-07-27 external audit: 12 entries whose questions.jsonl rows were
    actually complete). Rewriting per chapter keeps the ledger truthful."""
    path = DATA_DIR / "still_incomplete_after_retry.jsonl"
    if not chapter_id or not path.exists():
        return
    kept = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            if json.loads(ln).get("chapter_id") == chapter_id:
                continue
        except json.JSONDecodeError:
            pass
        kept.append(ln)
    path.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")


RESCUE_MAX_CALLS_PER_CHAPTER = 12   # page-focused last-ditch re-ask budget


def rescue_incomplete_records(model, page_files, pdf_path, chapter_records, state,
                              stats, chapter_id, printed_solution_qns=None,
                              max_calls=RESCUE_MAX_CALLS_PER_CHAPTER):
    """Chapter-end rescue pass -- the page-focused LAST DITCH for records the
    targeted retry could not fill (run-5 audit: 9 records still missing
    answer/options after 2 whole-chapter retry rounds, e.g. ch2 q25/26,
    ch18 q13, ch19 q11/12, ch24 q12/13, ch27 q11, ch33 q9).

    Why this is different from targeted_retry: the retry re-sends the ENTIRE
    chapter every round -- a big, diffuse ask that stalls ("filled 0
    field(s)"). The rescue locates the EXACT pages where each missing q_no
    is printed (question stem and/or 'Solution to Question N:' header, via
    the zero-token text layer) and re-asks ONE focused call PER PAGE with
    just that page's image -- the same small-focused-ask principle that made
    per-page retries consistently recover where batch asks failed.

    Merges fill-only (never overwrites existing content); respects the daily
    quota; rewrites this chapter's still_incomplete_after_retry.jsonl to the
    post-rescue truth. Returns the number of fields filled."""
    incomplete = find_incomplete_records(chapter_records, force_solution_qns=(),
                                         printed_solution_qns=printed_solution_qns)
    if not incomplete:
        return 0
    qn_missing = {qn: set(missing) for qn, missing in incomplete}
    located = locate_missing_record_pages(pdf_path, page_files, qn_missing, chapter_records)
    unlocated = [qn for qn in qn_missing if qn not in located]
    for qn in unlocated:
        print(f"  [RESCUE] q{qn}: no printed question/solution page locatable in the "
              f"text layer -- left for --auto-recover / manual review")
        _append_jsonl(DATA_DIR / "integrity_flags.jsonl",
                      {"kind": "rescue_no_page_located", "q_no": qn,
                       "chapter_id": chapter_id,
                       "missing": sorted(qn_missing[qn])})

    def _count_fields(rec, missing):
        n = 0
        if "question" in missing and (rec.get("question_text") or "").strip():
            n += 1
        if "answer" in missing and rec.get("correct_option"):
            n += 1
        if "options" in missing and len(rec.get("options") or {}) >= 4:
            n += 1
        if "solution" in missing and (rec.get("solution_text") or "").strip():
            n += 1
        return n

    filled, calls = 0, 0
    pages_of = {}
    for qn, pages in located.items():
        for p in pages:
            pages_of.setdefault(p, []).append(qn)
    for page_no in sorted(pages_of):
        if calls >= max_calls:
            print(f"  [RESCUE] call budget ({max_calls}) exhausted -- remaining gaps go "
                  f"to --auto-recover / manual review")
            break
        qns_here = pages_of[page_no]
        pf = next((p for p in page_files
                   if int(p.stem.split("-")[-1]) == page_no), None)
        if pf is None:
            continue
        reset_daily_counter_if_needed(state)
        if state["calls_today"] >= MAX_CALLS_PER_DAY:
            print("  [RESCUE] daily Gemini call limit reached -- saving, exiting")
            save_state(state)
            sys.exit(0)
        prompt = build_targeted_retry_prompt(
            [(qn, sorted(qn_missing[qn])) for qn in qns_here], chapter_records)
        before_n = sum(_count_fields(chapter_records[qn], qn_missing[qn]) for qn in qns_here)
        try:
            raw = call_gemini_on_pages(model, [pf], context=RECOVERY_CONTEXT, prompt=prompt)
            state["calls_today"] += 1
            save_state(state)
            calls += 1
        except Exception as e:
            print(f"  [RESCUE] page {page_no} call failed ({e}) -- skipping page")
            continue
        items, _meta = extract_batch_meta(raw)
        for it in items:
            if isinstance(it, dict):
                it["_prov"] = "RESCUE"   # page-focused rescue provenance
        chapter_records, skipped = merge_question_records(chapter_records, items, stats,
                                                          fill_only=True)
        for it in skipped:
            _append_jsonl(DATA_DIR / "integrity_flags.jsonl",
                          {"kind": "rescue_unmatched_fragment", "chapter_id": chapter_id,
                           "page": page_no, "item": str(it)[:300]})
        after_n = sum(_count_fields(chapter_records[qn], qn_missing[qn]) for qn in qns_here)
        gained = after_n - before_n
        filled += gained
        print(f"  [RESCUE] page {page_no}: q{','.join(map(str, qns_here))} "
              f"-> {gained} field(s) filled")
    stats["rescue_calls"] = stats.get("rescue_calls", 0) + calls
    stats["rescue_filled"] = stats.get("rescue_filled", 0) + filled

    # truth the ledger post-rescue (targeted_retry wrote it before this pass)
    still = find_incomplete_records(chapter_records, force_solution_qns=(),
                                    printed_solution_qns=printed_solution_qns)
    _prune_still_incomplete(chapter_id)
    if still:
        path = DATA_DIR / "still_incomplete_after_retry.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for qn, missing in still:
                f.write(json.dumps({"q_no": qn, "missing": missing, "chapter_id": chapter_id},
                                   ensure_ascii=False) + "\n")
    return filled


def pdftotext_page(pdf_path, true_page):
    out = subprocess.run(["pdftotext", "-f", str(true_page), "-l", str(true_page),
                          "-layout", str(pdf_path), "-"], capture_output=True, text=True)
    return out.stdout or ""


def probe_batch_pages(pdf_path, window_pages):
    """V2 zero-token pass-activation probe. Reads the pdftotext layer of the
    batch's pages ONCE (one subprocess for the whole window) and decides which
    of the 3 passes are worth a Gemini call:
      key_table=True  -> pages print an answer-key table  -> run A-pass
      solutions=True  -> pages print 'Solution to Question N:' -> run S-pass
    Text layer is used ONLY for activation decisions, never as content. On any
    probe failure (scanned-only PDF, pdftotext missing) fall back to running
    ALL passes -- accuracy must never depend on the text layer existing."""
    try:
        lo, hi = min(window_pages), max(window_pages)
        out = subprocess.run(["pdftotext", "-f", str(lo), "-l", str(hi),
                              "-layout", str(pdf_path), "-"],
                             capture_output=True, text=True)
        text = out.stdout or ""
    except Exception:
        return {"key_table": True, "solutions": True, "probe_failed": True}
    if not text.strip():
        return {"key_table": True, "solutions": True, "probe_failed": True}
    # >=2 solution headers = a real solutions page, not a stray cross-reference
    return {"key_table": bool(KEY_TABLE_PROBE_RE.search(text)),
            "solutions": len(SOLUTION_PROBE_RE.findall(text)) >= 2,
            "probe_failed": False}


def build_section_windows(page_files, pdf_path):
    """Section-aware window planner (run-6 user ask).

    Reads the chapter's text layer ONCE (zero-token) to find where the
    Solutions section starts (first page with >=2 'Solution to Question N:'
    headers), then builds windows so:
      * the whole questions+answers stretch is sent in LARGE
        QUESTIONS_CHUNK_PAGES windows with 1-page overlap (a chapter's
        question section usually fits in 1-2 calls -- every question shares
        one context, so boundary splits and cross-window option drops
        disappear, and fewer calls = less 15-RPM pressure + less token
        waste: overlap drops from 2/6 (33%) to 1/10 (10%));
      * the Solutions section is sent in SMALLER SOLUTIONS_CHUNK_PAGES
        windows with 1-page overlap (long verbatim spans are what trigger
        finish_reason=4 recitation -- page 218 class).

    Pass ACTIVATION is NOT changed: each window still runs the probe-based
    Q/A/S decision (and the sticky extraction boundary), so a question page
    that the text layer mislabels can never be skipped -- section labels
    here only SIZE the windows and mark the carry hard-reset at the
    Solutions boundary. Returns a list of (page_numbers, section_label)
    tuples in reading order, or [] when the text layer cannot be read / no
    solutions section is detected -- the caller then falls back to the
    fixed 6-page window loop (unchanged safe path)."""
    pages = []
    for p in page_files:
        try:
            pages.append(int(p.stem.split("-")[-1]))
        except (ValueError, IndexError):
            pass
    if not pages:
        return []
    try:
        text_by_page = {p: (pdftotext_page(pdf_path, p) or "") for p in pages}
    except Exception:
        return []
    if not any(t.strip() for t in text_by_page.values()):
        return []
    solutions_start = None
    for p in sorted(pages):
        t = text_by_page[p]
        if t.strip() and len(SOLUTION_PROBE_RE.findall(t)) >= 2:
            solutions_start = p
            break
    if solutions_start is None:
        return []  # no solutions section detectable -> fixed-window fallback

    def chunks(pagenos, size, overlap):
        wins, i, step = [], 0, max(1, size - overlap)
        while i < len(pagenos):
            wins.append(pagenos[i:i + size])
            i += step
        return wins

    ordered = sorted(pages)
    windows = []
    q_pages = [p for p in ordered if p < solutions_start]
    if q_pages:
        for w in chunks(q_pages, QUESTIONS_CHUNK_PAGES, SECTION_OVERLAP_PAGES):
            windows.append((w, "Q"))
    s_pages = [p for p in ordered if p >= solutions_start]
    if s_pages:
        for w in chunks(s_pages, SOLUTIONS_CHUNK_PAGES, SECTION_OVERLAP_PAGES):
            windows.append((w, "S"))
    return windows


def solution_headers_on_page(pdf_path, file_page, chapter_records):
    """Locate every printed "Solution to Question N:" header on a page WITH
    its vertical position. Returns [(q_no, y_baseline)] in reading order
    (top of page first), where y is the header's baseline in PDF user space
    (origin at the page BOTTOM-left -- the SAME space image_positions_on_page
    reports, so no coordinate conversion is needed).

    Implementation: pypdf's text visitor (no extra subprocess) -- words are
    grouped into lines by baseline and matched against the header pattern;
    headers whose q_no is not in chapter_records are ignored (foreign chapter
    references). Returns [] when the text layer is missing/garbled or no
    header survives the chapter filter.

    Why positions matter: the old owner lookup returned q_nos WITHOUT
    positions, so a page whose text layer decoded only ONE of seven headers
    let the caller dump ALL seven figures onto that one solution. With
    positions, each figure can be matched to the header actually drawn above
    it -- and when a header cannot be located, the caller claims NOTHING for
    that figure instead of guessing."""
    try:
        page = PdfReader(pdf_path).pages[file_page - 1]
    except Exception:
        return []
    words = []

    def _visitor(text, _cm, tm, _font_dict, _font_size):
        t = (text or "").strip()
        if t:
            # tm[4]=x, tm[5]=baseline y from the PDF bottom-left origin
            words.append((round(float(tm[5]), 1), round(float(tm[4]), 1), t))

    try:
        page.extract_text(visitor_text=_visitor)
    except Exception:
        return []
    if not words:
        return []
    lines = {}
    for y, x, t in words:
        lines.setdefault(y, []).append((x, t))
    headers, seen = [], set()
    for y in sorted(lines, reverse=True):   # larger y == higher on the page
        line = " ".join(t for _, t in sorted(lines[y]))
        for m in re.finditer(r"Solution\s+to\s+Question\s+(\d{1,3})", line, re.IGNORECASE):
            qn = int(m.group(1))
            if qn in chapter_records and qn not in seen:
                seen.add(qn)
                headers.append((qn, y))
    return headers


def qns_printed_on_page(pdf_path, true_page, chapter_records):
    """Which of this chapter's q_nos are printed on this page, read from the
    text layer (0 tokens). Conservative: a hit counts only if the number
    appears at a question-stem position AND the q_no exists in the chapter.
    Returns sorted unique list -- caller auto-attaches ONLY on exactly one."""
    text = pdftotext_page(pdf_path, true_page)
    if not text.strip():
        return []
    found = set()
    for m in re.finditer(r"(?m)^\s*(?:Q(?:uestion)?\s*[.:]?\s*)?(\d{1,3})\s*[.)]", text):
        qn = int(m.group(1))
        if qn in chapter_records:
            found.add(qn)
    return sorted(found)


def chapter_printed_solution_qns(pdf_path, page_files, chapter_records):
    """Which of this chapter's q_nos have a printed 'Solution to Question N:'
    header somewhere in its pages (zero-token text layer). Per-question
    proof that the book prints an explanation for q_no -- used to bypass the
    60% solution-gate for exactly those q_nos (ch25 class: 7/12 solutions =
    58%, gate suppressed 5 REAL solutions whose headers were printed).

    One pdftotext subprocess per page (only run when a gate bypass might
    matter -- see process_pdf). Returns a set of q_nos."""
    found = set()
    for pf in page_files:
        try:
            page_no = int(pf.stem.split("-")[-1])
        except (ValueError, IndexError):
            continue
        text = pdftotext_page(pdf_path, page_no)
        if not text.strip():
            continue
        for m in re.finditer(r"Solution\s+to\s+Question\s+(\d{1,3})", text, re.IGNORECASE):
            qn = int(m.group(1))
            if qn in chapter_records:
                found.add(qn)
    return found


def locate_missing_record_pages(pdf_path, page_files, qn_missing, chapter_records):
    """One zero-token text-layer pass over the chapter's pages; returns
    {qn: [true_pdf_page, ...]} for every incomplete qn whose number is
    printed as a question stem and/or a 'Solution to Question N:' header.

    Powers the chapter-end rescue pass: instead of re-sending the whole
    chapter (which targeted retry already did and stalled on), the rescue
    re-asks ONLY the pages where the missing record is actually printed."""
    qns = set(qn_missing)
    pages = {qn: set() for qn in qns}
    if not qns:
        return {}
    header_re = re.compile(r"Solution\s+to\s+Question\s+(\d{1,3})", re.IGNORECASE)
    stem_re_cache = {}
    for pf in page_files:
        try:
            page_no = int(pf.stem.split("-")[-1])
        except (ValueError, IndexError):
            continue
        text = pdftotext_page(pdf_path, page_no)
        if not text.strip():
            continue
        for qn in qns:
            if qn not in stem_re_cache:
                stem_re_cache[qn] = re.compile(
                    r"(?m)^\s*(?:Q(?:uestion)?\s*[.:]?\s*)?%d\s*[.)]" % qn)
            if stem_re_cache[qn].search(text):
                pages[qn].add(page_no)
        for m in header_re.finditer(text):
            qn = int(m.group(1))
            if qn in pages:
                pages[qn].add(page_no)
    return {qn: sorted(ps) for qn, ps in pages.items() if ps}


def _transient_gemini_err(err_text):
    t = err_text.lower()
    return ("500" in t or "503" in t or "internal error" in t
            or "high demand" in t or "unavailable" in t)


def gemini_json_call_splitting(model, prompt, page_files, state, label="", direct_page_fallback=False):
    """Execute ONE logical ask (prompt + page images) so that a single bad
    or heavy call can never sink it. Run-4 PROOF of why this exists: a
    whole-chapter targeted retry went out as ONE 14-page call and failed
    as a unit -- 500 on ch9's set, recitation (finish_reason=4) on ch16's
    set (page 217 inside it poisoned the whole call) -- and both rounds
    then just SKIPPED, permanently losing 4+5 solutions and q11's options.
    Ladder per failure:
      1. transient (500/503/high-demand): 20s backoff, identical re-try once
      2. 429/quota burst: 65s backoff, re-try once; still limited -> the
         usual clean save+exit (same as the main batch loop)
      3. any remaining failure: split the page set in HALVES, re-ask each;
         a failing half descends to SINGLE pages
    Returns a list of parsed JSON arrays from all successful sub-calls
    ([] = everything failed; deterministic per-page failures like
    recitation-on-one-page simply cost that page, not the ask)."""
    def one_call(files):
        reset_daily_counter_if_needed(state)
        if state["calls_today"] >= MAX_CALLS_PER_DAY:
            print("Daily Gemini call limit reached. Saving progress, exiting.")
            save_state(state)
            sys.exit(0)
        result = call_gemini_on_pages(model, files, prompt=prompt)
        state["calls_today"] += 1
        save_state(state)
        return result

    malformed_json = object()

    def attempt(files):
        try:
            return one_call(files)
        except Exception as e:
            t = str(e)
            if "429" in t or "quota" in t.lower():
                print(f"  [429] rate limited{label} -- backing off 65s once")
                time.sleep(65)
                try:
                    return one_call(files)
                except Exception as e2:
                    t2 = str(e2)
                    if "429" in t2 or "quota" in t2.lower():
                        print(f"  [QUOTA] still limited after backoff -- saving, exiting: {e2}")
                        save_state(state)
                        sys.exit(0)
                    print(f"  [WARN] post-backoff call failed differently{label}: {e2}")
                    return None
            if "Invalid Gemini JSON" in t or "empty JSON response" in t:
                # Splitting a malformed structured response into every single
                # page does not repair the model's output format; it merely
                # burns quota (the 2026-07-28 V2 smoke test made 17 such
                # calls). Keep the chapter data, log the retry failure, and
                # leave the targeted fields in the review ledger.
                print(f"  [WARN] malformed Gemini JSON{label}; not splitting into per-page retries")
                return malformed_json
            if _transient_gemini_err(t) or "Empty Gemini response" in t or "Gemini API error" in t:
                print(f"  [WARN] transient/empty Gemini response{label} ({t[:120]}) -- one 20s-backoff retry")
                time.sleep(20)
                try:
                    return one_call(files)
                except Exception as e2:
                    print(f"  [WARN] backoff retry failed{label}: {str(e2)[:160]}")
                    return None
            print(f"  [WARN] call failed{label}: {t[:200]}")
            return None

    if not page_files:
        return []
    whole = attempt(page_files)
    if whole is malformed_json:
        return []
    if whole is not None:
        return [whole]
    # A retry request already combines every remaining q_no into one focused
    # call. If that genuine combined call fails, targeted retry falls back
    # directly to single pages (not another cascade of arbitrary halves).
    if direct_page_fallback:
        halves = [[page] for page in page_files]
    else:
        mid = (len(page_files) + 1) // 2
        halves = [page_files[:mid], page_files[mid:]]
    results = []
    for half in halves:
        if not half:
            continue
        r = attempt(half)
        if r is malformed_json:
            continue
        if r is not None:
            results.append(r)
            continue
        if len(half) == 1:
            print(f"  [WARN] page {Path(half[0]).name} failed even alone{label} -- excluded from this ask")
            continue
        for single in half:
            r2 = attempt([single])
            if r2 is malformed_json:
                continue
            if r2 is not None:
                results.append(r2)
            else:
                print(f"  [WARN] page {Path(single).name} failed even alone{label} "
                      f"-- excluded from this ask (chapter-end recovery paths remain)")
    return results


def _page_crops(pf, parts, overlap_frac=0.12):
    """Split a page image into `parts` horizontal bands with a small overlap
    (a solution clipped at the cut line appears WHOLE in >=1 crop; overlap
    re-extraction is merge-safe because every consumer is fill-only/deduped).
    Returns [(label, crop_path)]; crops live next to the source page."""
    im = Image.open(pf)
    w, h = im.size
    labels_map = {2: ("TOP half", "BOTTOM half"),
                  4: ("quarter 1 (top)", "quarter 2", "quarter 3", "quarter 4 (bottom)")}
    labels = labels_map.get(parts) or [f"band {i + 1}/{parts}" for i in range(parts)]
    step = h / parts
    ov = step * overlap_frac
    crops = []
    for i in range(parts):
        top = max(0, int(i * step - ov))
        bot = min(h, int((i + 1) * step + ov))
        out = Path(str(pf).rsplit(".", 1)[0] + f"_crop{parts}x{i + 1}.jpg")
        im.crop((0, top, w, bot)).save(out, "JPEG", quality=90)
        crops.append((labels[i], out))
    return crops


def drain_failed_pages(model, entries, page_dir, chapter_records, state, stats, pdf_path=None):
    """Second-chance pass for pages that failed even alone (recitation-prone
    content, run-2: PSY page 217 cost ch16 5 solutions). Called at chapter
    end with the recovery framing -- a differently-phrased, focused
    single-page ask often clears a recitation filter that fired on the
    bulk prompt. Fill-only merge: never overwrites first-pass content.
    Returns (chapter_records, new_orphans, healed_entries)."""
    new_orphans = []
    healed = []
    for entry in entries:
        reset_daily_counter_if_needed(state)
        if state["calls_today"] >= MAX_CALLS_PER_DAY:
            print("Daily Gemini call limit reached during failed-page drain. Saving, exiting.")
            save_state(state)
            sys.exit(0)
        # PATCH-ONLY RECOVERY (run-7 hardening #2): this page failed WHICH
        # pass? Recover only the fields that pass is allowed to produce.
        # Unknown pass -> unrestricted (None scope) but still provenance-tagged.
        scope = _RECOVERY_SCOPE.get(entry.get("pass"))
        drain_prov = f"DRAIN_{entry.get('pass') or 'S'}"
        pf = page_dir / entry["page_file"]
        if not pf.exists() and pdf_path is not None:
            # cross-day run: /tmp may be wiped -- re-render just this page.
            subprocess.run(["pdftoppm", "-jpeg", "-r", "150",
                            "-f", str(entry["true_page"]), "-l", str(entry["true_page"]),
                            str(pdf_path), str(page_dir / "page")])
        if not pf.exists():
            print(f"  [DRAIN] {entry['page_file']} could not be re-rendered -- dropping from queue")
            healed.append(entry)  # nothing more we can do; don't loop forever
            continue
        try:
            recitation_safe = "finish_reason=4" in str(entry.get("reason", ""))
            if recitation_safe:
                print(f"  [RECITATION_RECOVERY] {entry['page_file']}: paraphrase-mode recovery")
            raw = call_gemini_on_pages(
                model, [pf], context=RECOVERY_CONTEXT,
                prompt=(SCHEMA_PROMPT + RECITATION_RECOVERY_CONTEXT) if recitation_safe else None)
            state["calls_today"] += 1
            save_state(state)
        except Exception as e:
            # CROP LADDER (run-4 PROOF: PSY ch16 page 217 failed with
            # finish_reason=4 recitation at batch, alone, AND here, and its
            # 5 solutions were lost; ch9's whole-chapter retry died as one
            # 500-prone call). Recitation/safety filters fire on long
            # verbatim spans, and heavy calls hit 500s -- smaller crops mean
            # a smaller span and a lighter call per ask. Halves -> quarters,
            # fill-only merge per successful crop; the page is healed only
            # when EVERY crop at some level came back as a valid call.
            print(f"  [DRAIN] {entry['page_file']} failed on second chance ({e}) -- trying crop ladder")
            ladder_healed = False
            for parts_n in (2, 4):
                all_ok, any_items = True, False
                for crop_label, crop_pf in _page_crops(pf, parts_n):
                    try:
                        crop_ctx = (f"RECOVERY NOTE: you are seeing one crop ({crop_label}) of a "
                                    f"page that must be extracted in pieces. {RECOVERY_CONTEXT}")
                        raw = call_gemini_on_pages(model, [crop_pf], context=crop_ctx)
                        state["calls_today"] += 1
                        save_state(state)
                    except Exception as e2:
                        print(f"  [DRAIN] {entry['page_file']} {crop_label} failed too ({e2})")
                        all_ok = False
                        continue
                    items2, _ = extract_batch_meta(raw)
                    items2 = [_apply_recovery_scope(dict(it), scope, drain_prov)
                              for it in items2 if isinstance(it, dict)]
                    if items2:
                        any_items = True
                    chapter_records, skipped2 = merge_question_records(
                        chapter_records, items2, stats, fill_only=True)
                    for it in skipped2:
                        new_orphans.append({"chapter_id": entry.get("chapter_id"), "batch_start": -1, "pass": entry.get("pass"),
                                            "pdf_pages": [int(entry["true_page"])], "new_pages": [],
                                            "carry_q_no": None, "item": it})
                    print(f"  [DRAIN] {entry['page_file']} {crop_label}: {len(items2)} item(s)")
                if all_ok:
                    print(f"  [DRAIN] {entry['page_file']} recovered via {parts_n}x crop ladder "
                          f"({any_items and 'items found' or 'page genuinely had no items'})")
                    healed.append(entry)
                    ladder_healed = True
                    break
            if not ladder_healed and recitation_safe:
                print(f"  [OCR_FALLBACK] {entry['page_file']}: attempting OCR+restructure", flush=True)
                try:
                    raw_text = ocr_fallback_text(pf)
                    if raw_text.strip():
                        header_recovered = _recover_ocr_solution_headers(raw_text, chapter_records)
                        # Printed solution headers are the reliable path for this
                        # book. Only ask Gemini to structure OCR when no header
                        # gave us a deterministic owner.
                        if header_recovered:
                            healed.append(entry)
                            ladder_healed = True
                            print(f"  [OCR_FALLBACK] {entry['page_file']}: header recovery completed")
                            continue
                        raw = call_gemini_text_only(model, RECITATION_RECOVERY_CONTEXT +
                            "\nRaw OCR text follows. Structure it into the normal JSON array; "
                            "correct obvious OCR errors but do not invent content.\n\nOCR TEXT:\n" + raw_text)
                        state["calls_today"] += 1; save_state(state)
                        items2, _ = extract_batch_meta(raw)
                        items2 = [normalize_ocr_fallback_item(item) for item in items2
                                  if isinstance(item, dict)]
                        # OCR recovery is field-scoped by the failed pass
                        # (run-7 hardening #2/#4): an OCR_S fragment carries
                        # solution-only content; its stray question/option
                        # text is dropped BEFORE anything can merge.
                        items2 = [_apply_recovery_scope(
                            it, scope, f"OCR_{entry.get('pass') or 'S'}")
                            for it in items2]
                        for item in items2:
                            owner_qn = _ocr_content_owner(item, chapter_records)
                            owner = chapter_records.get(owner_qn) if owner_qn is not None else None
                            continuation = (item.get("solution_text") or "").strip()
                            if owner and continuation and (owner.get("solution_text") or "").strip():
                                tail = _novel_solution_tail(owner["solution_text"], continuation)
                                if tail:
                                    owner["solution_text"] = owner["solution_text"].rstrip() + "\n" + tail
                                    print(f"  [OCR_FALLBACK] spliced novel continuation to q{owner_qn}")
                                else:
                                    print(f"  [OCR_FALLBACK] duplicate solution content ignored for q{owner_qn}")
                                item["solution_text"] = None
                                # A null q_no fragment becomes mergeable once its
                                # option-content evidence identifies an owner.
                                item["q_no"] = owner_qn
                        print(f"  [OCR_FALLBACK] {entry['page_file']}: normalized {len(items2)} OCR item(s)")
                        chapter_records, skipped2 = merge_question_records(chapter_records, items2, stats, fill_only=True)
                        new_orphans.extend({"chapter_id": entry.get("chapter_id"), "pass": entry.get("pass"), "item": it,
                                            "pdf_pages": [entry["true_page"]]} for it in skipped2)
                        healed.append(entry)
                        ladder_healed = True
                        print(f"  [OCR_FALLBACK] {entry['page_file']}: recovered {len(items2)} item(s)")
                except Exception as ocr_err:
                    print(f"  [OCR_FALLBACK] {entry['page_file']} failed: {ocr_err}")
            if not ladder_healed:
                print(f"  [DRAIN] {entry['page_file']} resisted crop+OCR recovery -- kept in failed_pages queue")
            continue
        print(f"  [DRAIN] {entry['page_file']} recovered on second chance")
        items, _meta = extract_batch_meta(raw)
        items = [_apply_recovery_scope(dict(it), scope, drain_prov)
                 for it in items if isinstance(it, dict)]
        chapter_records, skipped = merge_question_records(chapter_records, items, stats, fill_only=True)
        for it in skipped:
            new_orphans.append({"chapter_id": entry.get("chapter_id"), "batch_start": -1, "pass": entry.get("pass"),
                                "pdf_pages": [int(entry["true_page"])], "new_pages": [],
                                "carry_q_no": None, "item": it})
        healed.append(entry)
    return chapter_records, new_orphans, healed

# ============================================================
# FEATURE 2 — carry-forward context (Gemini's API is stateless:
# continuity must be injected manually into every new request)
# ============================================================

def extract_batch_meta(items):
    """Peel the {"_batch_meta": {...}} and {"_figure_map": [...]} control
    objects out of Gemini's array. Returns (question_items, meta_dict).
    Meta of a failed/absent call = {}. The figure map (q_no+slot per figure
    in reading order, run-6 user ask) rides along under meta["figure_map"]."""
    questions, meta = [], {}
    for it in items:
        if isinstance(it, dict) and "_batch_meta" in it:
            m = it.get("_batch_meta")
            if isinstance(m, dict):
                # keep a figure_map that arrived BEFORE the batch-meta object
                # (response order is not guaranteed)
                if meta.get("figure_map"):
                    m = {**m, "figure_map": meta["figure_map"]}
                meta = m          # last one wins (single-page retries)
            continue
        if isinstance(it, dict) and "_figure_map" in it:
            fm = it.get("_figure_map")
            if isinstance(fm, list) and not meta.get("figure_map"):
                meta["figure_map"] = fm   # first non-empty map wins
            continue
        questions.append(it)
    return questions, meta

def compute_carry(batch_meta, items, chapter_records, ending_page):
    """Decide whether a batch ended mid-question and build the payload carried
    into the NEXT request. Primary signal: Gemini's own _batch_meta (it can
    see the page bottom). Fallback when NO usable meta: detect the pass shape
    from the items themselves --
      * S-pass items carry solution_text (never question_text): a non-empty
        solution on the window's highest q_no that LOOKS TRUNCATED proves the
        page ended mid-solution -> carry that q_no as a "solution" cut. This
        was the run-8 root cause: the old fallback required question_text,
        which S-pass records never have, so carry-in was ALWAYS "-" and the
        unnumbered continuation on the next page came back q_no=null.
      * Q-pass items carry question_text: keep the battle-tested fallback
        (highest q_no with a stem but no solution yet -> carry as "solution").
    Stores: last_open_question, last_question_text, partial_solution,
    partial_options, ending_page."""
    have_meta = bool(batch_meta)
    meta_says_open = bool(batch_meta.get("ends_mid_content")) if have_meta else False
    last_qn = None
    if have_meta:
        try:
            last_qn = int(batch_meta.get("last_q_no"))
        except (TypeError, ValueError):
            last_qn = None
    cut_part = batch_meta.get("cut_part") or "unknown" if have_meta else "solution"

    if meta_says_open and last_qn is None:
        # model knows it's cut but can't see the number -- still carry the tail
        return {"last_open_question": None, "last_question_text": None,
                "partial_solution": batch_meta.get("tail_text") or None,
                "partial_options": None, "ending_page": ending_page,
                "cut_part": cut_part}
    if not meta_says_open:
        if have_meta:
            return None              # model says the page ended cleanly
        batch_qns = []
        s_shaped = False             # items look like S-pass output
        for it in items:
            try:
                batch_qns.append(int(it.get("q_no")))
            except (TypeError, ValueError):
                pass
            if (it.get("solution_text") or "").strip() \
                    and not (it.get("question_text") or "").strip():
                s_shaped = True
        if not batch_qns:
            return None
        candidate = max(batch_qns)
        rec = chapter_records.get(candidate, {})
        if s_shaped:
            # S-pass fallback (run-8): a truncated solution proves the page
            # ended mid-solution -> carry it so the next window's unnumbered
            # continuation resolves to this q_no instead of q_no=null.
            sol = (rec.get("solution_text") or "").strip()
            if sol and looks_truncated_solution(
                    sol, has_tables=bool(rec.get("tables"))):
                last_qn, cut_part = candidate, "solution"
            else:
                return None
        else:
            # Q-pass fallback (battle-tested): highest q_no with a stem but
            # no solution yet -> carry as "solution" (if it spans the next
            # window, the model continues it under the same q_no).
            if rec.get("question_text") and not rec.get("solution_text"):
                last_qn, cut_part = candidate, "solution"
            else:
                return None

    rec = chapter_records.get(last_qn, {})
    return {"last_open_question": last_qn,
            "last_question_text": rec.get("question_text"),
            "partial_solution": rec.get("solution_text") or batch_meta.get("tail_text") or None,
            "partial_options": rec.get("options"),
            "ending_page": ending_page,
            "cut_part": cut_part}

def build_carry_context(carry, overlap_pages, new_pages=None):
    """The actual text prepended to the next request.

    carry: the previous window's open item (its q_no + partial content).
    overlap_pages: PDF pages re-sent from the previous window (continuity).
    new_pages: the genuinely NEW pages of this window (run-8: made explicit
    so Gemini can resolve an unnumbered continuation's owner from the
    preceding overlap page instead of defaulting to q_no=null -- the orphan
    source the audit found)."""
    lines = []
    if carry:
        qn = carry["last_open_question"]
        lines += [
            "CONTEXT FROM PREVIOUS BATCH (continuity context only -- do NOT",
            "output any of this text as a new item):",
            "Previous batch ended with an incomplete question.",
            f"Question Number: {qn if qn is not None else 'unknown'}",
            f"Question: {(carry.get('last_question_text') or '')[:600]}",
            f"Options seen so far: {json.dumps(carry.get('partial_options'), ensure_ascii=False)[:400]}",
            f"Partial Solution: {(carry.get('partial_solution') or '')[:600]}",
            f"(cut part: {carry.get('cut_part')}; ended at PDF page {carry.get('ending_page')})",
            "If the first content in this batch belongs to this question,",
            "CONTINUE it under the SAME q_no instead of creating a new question.",
        ]
    if overlap_pages:
        lines.append(
            "OVERLAP / CONTEXT PAGES (supplied ONLY to establish continuity "
            "and ownership; do NOT re-output their content as new items): "
            f"PDF page(s) {', '.join(map(str, overlap_pages))}."
        )
        if new_pages:
            lines.append(
                "NEW PAGES TO EXTRACT (the pages whose content this pass must "
                f"return): PDF page(s) {', '.join(map(str, new_pages))}."
            )
        lines.append(
            "OWNERSHIP RULES for unnumbered continuations:\n"
            "- If a new page begins with an unnumbered continuation and the "
            "preceding OVERLAP page proves it belongs to Question N (e.g. the "
            "'Solution to Question N:' header or question stem N is visible "
            "at the bottom of the overlap page), return that continuation "
            "with q_no=N.\n"
            "- Keep assigning it to N until an explicit new question/solution "
            "heading establishes another owner.\n"
            "- Do NOT return q_no=null merely because the number is not "
            "repeated on the new page when ownership is clearly established "
            "by the overlap page.\n"
            "- NEVER invent a q_no when ownership is uncertain. If ownership "
            "genuinely cannot be established, return q_no=null as an explicit "
            "unassigned fragment for later recovery -- never attach it to a "
            "different question."
        )
    return "\n".join(lines)

ANSWER_KEY_ROW_RE = re.compile(r"\|\s*(\d{1,3})\s*\|\s*([A-Da-d])\s*\|")
SOLUTION_TO_Q_RE = re.compile(r"Solution to Question\s+(\d{1,3})", re.IGNORECASE)
# Dump-tail detector (stricter): title-case header WITH colon, i.e. the real
# printed "Solution to Question 2:" section header, not prose mentions.
SOLUTION_DUMP_HDR_RE = re.compile(r"Solution to Question\s+(\d{1,3})\s*:")

# --- stale carry-context guards (clarified RCA: the header-alone-at-page-end
# split is NORMAL in this book for questions AND solutions, and the overlap
# window resolves it -- do NOT touch that path. The actual bug is a carry
# context whose OWN split never resolves staying alive long enough to meet
# the same number again in the Solutions section ("Solution to Question 4:")
# and cross-merge question prose with solution prose).
CARRY_EXPIRY_BATCHES = 3      # unresolved carry dies after this many batches

SOLUTION_STYLE_STEM_RE = re.compile(
    r"^\s*(?:option\s+[a-d]\s*[:.)\-]|ans(?:wer)?\s*[:.)\-]|correct\s+answer\s+is\b|"
    r"the\s+(?:correct\s+)?answer\s+is\b|solution\s*[:.)\-]|explanation\s*[:.)\-]|"
    r"answer\s*[:.)\-]|solution\s+to\s+question\s+\d+)", re.IGNORECASE)

SECTION_HEADING_RE = re.compile(
    r"^\s*(?:chapter\s+\d{1,3}\s*[:.\-–]?\s*)?"
    r"(detailed\s+explanations?|answer\s*keys?|answers?\s+(?:and|&)\s+explanations?|"
    r"explanations?|answers?)\s*[.:\-–]?\s*$", re.IGNORECASE)

# --- run-4 audit RCA guards (2026-07-26 full-output audit; see
# ROOT_CAUSE_ANALYSIS.md "Run-4 audit" section). Deterministic, zero-token:
MAX_QUESTION_IMAGES = 3       # >3 question-side figures on ONE question is almost
                              # certainly wrong-owner attribution (PSY-022-003
                              # collected SEVEN via repeated model-confirmed passes).
MAX_SOLUTION_IMAGES = 2       # a solution block cites at most a figure or two;
                              # >2 on ONE solution from the deterministic path means
                              # under-detected headers dumped neighbours' figures onto
                              # it (user report: a 7-figure solutions page collapsed
                              # into just 2 solutions -- the old single-owner
                              # shortcut attached EVERY page image to the one header
                              # the text layer happened to decode).
MIN_IMAGE_BYTES = 1500        # <1.5 KB webp is virtually always an empty/broken crop
                              # (PSY-003-014_Q_01 was 414 bytes of nothing and shipped).
STEM_COHERENCE_MARGIN = 0.15  # stem-conflict resolver: stem<->payload coherence scores
                              # must differ by at least this to decide automatically;
                              # below it both variants are logged for review (no silent picks).
DANGLING_END_RE = re.compile(r"(:|\u2014|\u2013|\u2022)\s*$")   # ends ':' / em/en-dash / bullet
OPTION_LINE_START_RE = re.compile(r"^\s*Option\s+([A-D])\b\s*[:.)]\s*", re.IGNORECASE)

TERMINAL_PUNCT = ".!?)\"'\u201d\u00bb"


PIPE_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|.*\|\s*$")


def _table_body_rows(table):
    """Normalized non-header body rows for completeness/prefix comparison."""
    lines = []
    for line in str((table or {}).get("markdown") or "").splitlines():
        norm = re.sub(r"\s+", "", line).lower()
        if not norm or re.fullmatch(r"\|?-{3,}(?:\|-{3,})+\|?", norm):
            continue
        lines.append(norm)
    return lines


def _dedupe_tables(tables):
    """Keep one best table per overlap capture.

    Exact whitespace-insensitive matches are duplicates.  A shorter table
    whose normalized rows are a strict prefix of another is the page-break
    capture of the longer table, so retain the longer version.
    """
    candidates = [t for t in (tables or []) if isinstance(t, dict)]
    # Evaluate full captures first so a partial capture can never win by order.
    candidates.sort(key=lambda t: (-len(_table_body_rows(t)), -len(str(t.get("markdown") or "")),
                                  re.sub(r"\s+", "", str(t.get("markdown") or "").lower())))
    kept = []
    for table in candidates:
        key = re.sub(r"\s+", "", str(table.get("markdown") or "").lower())
        rows = _table_body_rows(table)
        duplicate = False
        for winner in kept:
            winner_key = re.sub(r"\s+", "", str(winner.get("markdown") or "").lower())
            winner_rows = _table_body_rows(winner)
            same = bool(key) and key == winner_key
            strict_prefix = bool(rows) and len(rows) < len(winner_rows) and winner_rows[:len(rows)] == rows
            if same or strict_prefix:
                duplicate = True
                break
        if not duplicate:
            kept.append(table)
    return kept


def _extract_inline_pipe_tables(text):
    """Remove 3+ consecutive markdown pipe-table lines from prose and return
    them as structured tables.  This is a schema firewall for retry output."""
    lines = (text or "").splitlines()
    prose, extracted, i = [], [], 0
    while i < len(lines):
        if not PIPE_TABLE_LINE_RE.match(lines[i]):
            prose.append(lines[i])
            i += 1
            continue
        j = i
        while j < len(lines) and PIPE_TABLE_LINE_RE.match(lines[j]):
            j += 1
        block = lines[i:j]
        if len(block) >= 3:
            extracted.append({"type": "recovered inline table", "markdown": "\n".join(block)})
        else:
            prose.extend(block)
        i = j
    return "\n".join(prose).strip(), extracted


def _normalize_solution_payload(text, tables, qn=None):
    clean, recovered = _extract_inline_pipe_tables(text)
    all_tables = _dedupe_tables(list(tables or []) + recovered)
    if recovered:
        print(f"  [SCHEMA_VIOLATION] q{qn if qn is not None else '?'}: moved "
              f"{len(recovered)} inline pipe-table block(s) from solution_text to tables")
    # This warning is deliberately after remediation: any surviving 3-line
    # pipe table is a future parser case, not silently shipped prose.
    if any(len([ln for ln in clean.splitlines()[i:i + 3] if PIPE_TABLE_LINE_RE.match(ln)]) == 3
           for i in range(max(0, len(clean.splitlines()) - 2))):
        print(f"  [SCHEMA_VIOLATION] q{qn if qn is not None else '?'}: pipe-table syntax remains in solution_text")
    return clean, all_tables


def looks_truncated_solution(text, has_tables=False, has_images=False):
    """REAL truncation patterns only (replaces the weak 'no terminal punct'
    heuristic that produced ~53 false positives against this book's
    bullet-list endings). Detects:
      - dangling connector endings  ('...criteria:', '...given below --')
      - raw trailing space after a word (stream cut mid-flow: '...• During ')
      - suspiciously short AND bare (no table/figure carrying the rest)
    """
    t = (text or "")
    s = t.rstrip()
    if not s:
        return False
    # This sweep is specifically a prose-continuation detector. A populated
    # table is the continuation for a lead-in ("stages are:"), so never send
    # that record to a truncation retry based on text ending alone.
    if has_tables:
        return False
    # Do not infer truncation from absent terminal punctuation, a trailing
    # OCR space, or a short explanation. Source pages frequently omit a final
    # period, and those heuristics created false retries (including q13).
    # Only an explicit dangling lead-in is deterministic enough to re-ask.
    return bool(DANGLING_END_RE.search(s))


def _stem_payload_coherence(stem, rec):
    """Share of stem word-tokens present in the record's OWN options+solution.
    A stem is explained by its own solution, so the right stem for a record
    coheres with the record's payload (run-4: PSY-012-001 kept PSY-012-013's
    chart stem while its solution described a mania vignette -- coherence
    0 vs the real stem)."""
    toks = [t for t in re.findall(r"\w+", (stem or "").lower()) if len(t) > 2]
    payload = " ".join(filter(None, [
        rec.get("solution_text") or "",
        " ".join(str(v) for v in (rec.get("options") or {}).values()),
    ]))
    ptoks = set(re.findall(r"\w+", payload.lower()))
    if not toks or not ptoks:
        return 0.0
    return sum(1 for t in toks if t in ptoks) / len(toks)


def _foreign_option_line(frag, rec):
    """Wrong-owner guard for solution fragments that BEGIN with an
    'Option X:' explanation (run-4: PSY-009-007 got PSY-009-006's
    'Option C: Catharsis...' line glued on top). A legitimate
    'Option X:' continuation names the OWNER's option X content; a
    foreign one does not."""
    m = OPTION_LINE_START_RE.match(frag or "")
    if not m:
        return False
    letter = m.group(1).upper()
    opt_text = (rec.get("options") or {}).get(letter)
    if opt_text is None:
        return True   # owner has no such option -> provably foreign
    otoks = [t for t in re.findall(r"\w+", str(opt_text).lower()) if len(t) > 2]
    if not otoks:
        return False
    head = " ".join(re.findall(r"\w+", (frag or "").lower())[:25])
    return sum(1 for t in otoks[:6] if t in head) == 0


def _solution_fragment_foreign(frag, qn, rec, chapter_records):
    """Deterministic 'this retry fragment does NOT belong to q{qn}' proofs
    for solution text returned by targeted_retry. External-audit class
    (2026-08-02): a truncated-solution re-ask for q16 came back carrying
    q17's text and the old code APPENDED it because 'not mostly present'
    was treated as new continuation -- blending two questions' solutions.
    Returns a short reason string, or None when no proof fires.

    Proofs (all zero-token, cross-record where possible):
      1. the fragment begins with an 'Option X:' explanation of an option
         this record does not own (reuses the orphan wrong-owner guard);
      2. the fragment carries an embedded 'Solution to Question N:' header
         naming a DIFFERENT question (self-labeled foreign block);
      3. the fragment's first content line exists verbatim in another
         record of this chapter (sibling-donor proof -- the same evidence
         the integrity sweep uses before it trims a head).

    A fragment that passes all three is kept: a genuine continuation after
    a cut point shares no tokens with the existing text by construction,
    so low overlap alone is deliberately NOT foreign evidence."""
    s = (frag or "").strip()
    if not s:
        return None
    if _foreign_option_line(s, rec):
        return "fragment begins with an 'Option' line the owner cannot own"
    for m in re.finditer(r"Solution\s+to\s+Question\s+(\d{1,3})", s, re.IGNORECASE):
        n = int(m.group(1))
        if n != qn:
            return f"fragment carries 'Solution to Question {n}:' (not q{qn})"
    first_line = s.splitlines()[0].strip()
    if first_line and len(first_line) >= 20:
        for other_qn, other in chapter_records.items():
            if other_qn == qn:
                continue
            if first_line in (other.get("solution_text") or ""):
                return f"first line exists verbatim in q{other_qn}'s solution"
    return None


def looks_like_solution_style_stem(text):
    """True when a would-be question_text is really solution prose
    ("Option A: ...", "Ans. is B", "Solution to Question 4: ...").
    Anchored at the start so real stems mentioning options mid-text are safe."""
    return bool(text and SOLUTION_STYLE_STEM_RE.search(str(text)))


def detect_section_boundary(items):
    """Return a short label on the FIRST batch whose extracted content shows
    the questions -> answers/solutions section boundary, else None. Signals,
    all from Gemini's own extraction (body-page pdftotext is garbled for this
    book, so deterministic page-text scanning is NOT an option):
      - a standalone heading line like "Detailed Explanations" / "Answer Key"
      - a self-labeled "Solution to Question N:" solution fragment
      - an Answer Key table (type says answer + 'Correct Option' markdown)"""
    for it in items:
        for t in it.get("tables") or []:
            md = t.get("markdown") or ""
            if "answer" in str(t.get("type", "")).lower() and "Correct Option" in md:
                return "Answer Key table"
        sol = it.get("solution_text") or ""
        m = SOLUTION_TO_Q_RE.search(sol)
        if m:
            return f"'Solution to Question {m.group(1)}' fragment"
        for field in (it.get("question_text"), sol):
            for line in str(field or "").splitlines():
                line = line.strip()
                if line and len(line) <= 60 and SECTION_HEADING_RE.match(line):
                    return f"'{line}' heading"
    return None


def _carry_resolved(rec, cut_part):
    """Has the piece this carry was waiting for actually arrived?"""
    cut = (cut_part or "solution").lower()
    if cut == "options":
        return len(rec.get("options") or {}) >= 4
    if cut == "question":
        return bool((rec.get("question_text") or "").strip()) and \
            len(rec.get("options") or {}) >= 4
    return bool((rec.get("solution_text") or "").strip())   # "solution"/"unknown"


def enforce_carry_expiry(carry, batch_seq, tracker, banned, chapter_records, chapter_id):
    """Kill stale carry contexts before they can cross-merge into the
    Solutions section. Rules, in order:
      - q_no already banned this chapter -> drop (the no-meta fallback in
        compute_carry would otherwise RESPAWN the same stale carry every
        batch that number stays the max text-no-solution candidate).
      - carried piece now filled -> resolved, drop quietly, un-track.
      - same q_no unresolved for CARRY_EXPIRY_BATCHES batches -> drop,
        mark the question still-incomplete IMMEDIATELY (turant), and ban
        the number for the rest of the chapter.
    Numberless tails (last_open_question=None) live one batch by
    construction and pass through untouched."""
    if carry is None:
        return None
    qn = carry.get("last_open_question")
    if qn is None:
        return carry
    if qn in banned:
        return None
    rec = chapter_records.get(qn) or {}
    if _carry_resolved(rec, carry.get("cut_part")):
        tracker.pop(qn, None)
        return None
    opened = tracker.setdefault(qn, batch_seq)
    if batch_seq - opened >= CARRY_EXPIRY_BATCHES:
        tracker.pop(qn, None)
        banned.add(qn)
        cut = carry.get("cut_part") or "solution"
        print(f"  [CARRY] q{qn} carry EXPIRED unresolved after "
              f"{CARRY_EXPIRY_BATCHES} batches (cut part: {cut}) -- dropped + "
              f"number banned this chapter + marked still-incomplete (a stale "
              f"context must never meet its number again in the Solutions "
              f"section)")
        _append_jsonl(DATA_DIR / "still_incomplete_after_retry.jsonl",
                      {"q_no": qn, "missing": [cut], "chapter_id": chapter_id,
                       "reason": "carry-context expired unresolved"})
        return None
    return carry


def _frag_mostly_present(frag, existing, threshold=0.85):
    """Token-overlap duplicate guard: substring checks miss near-dupes when
    punctuation differs ('...target.' vs '...target for...'), which caused a
    double-append in testing. True when >=threshold of frag's tokens already
    appear in existing's token set."""
    f = re.findall(r"\w+", (frag or "").lower())
    e = set(re.findall(r"\w+", (existing or "").lower()))
    if not f or not e:
        return False
    return sum(1 for t in f if t in e) / len(f) >= threshold


def recover_orphans(orphans, chapter_records, subject, chapter_no, stats):
    """FEATURE 3 -- second-pass owner matching for q_no=null fragments.
    Confidence rules, in order:
      0. ANSWER-KEY TABLE orphan (run-2 finding: pages 182/194/235/241 all
         produced {'q_no': None, 'tables':[Answer Key]} orphans and the old
         rules had NO handler for them -- a latent whole-chapter answer-loss
         bug). Parse the markdown rows deterministically and fill missing
         correct_options (fill-only, never overwrite).
      1. owner printed INSIDE the fragment text ("Solution to Question 3:")
         -- run-2: pages 85/159/273 all carried self-labeling fragments that
         the old matcher never parsed (0/13 orphans recovered that run).
      2. the carry-forward owner captured when the fragment arrived.
      3. the highest-numbered question from the SAME batch window is missing
         exactly the field the orphan provides (solution/options/question).
         NEW: also attaches continuation fragments to PARTIAL owners
         (half-solutions) via append, when the fragment leads NEW text.
    Recovered content is APPENDED / fill-only (existing text is never
    overwritten). Whatever remains unmatched is returned for orphans.jsonl
    -- never silently discarded."""
    remaining = []
    for orph in orphans:
        item = orph["item"]
        page = (orph.get("new_pages") or orph.get("pdf_pages") or ["?"])[0]

        # ---- rule 0: answer-key table -> deterministic correct_option fills
        # Upgraded (run-4 audit): a key whose rows ALL match existing answers
        # is CONSUMED as "verified" instead of lingering in orphans.jsonl as
        # noise (5 such orphans in the PSY run), and any DISAGREEING row is
        # written to data/integrity_flags.jsonl -- a free wrong-answer alarm.
        key_rows = []
        for t in item.get("tables") or []:
            if "answer" not in str(t.get("type", "")).lower() and "Correct Option" not in (t.get("markdown") or ""):
                continue
            for qn_s, letter in ANSWER_KEY_ROW_RE.findall(t.get("markdown") or ""):
                key_rows.append((int(qn_s), letter.upper()))
        if key_rows:
            filled_by_key, disagreed, unknown_qn = 0, [], []
            for kqn, letter in key_rows:
                rec = chapter_records.get(kqn)
                if rec is None:
                    unknown_qn.append(kqn)
                elif rec.get("correct_option"):
                    if str(rec["correct_option"]).strip().upper() != letter:
                        disagreed.append({"q_no": kqn, "record": rec["correct_option"], "key": letter})
                else:
                    rec["correct_option"] = letter
                    filled_by_key += 1
            if disagreed:
                _append_jsonl(DATA_DIR / "integrity_flags.jsonl",
                              {"kind": "answer_key_disagrees", "page": page,
                               "chapter_id": stats.get("chapter_id"), "rows": disagreed})
                print(f"  [WARN] [ORPHAN] answer-key table DISAGREES with extracted answers on "
                      f"{len(disagreed)} row(s) -- logged to integrity_flags.jsonl")
            if filled_by_key or not unknown_qn:
                stats["orphans_recovered"] += 1
                print(f"  [ORPHAN] Recovered orphan: page={page} answer-key table -> "
                      f"{filled_by_key} answer(s) filled, {len(key_rows) - filled_by_key - len(unknown_qn)} "
                      f"row(s) verified against existing answers -- consumed")
                continue
            # ENTIRELY foreign key (every row references another chapter's
            # q_nos): STOP here. Falling through to rules 1-4 would let this
            # key's table glue onto a local record via carry/last-qn merge.
            print(f"  [ORPHAN] answer-key table references q_nos outside this chapter "
                  f"({unknown_qn}) -- kept for review, NOT merged anywhere")
            remaining.append({**orph, "blocked_reason": "foreign answer key (all rows "
                              "reference q_nos not in this chapter)"})
            continue

        # ---- rule 0b: duplicate scrap consume (run-4 audit: both content
        # orphans in PSY-006 were re-extractions of records that ALREADY
        # exist complete -- a stem identical to some record's stem, or bare
        # options identical to that record's options). Consume them instead
        # of re-merging (idempotent) or persisting as noise.
        if not item.get("tables") and not item.get("solution_text"):
            itxt = (item.get("question_text") or "").strip()
            if itxt:
                dup = any((r2.get("question_text") or "").strip()
                          and _frag_mostly_present(itxt, r2["question_text"], 0.9)
                          and _frag_mostly_present(r2["question_text"], itxt, 0.9)
                          for r2 in chapter_records.values())
                if dup:
                    stats["orphans_recovered"] += 1
                    print(f"  [ORPHAN] Consumed orphan: page={page} stem already present "
                          f"verbatim in this chapter (duplicate re-extraction scrap)")
                    continue
            elif item.get("options"):
                cand = chapter_records.get(orph.get("last_qn_in_batch"))
                c_opts = {str(k).strip().upper(): str(v) for k, v in (cand or {}).get("options", {}).items()}
                if cand and all(str(k).strip().upper() in c_opts
                                and _frag_mostly_present(str(v), c_opts[str(k).strip().upper()], 0.9)
                                for k, v in item["options"].items()):
                    stats["orphans_recovered"] += 1
                    print(f"  [ORPHAN] Consumed orphan: page={page} options fragment already "
                          f"present on q{orph.get('last_qn_in_batch')} (duplicate scrap)")
                    continue

        owner, reason = None, None
        # ---- rule 1: owner self-labeled inside the fragment text
        m = SOLUTION_TO_Q_RE.search(item.get("solution_text") or "")
        if m:
            hint_qn = int(m.group(1))
            if hint_qn in chapter_records:
                owner, reason = hint_qn, "self-labeled 'Solution to Question N' fragment"
        # ---- rule 2: carry-forward owner
        carry_qn = orph.get("carry_q_no")
        if owner is None and carry_qn is not None and carry_qn in chapter_records:
            owner = carry_qn
            reason = f"{orph.get('cut_part') or 'content'} continuation (carry-forward)"
        # ---- rule 3: highest q_no of the same batch window missing that field
        if owner is None:
            last_qn = orph.get("last_qn_in_batch")
            rec = chapter_records.get(last_qn) if last_qn is not None else None
            if rec:
                frag = (item.get("solution_text") or "").strip()
                existing = (rec.get("solution_text") or "").strip()
                if item.get("solution_text") and not existing:
                    owner, reason = last_qn, "solution continuation"
                elif (item.get("solution_text") and existing and frag
                      and looks_truncated_solution(existing,
                                                   has_tables=bool(rec.get("tables")))
                      and not _frag_mostly_present(frag, existing)):
                    # PARTIAL owner append (run-8 tightening): only append a
                    # continuation to an owner whose existing solution PROVABLY
                    # ends mid-flow (truncated). Appending to a complete
                    # solution would glue a neighbour's or new question's text
                    # onto it -- a wrong-owner guess. The reliable signal
                    # matches the compute_carry S-pass fallback, so the two
                    # paths agree.
                    owner, reason = last_qn, "solution continuation (PARTIAL owner append)"
                elif item.get("options") and not rec.get("options") \
                        and orph.get("pass") in ("Q", None):
                    owner, reason = last_qn, "options continuation"
                elif item.get("question_text") and not rec.get("question_text") \
                        and orph.get("pass") in ("Q", None):
                    owner, reason = last_qn, "question continuation"
        # ---- rule 4: positional certainty (Gap-1). An orphan carrying the
        # STEM (+options) can only belong to a record that is MISSING its
        # stem. Text-similarity between a stem and its own solution is
        # always ~0 (they never overlap lexically), so similarity-based
        # matching provably fails here (prod: PSY-001-003 stayed stemless
        # with answer+solution intact). When the chapter has EXACTLY ONE
        # stem-less record, position alone is the proof. Gated to Q-pass
        # fragments: a solution/OCR fragment must never claim the stem slot.
        if owner is None and item.get("question_text") and item.get("options") \
                and orph.get("pass") in ("Q", None):
            stemless = [qn for qn, r in chapter_records.items()
                        if not (r.get("question_text") or "").strip()]
            if len(stemless) == 1:
                owner, reason = stemless[0], "question+options fallback (chapter's sole stem-less record)"
        if owner is None:
            print(f"  [ORPHAN] Could not determine owner: page={page} kept in orphans.jsonl")
            remaining.append(orph)
            continue
        rec = chapter_records[owner]
        # Wrong-owner guard (run-4: PSY-009-007): an orphan solution fragment
        # that BEGINS with an 'Option X:' explanation of an option the owner
        # does not have belongs to a DIFFERENT question -- never glue it on.
        # Other fields still merge; the blocked fragment stays visible.
        blocked_sol = bool(item.get("solution_text")
                           and _foreign_option_line(item["solution_text"].strip(), rec))
        if blocked_sol:
            stats["foreign_fragments_blocked"] = stats.get("foreign_fragments_blocked", 0) + 1
            print(f"  [WARN] [ORPHAN] blocked foreign solution fragment for q{owner} "
                  f"(starts with 'Option' line the owner cannot own) -- fragment kept in "
                  f"orphans.jsonl, other fields still merge")
            remaining.append({**orph, "blocked_reason":
                              "foreign Option-line head (wrong-owner guard); "
                              f"suspected owner differs from q{owner}"})
        sol_blocked = False
        if item.get("solution_text") and not blocked_sol:
            frag = item["solution_text"].strip()
            if frag and not _frag_mostly_present(frag, rec.get("solution_text") or ""):
                # Wrong-owner guard (same audit class as the retry append):
                # rule 3's "PARTIAL owner append" must not glue a NEIGHBOUR's
                # solution onto this record just because the overlap is low
                # (the audit's foreign-tail candidates: 006-014, 011-017,
                # 011-026, 012-002, 014-015, 022-008).
                foreign = _solution_fragment_foreign(frag, owner, rec, chapter_records)
                if foreign:
                    stats["foreign_fragments_blocked"] = stats.get("foreign_fragments_blocked", 0) + 1
                    print(f"  [WARN] [ORPHAN] blocked foreign solution fragment for q{owner} "
                          f"({foreign}) -- fragment kept in orphans.jsonl for review")
                    remaining.append({**orph, "blocked_reason": f"foreign solution fragment: {foreign}"})
                    sol_blocked = True
                else:
                    rec["solution_text"] = ((rec.get("solution_text") or "") + " " + frag).strip()
        orph_prov = f"ORPHAN_{str(orph.get('pass') or '?')}"
        # patch-only recovery (run-7 hardening #2): a fragment's question/
        # option content may ONLY merge when the fragment came from a Q-pass.
        # An S-pass/OCR solution fragment carrying stray question/option text
        # is blocked (cross-field contamination class), never merged.
        can_fill_question = orph.get("pass") in ("Q", None)
        if item.get("options") and can_fill_question:
            rec["options"] = rec["options"] or {}
            for k, v in item["options"].items():
                rec["options"].setdefault(str(k).strip().upper(), v)
            rec["_prov"]["options"] = orph_prov
        if item.get("question_text") and not rec.get("question_text"):
            if not can_fill_question:
                stats.setdefault("contaminated_stems_blocked", 0)
                stats["contaminated_stems_blocked"] += 1
                print(f"  [WARN] [ORPHAN] blocked {orph_prov} fragment from "
                      f"filling q{owner}'s stem (patch-only recovery) -- kept "
                      f"for review")
                remaining.append({**orph, "blocked_reason":
                                  f"{orph_prov} fragment carried question_text "
                                  f"(cross-field contamination) -- blocked"})
            else:
                stem_reason = _stem_reject_reason(item["question_text"], rec)
                if stem_reason:
                    stats.setdefault("contaminated_stems_blocked", 0)
                    stats["contaminated_stems_blocked"] += 1
                    print(f"  [WARN] [ORPHAN] blocked contaminated stem for q{owner} "
                          f"({stem_reason}) -- kept for review")
                    remaining.append({**orph, "blocked_reason":
                                      f"contaminated stem: {stem_reason}"})
                else:
                    rec["question_text"] = item["question_text"]
                    rec["_prov"]["question_text"] = orph_prov
        if item.get("correct_option") and not rec.get("correct_option"):
            rec["correct_option"] = str(item["correct_option"]).strip().upper()
            rec["_prov"]["correct_option"] = orph_prov
        if item.get("tables"):
            have = {t.get("markdown") for t in rec["tables"]}
            for t in item["tables"]:
                if t.get("markdown") not in have:
                    rec["tables"].append(t)
                    have.add(t.get("markdown"))
        qid = f"{subject}-{chapter_no:03d}-{owner:03d}"
        merged_something = bool(
            item.get("options") or item.get("question_text") or item.get("correct_option")
            or item.get("tables") or (item.get("solution_text") and not blocked_sol and not sol_blocked))
        if merged_something:
            note = (" (+ a foreign solution fragment was blocked, kept aside)"
                    if (blocked_sol or sol_blocked) else "")
            print(f"  [ORPHAN] Recovered orphan: page={page} assigned_to={qid} reason={reason}{note}")
            stats["orphans_recovered"] += 1
            if "carry-forward" in reason:
                stats["carry_merges"] = stats.get("carry_merges", 0) + 1
        elif blocked_sol:
            print(f"  [ORPHAN] owner q{owner} identified but the fragment added nothing new "
                  f"(foreign head) -- review the kept orphan entry")
    return remaining

IMAGE_ATTRIBUTION_PROMPT = """This image was extracted from one page of a medical MCQ chapter.

The chapter's questions are listed below (q_no: first words of stem):
{Q_LIST}

Look at the image and decide: does it BELONG to one of these questions
(a figure, diagram, chart, table, or clinical image that the question or
its solution refers to)?

Return ONE JSON object only:
{"q_no": <int>|null, "slot": "question"|"solution"|null, "decorative": true|false}
- q_no: the question this image belongs to. null if it belongs to none.
- slot: "question" if the figure appears with/above the stem, "solution" if
  it appears in the explanation region. null if q_no is null.
- decorative: true ONLY if you are confident this is decoration/unrelated to
  any question (portrait, logo, ornament, watermark, cover art, chapter icon).
Never guess a number. When unsure between decorative and a weak match,
prefer {"q_no": null, "decorative": true}.
"""


def attribute_orphan_image(model, rel_path, chapter_records, state):
    """FINAL safety net (Gap-2): one image, one call, one verdict. Never
    grouped -- a single image per call removes cross-image confusion.
    Returns (verdict_dict | None). Quota-brake-safe: returns
    {"decorative": "brake"} when the daily limit is hit so the caller can
    stop and persist instead of guessing."""
    reset_daily_counter_if_needed(state)
    if state["calls_today"] >= MAX_CALLS_PER_DAY:
        print("  [IMG] daily call limit reached during image attribution -- leftovers stay queued")
        return {"decorative": "brake"}
    q_list = "\n".join(
        f"q{qn}: {(chapter_records[qn].get('question_text') or '')[:80]}"
        for qn in sorted(chapter_records)
    ) or "(no question text available)"
    prompt = IMAGE_ATTRIBUTION_PROMPT.replace("{Q_LIST}", q_list)
    img_file = ASSETS_DIR / "questions" / rel_path
    # PACE THIS CALL (run-5 evidence): attribute_orphan_image calls
    # generate_content DIRECTLY, bypassing the 5s pacing every other path
    # enforces -- a page with 3 leftover images fired 3 calls in ~1.5s, and
    # a multi-page chapter fired several in the SAME microsecond (log:
    # 14:08:48.0293 x3). That burst is what pushes the free tier past its
    # 15 RPM window and triggers the 429s. Every Gemini call must be paced.
    _pace_gemini_call()
    try:
        resp = model.generate_content(
            [prompt, Image.open(img_file)],
            safety_settings=SAFETY_SETTINGS,
            request_options={"retry": None},
        )
        state["calls_today"] += 1
        save_state(state)
        if not resp.candidates:
            return None
        text = resp.text.strip()
        text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
        return json.loads(text)
    except Exception as e:
        print(f"  [IMG] attribution call failed for {rel_path}: {e}")
        return None


# ============================================================
# STEP 4: merge partial results (a question's text might be on one
# page and its answer/solution on a later page) into final records
# ============================================================

def merge_question_records(existing, new_items, stats=None, fill_only=False):
    """existing: dict keyed by q_no -> record (in progress for current chapter).
    stats: optional dict updated with "duplicates_merged"/"conflicts" counters.
    fill_only: recovery mode -- never overwrite a field that already has
    content; only fill what's missing (heals old rows without risking
    re-extraction noise replacing good data).

    Overlap-merge rules (sliding window re-extracts shared pages by design):
    - same q_no + question text similarity >= 95%  -> genuine re-extraction:
      merge fields (solutions/tables/options/images), count as duplicate.
    - same q_no but VERY different text AND a different answer key -> almost
      certainly a numbering collision: keep the first record, drop the item.
    Returns (existing, skipped): items with a missing/invalid q_no are NOT
    merged (never invent a number -- handoff rule #3) but ARE returned to the
    caller for orphan recovery (see ROOT_CAUSE_ANALYSIS.md RC-2)."""
    if stats is None:
        stats = {"duplicates_merged": 0, "conflicts": 0}
    skipped = []
    for item in new_items:
        raw_qn = item.get("q_no")
        if raw_qn is None:
            print(f"  [WARN] Gemini returned an item with no q_no, skipping: {str(item)[:200]}")
            skipped.append(item)
            continue
        try:
            qn = int(raw_qn)  # Gemini's JSON sometimes returns q_no as a
                               # string ("7") and sometimes as a number (7);
                               # force a consistent type so later sorting
                               # never compares int against str.
        except (TypeError, ValueError):
            print(f"  [WARN] Gemini returned a non-numeric q_no ({raw_qn!r}), skipping")
            skipped.append(item)
            continue
        # ---- PROVENANCE + PATCH-ONLY RECOVERY (run-7 hardening #1/#2/#4):
        # every item carries _prov (set by the pass that produced it, e.g.
        # "Q_PASS", "S_PASS", "A_RETRY", "OCR_S", "RECOVER"). A SOLUTION or
        # ANSWER recovery may ONLY patch solution/answer fields -- its
        # question_text/options are dropped here so a recovered solution
        # fragment can NEVER populate a stem (the cross-field contamination
        # class the audit found).
        prov = str(item.get("_prov") or "GEMINI")
        if prov.startswith("S") or prov.startswith("A"):
            if item.get("question_text") or item.get("options"):
                print(f"  [PROV] q{qn}: {prov} item carried question/option "
                      f"content -- dropped (patch-only recovery; a {prov} "
                      f"fragment must never fill a stem)")
                item = {**item, "question_text": None, "options": None}

        # Enforce the solution schema before any overlap merge.  A retry or
        # normal pass may put markdown tables in prose; route them to tables
        # so the final record never carries the same table twice.
        if item.get("solution_text"):
            clean_sol, item_tables = _normalize_solution_payload(
                item.get("solution_text"), item.get("tables") or [], qn)
            item = {**item, "solution_text": clean_sol, "tables": item_tables}

        rec = existing.setdefault(qn, {
            "q_no": qn, "question_text": None, "options": None,
            "correct_option": None, "solution_text": None, "tables": [],
            "has_figure_in_question": False, "has_figure_in_solution": False,
            "_prov": {},   # per-field provenance (run-7 hardening #4)
        })
        if "_prov" not in rec:
            rec["_prov"] = {}

        # ---- semantic stem guard (run-7 hardening #3/#6): a would-be stem
        # that OPENS with explanation language, or whose text is substantially
        # contained in this record's OWN solution, is not a stem -- reject the
        # field so the record stays stem-missing and becomes retry-eligible
        # (Gap-1 anchor: the solution names its question). Valid stems are
        # never touched.
        stem_reason = _stem_reject_reason(item.get("question_text"), rec)
        if stem_reason:
            stats.setdefault("contaminated_stems_rejected", 0)
            stats["contaminated_stems_rejected"] += 1
            _append_jsonl(DATA_DIR / "integrity_flags.jsonl",
                          {"kind": "contaminated_stem_rejected", "q_no": qn,
                           "chapter_id": stats.get("chapter_id"),
                           "detail": stem_reason,
                           "prov": prov,
                           "text": str(item["question_text"])[:300]})
            print(f"  [WARN] q{qn}: rejected contaminated stem ({stem_reason}; "
                  f"prov={prov}) -- field kept empty for retry")
            item = {**item, "question_text": None}
        # ---- duplicate / conflict classification for overlap pages ----
        old_q, new_q = rec.get("question_text"), item.get("question_text")
        if old_q and new_q:
            sim = difflib.SequenceMatcher(None, old_q, new_q).ratio()
            if sim >= 0.95:
                stats["duplicates_merged"] += 1   # expected overlap re-read
            else:
                a1, a2 = rec.get("correct_option"), item.get("correct_option")
                # normalize case BEFORE comparing: 'D' vs 'd' is the same
                # answer, but the raw string compare called it a conflict and
                # dropped the item (run-2 log 17:36:00, ch15 q1).
                a1 = str(a1).strip().upper() if a1 else None
                a2 = str(a2).strip().upper() if a2 else None
                if a1 and a2 and a1 != a2:
                    stats["conflicts"] += 1   # count only ACTUAL drops (matches "conflicts dropped" log label)
                    print(f"  [WARN] conflicting re-extraction for q{qn} "
                          f"(similarity {sim:.2f}, answers {a1} vs {a2}) -- keeping first, dropping item")
                    continue
                # STEM CONFLICT (run-4: PSY-012-001 silently got PSY-012-013's
                # stem, similarity 0.25 in the log, wrong stem won by write
                # order). A stem coheres with its OWN options+solution; pick
                # the variant that matches the record's payload. When the
                # scores can't decide, keep the first and log BOTH variants to
                # data/stem_conflicts.jsonl -- never silently guess again.
                # fill_only (recovery) mode never overwrites an existing stem:
                # the existing row keeps its text, only the ledger note is written.
                if fill_only:
                    # recovery mode never overwrites an existing stem; log only.
                    stats["stem_conflicts"] = stats.get("stem_conflicts", 0) + 1
                    _append_jsonl(DATA_DIR / "stem_conflicts.jsonl", {
                        "q_no": qn, "chapter_id": stats.get("chapter_id"),
                        "similarity": round(sim, 3), "verdict": "fill-only kept-existing",
                        "old_stem": old_q[:600], "new_stem": new_q[:600]})
                else:
                    co, cn = _stem_payload_coherence(old_q, rec), _stem_payload_coherence(new_q, rec)
                    if abs(co - cn) >= STEM_COHERENCE_MARGIN and max(co, cn) > 0:
                        keep, verdict = (old_q, "kept-old") if co > cn else (new_q, "kept-new")
                    else:
                        keep, verdict = old_q, "kept-old (undecidable -- review logged)"
                    stats["stem_conflicts"] = stats.get("stem_conflicts", 0) + 1
                    _append_jsonl(DATA_DIR / "stem_conflicts.jsonl", {
                        "q_no": qn, "chapter_id": stats.get("chapter_id"),
                        "similarity": round(sim, 3),
                        "coherence_old": round(co, 3), "coherence_new": round(cn, 3),
                        "verdict": verdict, "old_stem": old_q[:600], "new_stem": new_q[:600]})
                    print(f"  [WARN] stem conflict for q{qn} (similarity {sim:.2f}, "
                          f"coherence {co:.2f} vs {cn:.2f}) -- {verdict}; "
                          f"both variants logged to stem_conflicts.jsonl")
                    rec["question_text"] = keep
                item = {**item, "question_text": None}  # block the generic loop below
        for k in ["question_text", "solution_text"]:
            if item.get(k):
                if fill_only and rec.get(k):
                    continue  # recovery: never overwrite existing content
                rec[k] = item[k]
                rec["_prov"][k] = prov   # provenance of every patched field

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
                key = str(opt_id).strip().upper()
                if fill_only:
                    rec["options"].setdefault(key, opt_text)
                else:
                    rec["options"][key] = opt_text
            rec["_prov"]["options"] = prov

        if item.get("correct_option"):
            if not (fill_only and rec.get("correct_option")):
                rec["correct_option"] = str(item["correct_option"]).strip().upper()
                rec["_prov"]["correct_option"] = prov

        if item.get("tables"):
            # Overlap captures can be byte-identical OR a shorter prefix when
            # the first batch ends mid-table. Keep the fullest table, not the
            # first table merely because it arrived first.
            rec["tables"] = _dedupe_tables(list(rec.get("tables") or []) +
                                            list(item.get("tables") or []))
        rec["has_figure_in_question"] = rec["has_figure_in_question"] or item.get("has_figure_in_question", False)
        rec["has_figure_in_solution"] = rec["has_figure_in_solution"] or item.get("has_figure_in_solution", False)
    return existing, skipped

def sanitize_solution_text(text, own_qn=None):
    """Strip print furniture that leaks into solution text (run-4 audit):
      1. leading verbatim book headers  ("Solution to Question 2:") that the
         model carried over (PSY-032-001/002 shipped with them on).
      2. an EMBEDDED later "Solution to Question N:" header whose tail is a
         duplicate of what already precedes it -- the model dumped the whole
         recitation block into one question (PSY-032-003 carried its own
         solution twice plus Q4/Q5 inline). Non-duplicate tails (possibly the
         neighbor's real content) are LEFT intact and reported, never cut.
    Returns (cleaned_text, notes)."""
    notes = []
    s = text or ""
    if not s.strip():
        return s, notes
    while True:
        m = re.match(r"\s*Solution\s+to\s+Question\s+\d{1,3}\s*[:.\-]?\s*", s, re.IGNORECASE)
        if not m:
            break
        s = s[m.end():]
        notes.append("stripped leading 'Solution to Question N' header")
    m = SOLUTION_TO_Q_RE.search(s)
    if m and m.start() > 0:
        head = s[:m.start()].rstrip()
        tail = s[m.end():].lstrip(" :\n")
        # The precise dump proof: the chunk IMMEDIATELY after the header
        # restates THIS solution's own earlier content (the model re-recited
        # this question before dumping its neighbours). Neighbour content
        # further down the tail is never judged -- only the first line.
        tail_first = tail.split("\n", 1)[0][:150]
        if head and tail_first and _frag_mostly_present(tail_first, head, 0.8):
            s = head
            notes.append(f"truncated duplicated 'Solution to Question {m.group(1)}' dump")
        else:
            notes.append(f"embedded 'Solution to Question {m.group(1)}' header kept "
                         f"(tail not a duplicate -- needs model/review)")
    return s, notes


def _is_printed_answer_key(t):
    """The book's printed Answer Key grid is never solution content --
    when the model inline-reads the answers page it rides into solutions
    (zip-8: 39 stray-key flags across 18 chapters). Strip at the source
    so future books never carry them; fix_output.py P10 heals old files."""
    ty = str(t.get("type") or "").strip().lower().replace("_", " ")
    md = (t.get("markdown") or "")
    head = md.lstrip().splitlines()[0] if md.strip() else ""
    return ty == "answer key" or ("Question No." in head and "Correct Option" in head)


def _anchorless_record(rec):
    """True when a record carries NO usable content: no stem, no options and
    no solution (ch24 q12/13 class -- phantom rows born from an answer-key
    table spanning chapters, or fully-lost fragments). Such rows are dropped
    at build time with a ledger entry; the chapter's question count reflects
    only real records."""
    return not ((rec.get("question_text") or "").strip()
                or (rec.get("options") or {})
                or (rec.get("solution_text") or "").strip())


def build_final_question(subject, chapter_id, chapter_no, q_no, rec, image_files):
    qid = f"{subject}-{chapter_no:03d}-{q_no:03d}"

    def valid_images(imgs, kind):
        out = []
        for f in imgs:
            if not IMG_PATH_RE.match(f):
                print(f"  [WARN] Dropping malformed {kind} image path for {qid}: {f}")
                continue
            p = ASSETS_DIR / "questions" / f
            if not p.exists():
                print(f"  [WARN] Dropping missing {kind} image ref for {qid}: {f}")
                continue
            size = p.stat().st_size
            if size < MIN_IMAGE_BYTES:
                print(f"  [WARN] Dropping suspicious-tiny ({size}B) {kind} image ref for "
                      f"{qid}: {f} -- broken-crop guard (never ship a broken figure)")
                continue
            out.append({"type": "figure", "file": f})
        return out

    q_images = valid_images(image_files.get("question", []), "question")
    sol_images = valid_images(image_files.get("solution", []), "solution")
    sol_text, sanitize_notes = sanitize_solution_text(rec.get("solution_text"), own_qn=q_no)
    for note in sanitize_notes:
        print(f"  [SANITIZE] {qid}: {note}")
    tables = [{"type": t.get("type", "table"), "markdown": t["markdown"], "file": None}
              for t in _dedupe_tables(rec.get("tables", []))
              if not _is_printed_answer_key(t)]

    option_rows = [{"id": str(k).strip().upper(), "text": v, "images": []}
                   for k, v in (rec["options"] or {}).items()]
    # Last-resort release backfill: targeted retry above requests all options
    # when one is blank. If OCR/model extraction still leaves the *correct*
    # option blank, preserve usability with the solution's opening sentence
    # and make the repair conspicuous for validator/manual review.
    correct_id = str(rec.get("correct_option") or "").strip().upper()
    for opt in option_rows:
        if opt["id"] == correct_id and not str(opt.get("text") or "").strip():
            first = re.split(r"(?<=[.!?])\s+", sol_text.strip(), maxsplit=1)[0].strip()
            if first:
                opt["text"] = first
                print(f"  [OPTION_BACKFILLED] {qid}: correct option {correct_id} reconstructed from solution opening")
    # Correct clearly mislabelled "Option X:" explanation lines only when the
    # description overlaps another option at least twice as strongly.
    opt_text = {o["id"]: str(o.get("text") or "") for o in option_rows}
    def relabel(m):
        label, desc = m.group(1).upper(), m.group(2)
        words = {w for w in re.findall(r"\w+", desc.lower()) if len(w) > 2}
        scores = {k: len(words & set(re.findall(r"\w+", v.lower()))) for k, v in opt_text.items() if v}
        best = max(scores, key=scores.get) if scores else label
        if best != label and scores.get(best, 0) >= 2 * max(1, scores.get(label, 0)):
            print(f"  [LABEL_CORRECTED] {qid}: Option {label} -> Option {best}")
            return f"Option {best}: {desc}"
        return m.group(0)
    sol_text = re.sub(r"(?m)Option\s+([A-D])\s*:\s*([^\n]+)", relabel, sol_text)

    return {
        "id": qid,
        "subject": subject,
        "chapter_id": chapter_id,
        "question": {"text": rec["question_text"], "images": q_images},
        "options": option_rows,
        "correct_options": [rec["correct_option"]] if rec["correct_option"] else [],
        "solution": {"text": sol_text, "images": sol_images, "tables": tables},
        "tags": [],
    }

def repair_option_labels(question):
    """Correct only high-confidence mislabeled Option X explanation lines."""
    text = ((question.get("solution") or {}).get("text") or "")
    options = {str(o.get("id")).upper(): str(o.get("text") or "")
               for o in question.get("options") or []}
    def repl(match):
        label, desc = match.group(1).upper(), match.group(2)
        words = {w.lower() for w in re.findall(r"\w+", desc) if len(w) > 2}
        scores = {oid: len(words & set(re.findall(r"\w+", val.lower()))) / max(1, len(set(re.findall(r"\w+", val.lower()))))
                  for oid, val in options.items() if val}
        if not scores:
            return match.group(0)
        best = max(scores, key=scores.get)
        if best != label and scores[best] >= 2 * max(scores.get(label, 0.0), 0.01):
            print(f"  [LABEL_CORRECTED] {question.get('id')}: Option {label} -> Option {best}")
            return f"Option {best}: {desc}"
        return match.group(0)
    question["solution"]["text"] = re.sub(r"(?m)Option\s+([A-D])\s*:\s*([^\n]+)", repl, text)
    return question

# ============================================================
# MAIN DRIVER
# ============================================================

def _mat_mult(m1, m2):
    """2D affine composition CTM' = M1 x M2 (PDF row-vector convention)."""
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return (a1 * a2 + b1 * c2, a1 * b2 + b1 * d2,
            c1 * a2 + d1 * c2, c1 * b2 + d1 * d2,
            e1 * a2 + f1 * c2 + e2, e1 * b2 + f1 * d2 + f2)


def image_positions_on_page(pdf_path, file_page):
    """Best-effort map {image object idnum -> (y, draw_index)} for every
    image XObject drawn on a page, by walking the content stream and
    tracking the cm matrix before each `Do`. y = height from page bottom
    (PDF origin), so LARGER y == HIGHER on the page. Returns {} on any
    parse hiccup -- callers then fall back to plain reading order."""
    positions = {}
    try:
        page = PdfReader(pdf_path).pages[file_page - 1]
        xobjs = _page_xobjects(page)
        names = {str(name): ref for name, ref in xobjs.items()}
        contents = page.get_contents()
        if contents is None:
            return {}
        data = contents.get_data() if hasattr(contents, "get_data") else None
        if not data:
            return {}
        import zlib
        try:
            data = zlib.decompress(data)
        except Exception:
            pass
        # tokenize the small subset we care about: q, Q, cm, Do
        tokens = re.findall(rb"/[^\s\[\]()<>{}/%]+|\([^)]*\)|\[[^\]]*\]|"
                            rb"[-+]?\d*\.?\d+|[A-Za-z'\"]+", data)
        ctm = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        stack = []
        num_buf = []
        draw_idx = 0
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t == b"q":
                stack.append(ctm)
            elif t == b"Q":
                ctm = stack.pop() if stack else ctm
            elif t == b"cm" and len(num_buf) >= 6:
                m = tuple(float(x) for x in num_buf[-6:])
                ctm = _mat_mult(m, ctm)
                num_buf = []
            elif t == b"Do" and num_buf:
                name = num_buf[-1].decode("latin-1")
                if name in names:
                    obj = _resolve(names[name])
                    if obj.get("/Subtype") == "/Image":
                        oid = getattr(names[name], "idnum", None)
                        key = oid if oid is not None else name
                        positions[key] = (ctm[5], draw_idx)
                        draw_idx += 1
                num_buf = []
            i += 1
            if t not in (b"q", b"Q", b"cm", b"Do"):
                if t.startswith(b"/") or re.fullmatch(rb"[-+]?\d*\.?\d+", t):
                    num_buf.append(t)
                else:
                    num_buf = []
        return positions
    except Exception:
        return {}


def pending_image_slots(chapter_records, image_files_by_q):
    """Chapter's needy image slots in APPEARANCE order: q_no ascending, and
    within one question its question-side figure precedes its solution-side
    figure (that's the physical reading order in MCQ books)."""
    slots = []
    for qn in sorted(chapter_records):
        rec = chapter_records[qn]
        entry = image_files_by_q.setdefault(qn, {"question": [], "solution": []})
        if rec.get("has_figure_in_question") and not entry["question"]:
            slots.append((qn, "question"))
        if rec.get("has_figure_in_solution") and not entry["solution"]:
            slots.append((qn, "solution"))
    return slots


def _rename_for_slot(rel, qn, kind, subject, chapter_no, image_files_by_q):
    """Rename one extracted temp image into the locked convention for the
    given (q_no, "question"|"solution") slot. kind letter: Q or SOL.
    Returns the new rel path or None."""
    old_path = ASSETS_DIR / "questions" / rel
    if not old_path.exists():
        print(f"  [WARN] {rel} missing at rename time -- skipping (alias/dup ref)")
        return None
    qid = f"{subject}-{chapter_no:03d}-{qn:03d}"
    entry = image_files_by_q.setdefault(qn, {"question": [], "solution": []})
    # Broken-crop guard (run-4: PSY-003-014_Q_01 was 414 bytes): a sub-1.5KB
    # webp cannot hold a real MCQ figure. Do NOT auto-claim it -- the caller's
    # leftover path hands it to the model fourth-pass / manual review, which
    # decides on ACTUAL content instead of position.
    size = old_path.stat().st_size
    if size < MIN_IMAGE_BYTES:
        print(f"  [WARN] {rel} is only {size}B (< {MIN_IMAGE_BYTES}) -- refusing auto-claim "
              f"(broken-crop guard); left for model/manual review")
        return None
    # Over-attribution guard (run-4: PSY-022-003 collected 7 question-side
    # images through repeated model-confirmed passes -- every pass was
    # individually reasonable, the SUM was nonsense). One question in this
    # book never legitimately cites >3 figures.
    if kind == "question" and len(entry["question"]) >= MAX_QUESTION_IMAGES:
        print(f"  [WARN] over-attribution guard: {qid} already has {MAX_QUESTION_IMAGES} "
              f"question images -- refusing {rel}; left for model/manual review")
        return None
    # Same guard on the solution side (user report: a 7-figure solutions page
    # collapsed into 2 solutions because a single decoded header swallowed
    # every image under it). A solution block legitimately cites a figure or
    # two; beyond that the deterministic matcher is stacking neighbours'
    # figures -- refuse and let the model/manual pass decide on content.
    if kind == "solution" and len(entry["solution"]) >= MAX_SOLUTION_IMAGES:
        print(f"  [WARN] over-attribution guard: {qid} already has {MAX_SOLUTION_IMAGES} "
              f"solution images -- refusing {rel}; left for model/manual review")
        return None
    letter = "Q" if kind == "question" else "SOL"
    idx = len(entry[kind]) + 1
    new_name = f"{qid}_{letter}_{idx:02d}.webp"
    new_rel = f"{subject}/{new_name}"
    old_path.rename(ASSETS_DIR / "questions" / subject / new_name)
    return new_rel


def claim_page_images_one_to_one(imgs, pdf_path, file_page, subject, chapter_no,
                                 chapter_records, image_files_by_q):
    """Gap-2 core matcher: distribute one page's N extracted images across
    the chapter's needy slots ONE-TO-ONE, in reading order: images sorted
    top->bottom by their drawn y-position (positions parsed from the PDF
    content stream; falls back to resource order), slots in appearance order
    (pending_image_slots). Returns the list of files STILL unclaimed.
    With 0 or 1 needy slot, degenerates to the old greedy behavior (all
    page images go to that one slot) -- which is correct for a page whose
    images all belong to a single question."""
    # Never distribute page images across chapter-wide "pending" slots by
    # reading order alone. That heuristic silently mapped diagrams to the
    # wrong questions whenever a page had several nearby questions. Auto-claim
    # only with deterministic page evidence: exactly one printed q_no on this
    # page and exactly one matching needy slot. Everything else is retained
    # for the later explicit attribution/manual-review path.
    slots = pending_image_slots(chapter_records, image_files_by_q)
    if not slots:
        return list(imgs)
    try:
        printed = qns_printed_on_page(pdf_path, file_page, chapter_records)
    except Exception:
        printed = []
    if len(printed) != 1:
        print(f"  [IMG] page {file_page}: ambiguous printed owners {printed or '-'}; "
              "not auto-attaching image(s)")
        return list(imgs)
    candidates = [(qn, kind) for qn, kind in slots if qn == printed[0]]
    if len(candidates) != 1:
        print(f"  [IMG] page {file_page}: q{printed[0]} has {len(candidates)} eligible image slots; "
              "not auto-attaching image(s)")
        return list(imgs)
    if len(candidates) == 1:
        qn, kind = candidates[0]
        entry = image_files_by_q.setdefault(qn, {"question": [], "solution": []})
        leftover = []
        for rel in imgs:
            # append IMMEDIATELY after each rename: _rename_for_slot derives
            # the _01/_02/... suffix from len(entry[kind]), so deferring the
            # append would hand the same filename to every image on this
            # page and silently overwrite them (caught by tests).
            new_rel = _rename_for_slot(rel, qn, kind, subject, chapter_no, image_files_by_q)
            if new_rel:
                entry[kind].append(new_rel)
            else:
                leftover.append(rel)
        return leftover
    # N images, M>=2 slots: position-ordered one-to-one
    pos = image_positions_on_page(pdf_path, file_page)

    def order_key(rel):
        try:
            oid = int(Path(rel).stem.rsplit("-", 1)[-1])
        except (ValueError, IndexError):
            oid = None
        y, didx = pos.get(oid, (None, 10**6))
        return (-(y if y is not None else float("-inf")), didx)

    ordered_imgs = sorted(imgs, key=order_key)
    leftover = []
    for i, rel in enumerate(ordered_imgs):
        if i >= len(slots):
            leftover.append(rel)
            continue
        qn, kind = slots[i]
        new_rel = _rename_for_slot(rel, qn, kind, subject, chapter_no, image_files_by_q)
        if new_rel:
            image_files_by_q.setdefault(qn, {"question": [], "solution": []})[kind].append(new_rel)
            qid = f"{subject}-{chapter_no:03d}-{qn:03d}"
            print(f"  [IMG] one-to-one: {rel} -> {qid} ({kind} slot #{i + 1})")
        else:
            leftover.append(rel)
    return leftover


def claim_solution_page_images(imgs, pdf_path, file_page, subject, chapter_no,
                               chapter_records, image_files_by_q):
    """Deterministic owner for figures printed inside solution blocks.

    Every image on a solutions page belongs to the block it is DRAWN UNDER:
    the last "Solution to Question N:" header whose baseline sits above the
    image's bottom edge. Images and headers are matched by real PDF y
    positions (same coordinate space), never by count or by a single
    text-layer hit.

    This replaces the old shortcut that attached EVERY image of a page to the
    ONE solution header the text layer happened to decode -- the exact bug
    behind the user report of a 7-figure solutions page collapsing into just
    2 solutions (the text layer of scanned books decodes headers
    sporadically, and a single decoded header was treated as "the page's
    owner").

    Safety rules (each returns the image unclaimed rather than guessing):
      * no locatable header on the page -> claim NOTHING;
      * image position unparsable -> claim NOTHING;
      * image with no header ABOVE it (block started on the previous page,
        figure-above-header layout) -> claim NOTHING;
      * MAX_SOLUTION_IMAGES per solution (enforced in _rename_for_slot): the
        first figures under a header are kept, extras flow to the model/
        manual pass instead of stacking a whole page on one solution.
    Returns the files STILL unclaimed."""
    headers = solution_headers_on_page(pdf_path, file_page, chapter_records)
    if not headers:
        # No locatable header -- a question page, or a scanned solutions page
        # whose text layer cannot be read. Claim NOTHING by position (a
        # whole-page dump is exactly the bug being fixed); the leftovers flow
        # to the one-to-one matcher and the model/manual passes, which is the
        # same safe path the pipeline used before header binding existed.
        return list(imgs)
    pos = image_positions_on_page(pdf_path, file_page)
    if not pos:
        print(f"  [IMG] page {file_page}: image positions unparsable -- solution "
              f"figures left for model/manual attribution (no positional auto-claim)")
        return list(imgs)
    leftover = []
    for rel in imgs:
        try:
            oid = int(Path(rel).stem.rsplit("-", 1)[-1])
        except (ValueError, IndexError):
            leftover.append(rel)
            continue
        info = pos.get(oid)
        if info is None:
            leftover.append(rel)
            continue
        y_img, _didx = info[0], info[1]
        # The owner is the CLOSEST header drawn above the image, i.e. the
        # last one in reading order whose baseline still sits above the
        # image's bottom edge. Iterate headers bottom-first and take the
        # first hit -- taking the topmost header above the image instead
        # would hand every figure on the page to the first solution block.
        owner = next((qn for qn, y_hdr in reversed(headers) if y_hdr > y_img), None)
        if owner is None:
            leftover.append(rel)   # no block starts above this figure
            continue
        new_rel = _rename_for_slot(rel, owner, "solution", subject, chapter_no,
                                   image_files_by_q)
        if new_rel:
            image_files_by_q.setdefault(owner, {"question": [], "solution": []})["solution"].append(new_rel)
            qid = f"{subject}-{chapter_no:03d}-{owner:03d}"
            print(f"  [IMG] page {file_page}: solution-block position -> {rel} -> {qid} (solution)")
        else:
            leftover.append(rel)
    return leftover


def _order_imgs_by_position(imgs, pos):
    """Sort extracted image rel paths top->bottom by their drawn y-position
    (PDF content stream), falling back to resource order when positions are
    unparsable. Both the figure-map pass and the one-to-one matcher rely on
    this ordering matching Gemini's top-to-bottom reading order."""
    def order_key(rel):
        try:
            oid = int(Path(rel).stem.rsplit("-", 1)[-1])
        except (ValueError, IndexError):
            oid = None
        y, didx = pos.get(oid, (None, 10**6))
        return (-(y if y is not None else float("-inf")), didx)
    return sorted(imgs, key=order_key)


def claim_figure_map_images(fig_map, window_rows, subject, chapter_no,
                            chapter_records, image_files_by_q):
    """Run-6 user ask ("bta ye image kis question ki h"): claim every image
    of a window using Gemini's OWN _figure_map (one {q_no, slot} entry per
    figure, in top-to-bottom reading order page by page, returned by the
    extraction prompt).

    window_rows: [(file_page_num, [rel paths in top-to-bottom order]), ...]
    with pages in window order.

    EXACT-COUNT GUARD: the map only fires when len(fig_map) == the total
    number of images extracted for the window -- then the alignment is exact
    because both lists are top-to-bottom, page by page. ANY mismatch
    (watermark skipped, tiny crop dropped by the guard, model double-counted
    or missed a figure) skips the whole pass safely; those images stay for
    the deterministic positional passes and the 4th-pass model attribution.
    Returns {page_no: [rels still unclaimed]}."""
    if not fig_map:
        return {p: rels for p, rels in window_rows}
    total_imgs = sum(len(rels) for _, rels in window_rows)
    if total_imgs != len(fig_map):
        print(f"  [IMG] figure-map count mismatch ({len(fig_map)} declared vs "
              f"{total_imgs} extracted) -- skipping model figure-map claim; "
              f"left for positional/model passes")
        return {p: rels for p, rels in window_rows}
    remaining = {}
    it = iter(fig_map)
    for page_no, rels in window_rows:
        still = []
        for rel in rels:
            entry = next(it, None)
            if not entry:
                still.append(rel)
                continue
            try:
                qn = int(entry.get("q_no"))
            except (TypeError, ValueError):
                still.append(rel)
                continue
            if qn not in chapter_records:
                still.append(rel)
                continue
            slot = entry.get("slot")
            if slot not in ("question", "solution"):
                still.append(rel)
                continue
            new_rel = _rename_for_slot(rel, qn, slot, subject, chapter_no,
                                       image_files_by_q)
            if new_rel:
                image_files_by_q.setdefault(qn, {"question": [], "solution": []})[slot].append(new_rel)
                qid = f"{subject}-{chapter_no:03d}-{qn:03d}"
                print(f"  [IMG] figure-map: {rel} -> {qid} ({slot} side, model-declared)")
            else:
                still.append(rel)   # tiny-crop / over-attribution guard refused
        if still:
            remaining[page_no] = still
    return remaining


def claim_page_images(imgs, pdf_path, file_page, subject, chapter_no,
                      chapter_records, image_files_by_q):
    """Two-stage deterministic claimer for one page's images:

      1. solution-header mapping -- every image drawn under a printed
         "Solution to Question N:" header goes to THAT solution (position
         evidence; see claim_solution_page_images). This fixes the
         whole-page-dump: a single decoded header no longer swallows every
         figure of a multi-solution page.
      2. leftovers fall to the one-to-one matcher (exactly one printed q_no +
         exactly one needy slot, else nothing is claimed).

    Returns the files STILL unclaimed (they reach the second pass / model
    attribution / manual review)."""
    leftover = claim_solution_page_images(imgs, pdf_path, file_page, subject,
                                          chapter_no, chapter_records, image_files_by_q)
    if leftover:
        leftover = claim_page_images_one_to_one(leftover, pdf_path, file_page,
                                                subject, chapter_no, chapter_records,
                                                image_files_by_q)
    return leftover


def _append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def process_pdf(pdf_cfg, state, genai_model, chapters_out, questions_fh,
                only_chapter_no=None):
    """only_chapter_no (v2 test hook): when set, every other chapter is
    skipped -- lets test_v2_chapter.py run the full 3-pass machinery on ONE
    chapter without touching the rest of the book."""
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
        if only_chapter_no is not None and ch["chapter_no"] != only_chapter_no:
            continue
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
        image_files_by_q = {}
        pages_imaged = set()       # overlap pages must not be image-extracted twice
        unmatched_images = []      # no claimant yet -- retried at chapter end
        orphans = []               # Gemini items with null/invalid q_no (RC-2)
        stats = {"batches": 0, "duplicates_merged": 0, "conflicts": 0,
                 "carry_used": 0, "carry_merges": 0,
                 "orphans_recovered": 0, "orphans_buffered": 0, "orphans_remaining": 0,
                 "chapter_id": chapter_id}
        # V2: per-pass carry-forward state (Q-pass and S-pass track their own
        # open items; A-pass items are one-shot key rows, no carry needed).
        carry_by_pass = {"Q": None, "S": None}     # FEATURE 2 payloads, per pass
        carry_trackers = {"Q": {}, "S": {}}        # q_no -> batch-seq of UNRESOLVED carry
        carry_banned = {"Q": set(), "S": set()}    # expired q_nos: never respawn
        solutions_section_seen = False             # sticky once the Solutions section begins
        prev_window_last_page = None

        # SECTION-AWARE WINDOWS (run-6): detect questions/answers/solutions
        # boundaries from the text layer ONCE and send each section in its own
        # larger windows (whole question section in 1-2 calls -> no boundary
        # splits, no overlap waste; answers in one call; solutions in
        # recitation-safe chunks). Falls back to the fixed 6-page window loop
        # when the text layer can't be read (scanned-only pages).
        section_windows = build_section_windows(page_files, pdf_path)
        if section_windows:
            page_by_no = {int(p.stem.split("-")[-1]): p for p in page_files}
            window_specs = []
            for page_nos, sec in section_windows:
                batch = [page_by_no[n] for n in page_nos if n in page_by_no]
                if batch:
                    window_specs.append((batch, sec))
            if not window_specs:
                section_windows = []  # degenerate -> fall back below
        if not section_windows:
            overlap = max(0, min(BATCH_OVERLAP_PAGES, PAGES_PER_GEMINI_CALL - 1))
            batch_step = PAGES_PER_GEMINI_CALL - overlap
            window_specs = []
            for batch_start in range(0, len(page_files), batch_step):
                if batch_start and batch_start + overlap >= len(page_files):
                    break  # trailing window would contain ONLY overlap pages
                           # (nothing new) -- don't spend a quota call on it
                window_specs.append(
                    (page_files[batch_start:batch_start + PAGES_PER_GEMINI_CALL], None))
        prev_section = None
        for batch, section in window_specs:
            window_pages = [int(p.stem.split("-")[-1]) for p in batch]
            # provenance anchor used by the log lines / orphan records below
            # (the fixed-window loop used its index; the section loop uses the
            # window's first PDF page -- equally unique per window)
            batch_start = window_pages[0] if window_pages else 0
            if section is None:
                # fixed-window fallback: keep the original overlap semantics
                overlap_pages = [pn for pn in window_pages
                                 if prev_window_last_page is not None
                                 and pn <= prev_window_last_page]
            elif section == prev_section:
                # intra-section overlap only -- cross-section windows share
                # NOTHING (that was the token waste: fixed windows re-sent the
                # previous section's tail pages in every new window)
                overlap_pages = [pn for pn in window_pages
                                 if prev_window_last_page is not None
                                 and pn <= prev_window_last_page]
            else:
                overlap_pages = []  # first window of a section: no overlap
            new_pages = [pn for pn in window_pages if pn not in overlap_pages]
            if not new_pages:
                continue  # trailing window = pure overlap; nothing new
            stats["batches"] += 1

            if section == "S" and not solutions_section_seen:
                # text-layer section boundary: hard-reset ALL carry context
                # before the Solutions section, exactly like the extraction-
                # based boundary guard, so question/solution prose can never
                # cross-merge (the stale-carry class). NOTE: we deliberately do
                # NOT flip solutions_section_seen here -- Q-pass stays active
                # until the EXTRACTION-based boundary fires (probe below),
                # exactly like the old fixed windows, so the handful of
                # questions that tail into the first solution pages (ch1 class:
                # 3 questions on pages 11-16) are never skipped by a text-layer
                # guess.
                had_pending = any(v is not None for v in carry_by_pass.values()) \
                    or any(carry_trackers.values())
                carry_by_pass = {"Q": None, "S": None}
                carry_trackers = {"Q": {}, "S": {}}
                print(f"  [SECTION] solutions section begins at page "
                      f"{window_pages[0]} (text-layer detected) -- ALL carry "
                      f"context HARD-RESET"
                      f"{' (dropped pending context)' if had_pending else ''}; "
                      f"pass activation unchanged (extraction boundary decides)")
            # V2 pass activation (zero-token pdftotext probe + sticky section
            # state) -- IDENTICAL for section and fallback windows: questions-
            # section batch -> Q-pass only; solutions section -> S-pass only;
            # key tables / solution headers on THESE pages -> +A-pass / S-pass.
            # A probe failure (scanned-only PDF) returns all-True -> all passes
            # run (safe). Never let a window-sizer disable a pass -- the text
            # layer of scanned books mislabels pages, and a skipped Q-pass
            # would silently drop those questions.
            probe = probe_batch_pages(pdf_path, window_pages)
            do_s = solutions_section_seen or probe["solutions"]
            do_a = probe["key_table"]
            do_q = not solutions_section_seen
            if not (do_q or do_s or do_a):
                do_q = True  # eerily silent page (figures only?) -- default to Q-pass

            # Proactive route: do not first trigger Gemini recitation on a
            # printed, clinically sensitive solutions page. OCR/header merge
            # owns it before any vision pass is built.
            routed_pages = set()
            for pf in batch:
                page_no = int(pf.stem.split("-")[-1])
                if is_recitation_risk_solution_page(pdf_path, page_no):
                    raw_ocr = ocr_fallback_text(pf)
                    n = _recover_ocr_solution_headers(raw_ocr, chapter_records)
                    if n:
                        routed_pages.add(pf)
                        print(f"  [PREFLIGHT_OCR] {pf.name}: header-routed {n} solution(s); Gemini skipped")

            fig_map_by_pass = {}   # this window's _figure_map control objects
            for pass_name, prompt, active in (
                    ("Q", SCHEMA_PROMPT_Q, do_q),
                    ("A", SCHEMA_PROMPT_A, do_a),
                    ("S", SCHEMA_PROMPT_S, do_s)):
                if not active:
                    continue
                reset_daily_counter_if_needed(state)
                if state["calls_today"] >= MAX_CALLS_PER_DAY:
                    print("Daily Gemini call limit reached. Saving progress, exiting.")
                    save_state(state)
                    sys.exit(0)
                pass_batch = [pf for pf in batch if pf not in routed_pages]
                if not pass_batch:
                    continue
                carry_in = carry_by_pass.get(pass_name) if pass_name in ("Q", "S") else None
                context_str = build_carry_context(carry_in, overlap_pages, new_pages)
                if carry_in:
                    stats["carry_used"] += 1
                try:
                    raw_items = call_gemini_on_pages(genai_model, pass_batch,
                                                     context=context_str, prompt=prompt)
                    state["calls_today"] += 1
                    save_state(state)
                except Exception as e:
                    err_text = str(e)
                    if "finish_reason=8" in err_text or "PROHIBITED_CONTENT" in err_text:
                        event = {"subject": subject, "chapter_id": chapter_id, "chapter_no": ch["chapter_no"],
                                 "pass": pass_name, "pages": window_pages, "reason": err_text[:240]}
                        state.setdefault("safety_blocked", []).append(event)
                        save_state(state)
                        print(f"  [SAFETY_BLOCKED] {subject} {chapter_id} {pass_name}-pass pages {window_pages} "
                              "-- queued for recovery and manual review", flush=True)
                    if "429" in err_text or "quota" in err_text.lower():
                        # Free tier = ~1500 req/day PER DAY but also ~15 RPM per
                        # minute. A burst 429 is NOT the daily cap -- back off
                        # once before declaring the whole day over.
                        print(f"  [429] rate limited on {subject} ch{ch['chapter_no']} "
                              f"batch {batch_start} {pass_name}-pass -- backing off 65s "
                              f"(could be the per-minute cap, not the daily one)")
                        time.sleep(65)
                        try:
                            raw_items = call_gemini_on_pages(genai_model, batch,
                                                             context=context_str, prompt=prompt)
                            state["calls_today"] += 1
                            save_state(state)
                        except Exception as e2:
                            t2 = str(e2)
                            if "429" in t2 or "quota" in t2.lower():
                                print(f"  [QUOTA] still limited after 65s backoff -- daily cap "
                                      f"it is. Saving progress, exiting: {e2}")
                                save_state(state)
                                sys.exit(0)
                            print(f"  [WARN] post-backoff call failed differently "
                                  f"({pass_name}-pass): {e2}")
                            raw_items = retry_batch_page_by_page(
                                genai_model, pass_batch, state,
                                ctx={"subject": subject, "chapter_no": ch["chapter_no"],
                                     "chapter_id": chapter_id, "pass": pass_name},
                                prompt=prompt)
                            if not raw_items:
                                continue
                    elif "Invalid Gemini JSON" in err_text or "empty JSON response" in err_text:
                        # A malformed structured response is a model output
                        # glitch, not a page problem (run-5: 2 A-pass batches
                        # burned 6 single-page calls each before recovering).
                        # Re-ask the SAME batch once (1 call); only if that
                        # also fails, descend to page-by-page salvage.
                        print(f"  [WARN] malformed JSON on {pass_name}-pass pages "
                              f"{window_pages} -- one same-batch re-ask before "
                              f"page-by-page salvage")
                        try:
                            raw_items = call_gemini_on_pages(genai_model, pass_batch,
                                                             context=context_str,
                                                             prompt=prompt)
                            state["calls_today"] += 1
                            save_state(state)
                        except Exception as e2:
                            print(f"  [WARN] same-batch re-ask failed ({e2}) "
                                  f"-- page-by-page salvage")
                            raw_items = retry_batch_page_by_page(
                                genai_model, pass_batch, state,
                                ctx={"subject": subject, "chapter_no": ch["chapter_no"],
                                     "chapter_id": chapter_id, "pass": pass_name},
                                prompt=prompt)
                            if not raw_items:
                                continue
                    else:
                        print(f"  [WARN] Gemini {pass_name}-pass failed on {subject} "
                              f"ch{ch['chapter_no']} batch {batch_start}: {e}")
                        # don't lose the whole batch over one bad page
                        raw_items = retry_batch_page_by_page(
                            genai_model, pass_batch, state,
                            ctx={"subject": subject, "chapter_no": ch["chapter_no"],
                                 "chapter_id": chapter_id, "pass": pass_name},
                            prompt=prompt)
                        if not raw_items:
                            continue

                if pass_name == "S":
                    raw_items, n_clip = clip_pass_solutions(raw_items)
                    if n_clip:
                        print(f"  [S-CLIP] {n_clip} foreign 'Solution to Question N:' "
                              f"tail(s) clipped in S-pass output (sibling item "
                              f"present -- provably zero-loss)")
                items, batch_meta = extract_batch_meta(raw_items)
                # provenance of every normal-pass item (run-7 hardening #4):
                # used by merge to enforce patch-only recovery and to reject
                # contamination (an S/A item's stray stem is never merged).
                for it in items:
                    if isinstance(it, dict):
                        it["_prov"] = f"{pass_name}_PASS"
                if batch_meta.get("figure_map"):
                    # Q-pass sees question-side figures, S-pass solution-side;
                    # keep the first non-empty map per pass for this window.
                    fig_map_by_pass[pass_name] = batch_meta["figure_map"]
                chapter_records, skipped = merge_question_records(chapter_records, items, stats)
                try:
                    last_qn_in_batch = max(int(it.get("q_no")) for it in items
                                           if it.get("q_no") is not None)
                except (ValueError, TypeError):
                    last_qn_in_batch = None
                for it in skipped:
                    # RC-2 salvage buffer: fragments (usually batch-boundary
                    # continuations) carry real content -- keep with provenance
                    # (+ which pass produced them) for chapter-end recovery.
                    orphans.append({
                        "chapter_id": chapter_id, "batch_start": batch_start,
                        "pdf_pages": window_pages, "new_pages": new_pages,
                        "carry_q_no": carry_in["last_open_question"] if carry_in else None,
                        "cut_part": carry_in.get("cut_part") if carry_in else None,
                        "last_qn_in_batch": last_qn_in_batch,
                        "pass": pass_name,
                        "item": it,
                    })
                stats["orphans_buffered"] += len(skipped)
                if pass_name in ("Q", "S"):
                    # per-pass carry-forward (Feature 2), then stale-carry guard #1
                    carry_by_pass[pass_name] = compute_carry(
                        batch_meta, items, chapter_records, max(window_pages))
                    carry_by_pass[pass_name] = enforce_carry_expiry(
                        carry_by_pass[pass_name], stats["batches"],
                        carry_trackers[pass_name], carry_banned[pass_name],
                        chapter_records, chapter_id)
                # stale-carry guard #2: questions->solutions SECTION boundary.
                # The first batch that shows the solutions section hard-resets
                # ALL carry context -- resolved or not -- so it can NEVER bleed
                # into the Solutions section and cross-merge there. Q-pass also
                # turns OFF from the next batch (probe may still enable S/A).
                if pass_name == "Q" and not solutions_section_seen:
                    boundary = detect_section_boundary(items)
                    if boundary:
                        solutions_section_seen = True
                        had_pending = any(v is not None for v in carry_by_pass.values()) \
                            or any(carry_trackers.values())
                        carry_by_pass = {"Q": None, "S": None}
                        carry_trackers = {"Q": {}, "S": {}}
                        print(f"  [SECTION] {boundary} first seen at pages "
                              f"{window_pages[0]}-{window_pages[-1]} -- solutions "
                              f"section begins; ALL carry context HARD-RESET"
                              f"{' (dropped pending context)' if had_pending else ''}; "
                              f"Q-pass disabled from the next batch")
                carry_obj = carry_by_pass.get(pass_name)
                last_open = (f"q{carry_obj['last_open_question']}"
                             if carry_obj and carry_obj["last_open_question"] is not None
                             else ("open (no number)" if carry_obj else "-"))
                print(f"  [GEMINI:{pass_name}] pages {window_pages[0]}-{window_pages[-1]}"
                      f" | overlap: {overlap_pages if overlap_pages else '-'}"
                      f" | carry-in: {('q' + str(carry_in['last_open_question'])) if carry_in and carry_in['last_open_question'] is not None else '-'}"
                      f" | last-open: {last_open}"
                      f" | items: {len(items)} | orphans buffered: {len(skipped)}")

            prev_window_last_page = max(window_pages)
            prev_section = section

            # extract real (non-watermark) images from this batch's pages.
            # pdftoppm names output files using the ACTUAL pdf page number
            # (e.g. page-005.jpg for real page 5) -- read it directly from
            # the filename, don't recompute it relative to ch["file_start"].
            # First collect every page's images (top-to-bottom order) for the
            # window, THEN claim: figure-map first, positional after.
            window_rows = []
            for pf in batch:
                file_page_num = int(pf.stem.split("-")[-1])
                if file_page_num in pages_imaged:
                    continue  # overlap page -- images already extracted once
                pages_imaged.add(file_page_num)
                imgs = extract_real_images(pdf_path, file_page_num, watermark_id, subject, ASSETS_DIR / "questions")
                if not imgs:
                    continue
                pos = image_positions_on_page(pdf_path, file_page_num)
                ordered = _order_imgs_by_position(imgs, pos)
                window_rows.append((file_page_num, ordered))

            # FIGURE-MAP pass (run-6 user ask: "bta ye image kis question ki
            # h") -- Gemini's own _figure_map declares q_no+slot per figure in
            # reading order. Claim those first (exact-count guard inside: any
            # mismatch skips safely). This is what stops images from sitting
            # "unclaimed": the model that READ the page tells us the owner.
            window_fig_map = fig_map_by_pass.get("Q") or fig_map_by_pass.get("S") or None
            fig_leftover = claim_figure_map_images(
                window_fig_map, window_rows, subject, ch["chapter_no"],
                chapter_records, image_files_by_q)

            for file_page_num, _rels in window_rows:
                leftover = fig_leftover.get(file_page_num) or []
                if not leftover:
                    continue
                # A figure between "Solution to Question N" and the next
                # header belongs to THAT solution, not to whichever question
                # happens to be pending in reading order. Match by POSITION
                # (each image goes to the header drawn above it), never by a
                # single text-layer hit: when the text layer decodes only ONE
                # of several headers, the old shortcut dumped every figure of
                # the page onto that one solution (user report: 7 figures
                # collapsed into 2 solutions). Images with no locatable header
                # above them are left for the later passes.
                leftover = claim_page_images(leftover, pdf_path, file_page_num, subject,
                                             ch["chapter_no"], chapter_records, image_files_by_q)
                if leftover:
                    unmatched_images.append({"page": file_page_num, "files": leftover})
                    print(f"  [INFO] Page {file_page_num}: image(s) {leftover} unclaimed for now "
                          f"-- will retry after all batches (owner may be in a later batch)")

        # FEATURE 3 -- orphan recovery runs BEFORE image claiming and JSON
        # writing: recovered fragments can complete solutions/options, and
        # only genuinely ownerless orphans are persisted (after the drain's
        # second recovery -- persisting early wrote "unresolved" entries for
        # fragments the drain later healed).
        orphans = recover_orphans(orphans, chapter_records, subject, ch["chapter_no"], stats)
        stats["orphans_remaining"] = len(orphans)

        # SECOND PASS image claiming: a figure can be extracted BEFORE the
        # batch that introduces its owning question (plate printed just before
        # the question text, or owner arrived via an overlap window). Chapter
        # records are complete now -- retry every leftover once.
        n_unmatched = 0
        for um in unmatched_images:
            # Full two-stage claimer again: chapter records are now complete,
            # so a solution header whose q_no was missing at first-pass time
            # becomes usable for position mapping; leftovers still fall to the
            # one-to-one matcher.
            leftover2 = claim_page_images(um["files"], pdf_path, um["page"],
                                          subject, ch["chapter_no"],
                                          chapter_records, image_files_by_q)
            um["files"] = leftover2
            if not leftover2:
                print(f"  [INFO] second pass: page {um['page']} image(s) matched to a question")
                um["matched"] = True
                continue
            # THIRD pass (0 tokens): Gemini never set has_figure flags, so
            # the flag-only matcher can never fire (run-2: p127/128/188/295/
            # 318). Read the page's printed question numbers via pdftotext --
            # if EXACTLY ONE of this chapter's questions lives on the image's
            # page, that question is the owner with high confidence. Zero or
            # multiple candidates stay unmatched (evidence insufficient).
            try:
                qns = qns_printed_on_page(pdf_path, um["page"], chapter_records)
            except Exception as e:
                print(f"  [WARN] third-pass pdftotext failed for page {um['page']}: {e}")
                qns = []
            if len(qns) == 1:
                qn = qns[0]
                rec = chapter_records[qn]
                entry = image_files_by_q.setdefault(qn, {"question": [], "solution": []})
                qt, st = (rec.get("question_text") or "").lower(), (rec.get("solution_text") or "").lower()
                side = "question" if ("fig" in qt or "diagram" in qt or not st) else "solution"
                if not entry[side]:
                    qid = f"{subject}-{ch['chapter_no']:03d}-{qn:03d}"
                    # _rename_for_slot: collision-proof suffixing (two pages can
                    # print the same q_no across a page break), the
                    # MAX_QUESTION_IMAGES cap, and the tiny-crop guard -- the old
                    # hand-rolled rename here bypassed all three (overwrite risk).
                    consumed = []  # (old_rel, new_rel)
                    for old_rel in list(um["files"]):
                        new_rel = _rename_for_slot(old_rel, qn, side, subject,
                                                   ch["chapter_no"], image_files_by_q)
                        if new_rel:
                            consumed.append((old_rel, new_rel))
                    if consumed:
                        entry[side].extend(nr for _, nr in consumed)
                        done = {o for o, _ in consumed}
                        um["files"] = [f for f in um["files"] if f not in done]
                        um["matched"] = not um["files"]
                        if um["matched"]:
                            print(f"  [INFO] third pass: page {um['page']} image(s) attached to {qid} "
                                  f"(sole printed question on that page, {side} side)")
                        else:
                            print(f"  [INFO] third pass: page {um['page']}: {len(consumed)} image(s) "
                                  f"attached to {qid}; rest refused by guards -- left for review")
                    else:
                        print(f"  [WARN] third pass: rename failed for page {um['page']} "
                              f"-- left unmatched (file already moved earlier?)")
            elif qns:
                print(f"  [INFO] third pass: page {um['page']} has {len(qns)} printed questions "
                      f"{qns} -- ambiguous owner, left for manual review")
        # FOURTH pass (Gemini, ONE image per call, never grouped): the final
        # safety net (Gap-2). The model attributes each leftover image to a
        # printed q_no (with a question/solution slot), or confidently marks
        # it decorative. Only files the model could not decide on (call
        # failure / null verdict / number not in this chapter) -- or files
        # left when the daily quota brake fires -- stay unmatched.
        for um in unmatched_images:
            if um.get("matched"):
                continue
            still, brake_hit, verdicts = [], False, {}
            for rel in um["files"]:
                if brake_hit:
                    still.append(rel)
                    continue
                verdict = attribute_orphan_image(genai_model, rel, chapter_records, state)
                if verdict and verdict.get("decorative") == "brake":
                    brake_hit = True
                    still.append(rel)
                    continue
                if not verdict:
                    still.append(rel)   # undecided / call failed -> manual review
                    continue
                if verdict.get("decorative") is True:
                    print(f"  [IMG] fourth pass: page {um['page']} {rel} is decorative/unrelated "
                          f"(model-confirmed) -- logged to decorative_images.jsonl")
                    _append_jsonl(DATA_DIR / "decorative_images.jsonl",
                                  {"subject": subject, "chapter_id": chapter_id,
                                   "page": um["page"], "file": rel,
                                   "reason": "model-confirmed decorative/unrelated"})
                    continue
                qn_attr = verdict.get("q_no")
                if isinstance(qn_attr, bool) or not isinstance(qn_attr, int) \
                        or qn_attr not in chapter_records:
                    still.append(rel)   # weak/no match the model wouldn't stand behind
                    verdicts[rel] = verdict
                    continue
                slot = verdict.get("slot")
                if slot not in ("question", "solution"):
                    slot = "question"
                new_rel = _rename_for_slot(rel, qn_attr, slot, subject, ch["chapter_no"],
                                           image_files_by_q)
                if new_rel:
                    image_files_by_q.setdefault(qn_attr, {"question": [], "solution": []})[slot].append(new_rel)
                    qid = f"{subject}-{ch['chapter_no']:03d}-{qn_attr:03d}"
                    print(f"  [IMG] fourth pass: page {um['page']} {rel} -> {qid} "
                          f"({slot} side, model-attributed)")
                else:
                    # model DECLARED the owner but a guard (tiny-crop /
                    # over-attribution cap) refused the rename -- keep the
                    # verdict visible so nothing is silently unclaimed.
                    verdicts[rel] = verdict
                    print(f"  [IMG] fourth pass: page {um['page']} {rel} -> q{qn_attr} "
                          f"({slot}) DECLARED by model but guard refused rename "
                          f"-- verdict recorded, left for review")
                    still.append(rel)
            um["files"] = still
            um["model_verdicts"] = verdicts
            if not still:
                um["matched"] = True
            elif brake_hit:
                print(f"  [WARN] image attribution stopped early (daily quota) -- "
                      f"{len(still)} file(s) from page {um['page']} stay queued")
        for um in unmatched_images:
            if not um.get("matched"):
                n_unmatched += 1
                print(f"  [WARN] Page {um['page']}: extracted image(s) {um['files']} but no "
                      f"question/solution in this chapter claimed one -- left under its temp "
                      f"filename for manual review (see data/unmatched_images.jsonl).")
                entry = {"subject": subject, "chapter_id": chapter_id,
                         "page": um["page"], "files": um["files"]}
                if um.get("model_verdicts"):
                    entry["model_verdicts"] = um["model_verdicts"]  # model's q_no/slot answers
                _append_jsonl(DATA_DIR / "unmatched_images.jsonl", entry)

        # FAILED-PAGE DRAIN: second chance for recitation-skipped pages
        # BEFORE orphan recovery (drained fragments may join the orphan
        # pool) and BEFORE targeted retry (so drained solutions count when
        # the 60% book-prints-solutions gate is evaluated).
        pending_failed = [e for e in state.get("failed_pages", [])
                          if e.get("chapter_id") == chapter_id]
        if pending_failed:
            chapter_records, drain_orphans, healed = drain_failed_pages(
                genai_model, pending_failed, page_dir, chapter_records, state, stats,
                pdf_path=pdf_path)
            orphans.extend(drain_orphans)
            orphans = recover_orphans(orphans, chapter_records, subject, ch["chapter_no"], stats)
            stats["orphans_remaining"] = len(orphans)
            if healed:
                healed_ids = {(e["subject"], e["chapter_no"], e["true_page"]) for e in healed}
                state["failed_pages"] = [e for e in state.get("failed_pages", [])
                                         if (e.get("subject"), e.get("chapter_no"), e.get("true_page"))
                                         not in healed_ids]
                save_state(state)
            print(f"  [DRAIN] second chance: {len(healed)}/{len(pending_failed)} previously-failed page(s) recovered")

        # persist only the FINAL unresolved orphans (after the drain's second
        # recovery pass) -- never ledger entries the drain later healed.
        for orph in orphans:
            _append_jsonl(DATA_DIR / "orphans.jsonl", orph)

        # INTEGRITY SWEEP: zero-token deterministic proofs (run-4 audit RCA:
        # duplicated wrong-owner stems, foreign 'Option' heads, truncated
        # solutions, over-attributed images) BEFORE targeted retry, so
        # stripped/provably-incomplete fields are re-asked in the SAME run.
        forced_solution_qns = chapter_integrity_sweep(
            chapter_records, image_files_by_q, subject, ch["chapter_no"], stats)

        # PRINTED-SOLUTION EVIDENCE (ch25 class): a q_no whose "Solution to
        # Question N:" header exists in the chapter's text layer PROVES the
        # book prints an explanation for it. When the chapter sits below the
        # 60% solution-gate, scan the pages ONCE (zero-token) and bypass the
        # gate for exactly those q_nos -- the run-5 audit showed the gate
        # suppressing 5 REAL solutions in ch25 (7/12 = 58%).
        n_with_sol = sum(1 for r in chapter_records.values()
                         if (r.get("solution_text") or "").strip())
        gate_marginal = (
            chapter_records
            and n_with_sol / len(chapter_records) < SOLUTION_GATE_MIN_SHARE
            and any(r.get("question_text") and not (r.get("solution_text") or "").strip()
                    for r in chapter_records.values())
        )
        printed_sol_qns = chapter_printed_solution_qns(
            pdf_path, page_files, chapter_records) if gate_marginal else set()
        if printed_sol_qns:
            print(f"  [GATE] chapter {ch['chapter_no']} below the {SOLUTION_GATE_MIN_SHARE:.0%} "
                  f"solution gate but {len(printed_sol_qns)} printed 'Solution to Question N:' "
                  f"header(s) found ({sorted(printed_sol_qns)}) -- retry eligible for those")

        # FEATURE: targeted gap-retry -- AFTER normal batches + orphan
        # recovery, BEFORE writing the chapter's questions to disk.
        n_fixed = targeted_retry(genai_model, page_files, chapter_records,
                                 state, max_rounds=TARGETED_RETRY_MAX_ROUNDS,
                                 force_solution_qns=forced_solution_qns,
                                 chapter_id=chapter_id,
                                 printed_solution_qns=printed_sol_qns,
                                 stats=stats)
        if n_fixed:
            print(f"  [RETRY] closed {n_fixed} field(s) via targeted retry")

        # RESCUE PASS (run-5 audit): records still incomplete after the
        # whole-chapter retry get one page-focused last ditch -- re-ask ONLY
        # the pages where their q_no is printed, instead of the whole chapter
        # again. This is what the 9 persistent gaps (ch2 q25/26, ch18 q13,
        # ch19 q11/12, ch24 q12/13, ch27 q11, ch33 q9) needed.
        n_rescued = rescue_incomplete_records(
            genai_model, page_files, pdf_path, chapter_records, state, stats,
            chapter_id, printed_solution_qns=printed_sol_qns)
        if n_rescued:
            print(f"  [RESCUE] closed {n_rescued} field(s) via page-focused rescue")

        # ANCHORLESS RECORDS (ch24 q12/13 class): rows with NO stem, NO
        # options and NO solution after batch + 2 retry rounds + rescue are
        # phantom answer-key rows (a printed key table spanning chapters) or
        # fully-lost fragments -- shipping them pollutes the app with empty
        # questions. Drop with a ledger entry; nothing is silently lost.
        dropped_anchorless = []
        kept_records = {}
        for qn, rec in sorted(chapter_records.items(), key=lambda x: x[0]):
            if not ((rec.get("question_text") or "").strip()
                    or (rec.get("options") or {})
                    or (rec.get("solution_text") or "").strip()):
                dropped_anchorless.append(qn)
                _append_jsonl(DATA_DIR / "dropped_anchorless.jsonl",
                              {"chapter_id": chapter_id, "q_no": qn,
                               "correct_option": rec.get("correct_option"),
                               "reason": "no stem/options/solution after "
                                         "batch+retry+rescue (phantom key row "
                                         "or fully-lost fragment)"})
                continue
            kept_records[qn] = rec
        if dropped_anchorless:
            print(f"  [DROP] {len(dropped_anchorless)} anchorless record(s) removed "
                  f"(q{sorted(dropped_anchorless)}) -- logged to "
                  f"data/dropped_anchorless.jsonl")
            chapter_records = kept_records
            stats["anchorless_dropped"] = stats.get("anchorless_dropped", 0) + len(dropped_anchorless)

        chapter_rows = []
        for qn, rec in sorted(chapter_records.items(), key=lambda x: x[0]):
            final_q = build_final_question(
                subject, chapter_id, ch["chapter_no"], qn, rec,
                image_files_by_q.get(qn, {"question": [], "solution": []})
            )
            final_q = repair_option_labels(final_q)
            questions_fh.write(json.dumps(final_q, ensure_ascii=False) + "\n")
            questions_fh.flush()
            chapter_rows.append(final_q)
        # per-chapter file: written only NOW, when this chapter has FULLY
        # finished every process -- the batch loop of the NEXT chapter has
        # not started yet.
        write_chapter_file(subject, chapter_id, chapter_rows)

        progress["chapters_done"].append(chapter_id)
        save_state(state)
        # Persist chapters.json incrementally too. main() also writes it at the
        # end, but if we exit early (daily Gemini limit -> sys.exit, crash,
        # redeploy) that final write never happens -- and since completed
        # chapters are in chapters_done, the next run would skip them and
        # they'd be permanently missing from chapters.json.
        chapters_path = DATA_DIR / "chapters.json"
        write_chapters(chapters_path, chapters_out)
        n_no_answer = sum(1 for r in chapter_records.values() if not r.get("correct_option"))
        n_no_solution = sum(1 for r in chapter_records.values() if not r.get("solution_text"))
        print(f"[{subject}] chapter {ch['chapter_no']} ({ch['chapter_title']}) done -> "
              f"{len(chapter_records)} questions ({n_no_answer} missing answer, {n_no_solution} missing solution)")
        if n_no_solution and chapter_records:
            coverage = 1 - n_no_solution / len(chapter_records)
            if coverage >= SOLUTION_GATE_MIN_SHARE:
                print(f"  [WARN] {n_no_solution} solution(s) still missing although this chapter "
                      f"prints explanations ({coverage:.0%} coverage) -- extraction loss, "
                      f"see data/still_incomplete_after_retry.jsonl; re-run or --recover these pages")
        print(f"[{subject}]   batches: {stats['batches']} | duplicates merged: {stats['duplicates_merged']}"
              f" | conflicts dropped: {stats['conflicts']} | carry-forward used: {stats['carry_used']}"
              f" | carry merges: {stats['carry_merges']}"
              f" | orphans: {stats['orphans_recovered']} recovered, {stats['orphans_remaining']} unresolved"
              f" | unmatched images: {n_unmatched}"
              f" | rescue: {stats.get('rescue_filled', 0)} filled / {stats.get('rescue_calls', 0)} calls"
              f" | anchorless dropped: {stats.get('anchorless_dropped', 0)}")

    # ALL chapters of this subject are complete now -> bundle everything into
    # a subject-named folder (per-chapter files were written as each chapter
    # closed; earlier run's files persist on the volume, so a resumed run
    # still produces the full bundle here).
    build_subject_bundle(subject, chapters_out)

# ============================================================
# TARGETED RECOVERY MODE (--recover plan.json)
# Heals already-written questions.jsonl rows WITHOUT reprocessing whole
# chapters and WITHOUT touching state.json / chapters_done.
# plan.json shape:
#   {"PSY-016": {"pages": [214, 217], "reason": "recitation batch loss"},
#    "PSY-001": {"pages": [17],      "reason": "missing solution for q13"}}
# Pages are TRUE PDF file page numbers (same numbering used by
# orphans.jsonl / unmatched_images.jsonl / temp image filenames).
# ============================================================

def final_q_to_record(q):
    """Reverse build_final_question: folded an existing JSONL row back into a
    merge-ready record (+ its already-owned images for re-emission)."""
    options = {o["id"]: o["text"] for o in (q.get("options") or [])} or None
    correct = q.get("correct_options") or []
    rec = {
        "q_no": int(q["id"].rsplit("-", 1)[-1]),
        "question_text": q["question"]["text"],
        "options": options,
        "correct_option": correct[0] if correct else None,
        "solution_text": q["solution"]["text"],
        "tables": [{"type": t.get("type", "table"), "markdown": t["markdown"]}
                   for t in q["solution"].get("tables", [])],
        "has_figure_in_question": bool(q["question"]["images"]),
        "has_figure_in_solution": bool(q["solution"]["images"]),
        "_prov": {},   # provenance resets on re-import; new merges re-tag
    }
    owned = {"question": [i["file"] for i in q["question"]["images"]],
             "solution": [i["file"] for i in q["solution"]["images"]]}
    return rec, owned

RECITATION_RECOVERY_CONTEXT = (
    "RECITATION-SAFE RECOVERY: describe the visible educational content in your own words. "
    "Do not quote or transcribe long passages verbatim; preserve question numbers, answer letters, "
    "and the meaning of explanations. Return the normal JSON schema. "
)

RECOVERY_CONTEXT = (
    "RECOVERY NOTE: these are SELECTED pages from a single chapter, sent to "
    "fill specific extraction gaps. Pages may be non-adjacent and each page "
    "may begin or end mid-flow. Extract everything visible exactly as usual; "
    "if a fragment at a page edge has no visible question number, return it "
    'with "q_no": null as usual. Never invent numbers.'
)

def recover_pages(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(GEMINI_MODEL)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()   # real state: quota tracking + failed_pages queue

    questions_path = DATA_DIR / "questions.jsonl"
    all_lines = [json.loads(l) for l in
                 questions_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    for chapter_id, spec in plan.items():
        pages = sorted(set(int(p) for p in spec["pages"]))
        subject, chap_str = chapter_id.split("-", 1)
        chapter_no = int(chap_str)
        pdf_cfg = next((c for c in PDFS if c["subject"] == subject), None)
        if not pdf_cfg:
            print(f"[RECOVER] no PDF configured for {subject} -- skipping {chapter_id}")
            continue
        pdf_path = pdf_cfg["path"]
        total_pages = len(PdfReader(pdf_path).pages)
        watermark_id = find_watermark_object_id(pdf_path)

        # rebuild existing chapter rows so recovery MERGES into them
        chapter_lines = [q for q in all_lines if q.get("chapter_id") == chapter_id]
        records, image_files_by_q = {}, {}
        for q in chapter_lines:
            rec, owned = final_q_to_record(q)
            records[rec["q_no"]] = rec
            image_files_by_q[rec["q_no"]] = owned
        print(f"[RECOVER] {chapter_id}: {len(records)} existing rows; "
              f"target pages {pages} ({spec.get('reason', 'no reason given')})")

        # render targets +/- 1 neighbour (continuation context!) at higher DPI
        neighbour = sorted({p for t in pages for p in (t - 1, t, t + 1)
                            if 1 <= p <= total_pages})
        rec_dir = Path(f"/tmp/{subject}_recover_{chapter_no:03d}")
        rec_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["pdftoppm", "-jpeg", "-r", "200",
                        "-f", str(neighbour[0]), "-l", str(neighbour[-1]),
                        pdf_path, str(rec_dir / "page")])
        page_files = sorted(rec_dir.glob("page-*.jpg"))
        pages_imaged = set()
        stats = {"duplicates_merged": 0, "conflicts": 0,
                 "carry_merges": 0, "orphans_recovered": 0,
                 "chapter_id": f"{subject}-{chapter_no:03d}"}
        orphans = []
        unmatched_images = []

        for win_start in range(0, len(page_files), PAGES_PER_GEMINI_CALL):
            batch = page_files[win_start:win_start + PAGES_PER_GEMINI_CALL]
            window_pages = [int(p.stem.split("-")[-1]) for p in batch]
            reset_daily_counter_if_needed(state)
            if state["calls_today"] >= MAX_CALLS_PER_DAY:
                print("Daily Gemini call limit reached during recovery. Saving, exiting.")
                save_state(state)
                sys.exit(0)
            try:
                raw = call_gemini_on_pages(model, batch, context=RECOVERY_CONTEXT)
                # quota accounting: recovery's direct calls used to bypass
                # calls_today (only the fallback retries counted) -- a long
                # recovery could overshoot the per-day cap blind.
                state["calls_today"] += 1
                save_state(state)
            except Exception as e:
                print(f"  [WARN] recovery call failed for {chapter_id} pages "
                      f"{window_pages}: {e}")
                raw = retry_batch_page_by_page(model, batch, state,
                                               ctx={"subject": subject,
                                                    "chapter_no": chapter_no,
                                                    "chapter_id": chapter_id})
                if not raw:
                    continue
            items, _meta = extract_batch_meta(raw)
            # recovery items are provenance-tagged so merge applies the
            # semantic stem guard to anything that looks like solution prose
            # (run-7 hardening #4).
            for it in items:
                if isinstance(it, dict):
                    it["_prov"] = "RECOVER"
            records, skipped = merge_question_records(records, items, stats, fill_only=True)
            for it in skipped:
                orphans.append({"chapter_id": chapter_id, "batch_start": win_start,
                                "pdf_pages": window_pages, "new_pages": window_pages,
                                "carry_q_no": None, "cut_part": None,
                                "last_qn_in_batch": None, "item": it})
            for pf in batch:
                file_page_num = int(pf.stem.split("-")[-1])
                if file_page_num in pages_imaged:
                    continue
                pages_imaged.add(file_page_num)
                imgs = extract_real_images(pdf_path, file_page_num, watermark_id,
                                           subject, ASSETS_DIR / "questions")
                if imgs:
                    rec_leftover = claim_page_images(imgs, pdf_path, file_page_num,
                                                     subject, chapter_no,
                                                     records, image_files_by_q)
                    if rec_leftover:
                        unmatched_images.append({"page": file_page_num, "files": rec_leftover})

        # drain this chapter's queued failed pages (incl. ones recovery itself
        # just queued): crop-ladder second chance, fragments join the orphan
        # pool BEFORE recover_orphans runs -- same order as the production path.
        pending_failed = [e for e in state.get("failed_pages", [])
                          if e.get("chapter_id") == chapter_id]
        if pending_failed:
            records, rec_drain_orphans, healed = drain_failed_pages(
                model, pending_failed, rec_dir, records, state, stats,
                pdf_path=pdf_path)
            orphans.extend(rec_drain_orphans)
            if healed:
                healed_ids = {(e["subject"], e["chapter_no"], e["true_page"]) for e in healed}
                state["failed_pages"] = [e for e in state.get("failed_pages", [])
                                         if (e.get("subject"), e.get("chapter_no"), e.get("true_page"))
                                         not in healed_ids]
                save_state(state)

        orphans = recover_orphans(orphans, records, subject, chapter_no, stats)
        for orph in orphans:
            _append_jsonl(DATA_DIR / "orphans.jsonl", orph)

        # Close the healing loop: without this, a recovery could never fix a
        # TRUNCATED solution (merges are fill-only, and production's sweep-
        # forced re-ask lives only in process_pdf). Detection only -- the
        # sweep's destructive parts are skipped here because the recovery
        # page window is narrower than a full chapter pass.
        forced = {qn for qn, r in records.items()
                  if (r.get("solution_text") or "").strip()
                  and looks_truncated_solution(r["solution_text"],
                                               has_tables=bool(r.get("tables")))}
        targeted_retry(model, page_files, records, state,
                       force_solution_qns=forced, chapter_id=chapter_id,
                       stats=stats)
        for um in unmatched_images:
            rec_leftover = claim_page_images(um["files"], pdf_path, um["page"],
                                             subject, chapter_no,
                                             records, image_files_by_q)
            if rec_leftover:
                _append_jsonl(DATA_DIR / "unmatched_images.jsonl",
                              {"subject": subject, "chapter_id": chapter_id,
                               "page": um["page"], "files": rec_leftover})

        # rewrite questions.jsonl: keep other chapters' rows, replace this one
        others = [q for q in all_lines if q.get("chapter_id") != chapter_id]
        emitted = [build_final_question(subject, chapter_id, chapter_no, qn, rec,
                                        image_files_by_q.get(qn, {"question": [], "solution": []}))
                   for qn, rec in sorted(records.items())]
        out_ids = [q["id"] for q in emitted]
        assert len(out_ids) == len(set(out_ids)), "duplicate ids after recovery"
        with open(questions_path, "w", encoding="utf-8") as fh:
            for q in others + emitted:
                fh.write(json.dumps(q, ensure_ascii=False) + "\n")
        all_lines = others + emitted  # next chapter's rebuild sees fresh rows

        n_no_solution = sum(1 for r in records.values() if not r.get("solution_text"))
        n_no_answer = sum(1 for r in records.values() if not r.get("correct_option"))
        print(f"[RECOVER] {chapter_id} done -> {len(records)} questions "
              f"({n_no_answer} missing answer, {n_no_solution} missing solution)"
              f" | conflicts dropped: {stats['conflicts']}"
              f" | orphans unresolved: {len(orphans)}")

    print("[RECOVER] all planned chapters processed.")

def _dedupe_questions_by_id(path):
    """questions.jsonl is append-only, and surgically re-running a chapter
    (removing its id from chapters_done) appends its rows AGAIN. At 20-book
    scale that accumulates duplicate ids the app renders twice. Rewrite the
    file keeping the LAST row per id (the newest extraction wins). Returns
    the number of duplicate rows removed."""
    if not path.exists():
        return 0
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rows = {}
    n_dups = 0
    for ln in lines:
        try:
            rid = json.loads(ln).get("id")
        except json.JSONDecodeError:
            continue
        if rid in rows:
            n_dups += 1
        rows[rid] = ln  # last wins (newest extraction)
    if n_dups:
        path.write_text("".join(rows[rid] + "\n" for rid in rows), encoding="utf-8")
    return n_dups


def build_auto_recovery_plan():
    """Assemble a recover_pages plan from the run's ledgers, so a whole
    book (or 20) can be healed with ONE command instead of hand-writing
    plan.json:
      * still_incomplete_after_retry.jsonl -- every gap's pages are located
        in the PDF text layer (question stem and/or 'Solution to Question
        N:' header);
      * orphans.jsonl -- their pdf_pages;
      * unmatched_images.jsonl -- their pages.
    Returns {chapter_id: {"pages": [...], "reason": "..."}}."""
    plan = {}
    pdf_by_subject = {c["subject"]: c for c in PDFS}

    def add(chapter_id, pages, reason):
        if not pages:
            return
        entry = plan.setdefault(chapter_id, {"pages": [], "reasons": []})
        entry["pages"] = sorted(set(entry["pages"] + [int(p) for p in pages]))
        entry["reasons"].append(reason)

    def pages_for_questions(subject, chapter_no, qns):
        cfg = pdf_by_subject.get(subject)
        if not cfg or not qns:
            return []
        total = len(PdfReader(cfg["path"]).pages)
        chs = compute_page_ranges(extract_toc_chapters(cfg["path"]),
                                  cfg["page_offset"], total)
        ch = next((c for c in chs if c["chapter_no"] == chapter_no), None)
        if not ch:
            return []
        found = []
        qns = set(qns)
        header_re = re.compile(r"Solution\s+to\s+Question\s+(\d{1,3})", re.IGNORECASE)
        for page_no in range(ch["file_start"], ch["file_end"] + 1):
            text = pdftotext_page(cfg["path"], page_no)
            if not text.strip():
                continue
            if any(re.search(r"(?m)^\s*(?:Q(?:uestion)?\s*[.:]?\s*)?%d\s*[.)]" % qn, text)
                   for qn in qns):
                found.append(page_no)
                continue
            if any(int(m.group(1)) in qns for m in header_re.finditer(text)):
                found.append(page_no)
        return found

    # 1) still-incomplete records
    inc_path = DATA_DIR / "still_incomplete_after_retry.jsonl"
    if inc_path.exists():
        by_chapter = {}
        for ln in inc_path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                e = json.loads(ln)
            except json.JSONDecodeError:
                continue
            cid, qn = e.get("chapter_id"), e.get("q_no")
            if not cid or qn is None:
                continue
            by_chapter.setdefault(cid, set()).add(int(qn))
        for cid, qns in by_chapter.items():
            try:
                subject, chap_str = cid.split("-", 1)
                chapter_no = int(chap_str)
            except (ValueError, TypeError):
                continue
            pages = pages_for_questions(subject, chapter_no, qns)
            add(cid, pages, f"still incomplete q{','.join(map(str, sorted(qns)))} "
                            f"({len(pages)} page(s) located)")
    # 2) unresolved orphans -> their pdf_pages
    orph_path = DATA_DIR / "orphans.jsonl"
    if orph_path.exists():
        for ln in orph_path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                e = json.loads(ln)
            except json.JSONDecodeError:
                continue
            cid = e.get("chapter_id")
            pages = e.get("pdf_pages") or e.get("new_pages") or []
            if cid and pages:
                add(cid, pages, "unresolved orphan fragment")
    # 3) unmatched images -> their pages
    um_path = DATA_DIR / "unmatched_images.jsonl"
    if um_path.exists():
        for ln in um_path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                e = json.loads(ln)
            except json.JSONDecodeError:
                continue
            cid, page = e.get("chapter_id"), e.get("page")
            if cid and page:
                add(cid, [page], "unclaimed image(s)")
    return {cid: {"pages": e["pages"], "reason": "; ".join(e["reasons"])}
            for cid, e in plan.items()}


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

    write_chapters(chapters_path, chapters_out)
    save_state(state)
    # surgical re-runs append duplicate rows; keep the newest per id.
    n_dups = _dedupe_questions_by_id(questions_path)
    if n_dups:
        print(f"Deduplicated {n_dups} stale duplicate row(s) from questions.jsonl "
              f"(newest extraction kept).")
    print("All done (or paused at daily limit -- just re-run this script to resume).")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--recover":
        # targeted healing of already-written rows, e.g.:
        #   python3 qbank_pipeline.py --recover recovery_plan.json
        recover_pages(sys.argv[2] if len(sys.argv) > 2 else "recovery_plan.json")
    elif len(sys.argv) > 1 and sys.argv[1] == "--auto-recover":
        # one-command whole-book heal: build the plan from the run ledgers
        # (still-incomplete, orphans, unmatched images) and run it.
        plan = build_auto_recovery_plan()
        if not plan:
            print("[AUTO-RECOVER] no gaps found in the ledgers -- nothing to heal")
            sys.exit(0)
        plan_path = DATA_DIR / "auto_recovery_plan.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print(f"[AUTO-RECOVER] plan -> {plan_path} "
              f"({len(plan)} chapter(s), "
              f"{sum(len(e['pages']) for e in plan.values())} page(s))")
        recover_pages(str(plan_path))
    else:
        main()
