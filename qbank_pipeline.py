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
import difflib
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
BATCH_OVERLAP_PAGES = 2         # consecutive batches share 2 pages, so a
                                 # question/solution split across a batch
                                 # boundary is seen WHOLE (with its q_no) in at
                                 # least one call -- see ROOT_CAUSE_ANALYSIS.md
                                 # RC-3. Step size = 6-2 = 4 new pages/call.
                                 # Merge by q_no makes re-extraction idempotent.
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

def write_chapters(path, chapters_out):
    """Write chapters.json deduplicated by chapter_id (last entry wins).
    Dedup matters when a chapter is re-processed after manual state surgery
    (removing its id from chapters_done to force re-extraction): the id gets
    appended again while an older entry is already in chapters.json."""
    uniq = {}
    for c in chapters_out:
        uniq[c["chapter_id"]] = c
    path.write_text(json.dumps(list(uniq.values()), indent=2, ensure_ascii=False))

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

def call_gemini_on_pages(model, image_paths, context=""):
    parts = [SCHEMA_PROMPT]
    if context:
        parts.append(context)  # carry-forward / overlap context (stateless API)
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

def retry_batch_page_by_page(model, batch, state):
    """A whole-batch failure (RECITATION/safety finish_reason, token limit)
    is usually caused by just ONE page in the batch. Retrying each page
    alone isolates the bad page instead of losing the whole batch's worth
    of questions/answers/solutions (seen in prod: finish_reason=4 killed a
    6-page batch, wiping one chapter's answers and another's solutions).
    Respects the same daily quota and exits cleanly if it's hit."""
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
            items.extend(call_gemini_on_pages(model, [pf]))
            state["calls_today"] += 1
            recovered += 1
        except Exception as e2:
            t2 = str(e2)
            if "429" in t2 or "quota" in t2.lower():
                print(f"  [QUOTA] Gemini quota exhausted during retry -- stopping run for now: {e2}")
                save_state(state)
                sys.exit(0)
            print(f"  [WARN] page {pf.name} failed even alone ({e2}) -- skipping just this page")
    save_state(state)
    print(f"  [INFO] single-page retry: {recovered}/{len(batch)} pages recovered")
    return items

# ============================================================
# FEATURE 2 — carry-forward context (Gemini's API is stateless:
# continuity must be injected manually into every new request)
# ============================================================

def extract_batch_meta(items):
    """Peel the {"_batch_meta": {...}} control object out of Gemini's array.
    Returns (question_items, meta_dict). Meta of a failed/absent call = {}."""
    questions, meta = [], {}
    for it in items:
        if isinstance(it, dict) and "_batch_meta" in it:
            m = it.get("_batch_meta")
            if isinstance(m, dict):
                meta = m          # last one wins (single-page retries)
            continue
        questions.append(it)
    return questions, meta

def compute_carry(batch_meta, items, chapter_records, ending_page):
    """Decide whether a batch ended mid-question and build the payload carried
    into the NEXT request. Primary signal: Gemini's own _batch_meta (it can
    see the page bottom). Fallback when no usable meta: the highest q_no from
    this batch whose record has question text but no solution yet.
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
        for it in items:
            try:
                batch_qns.append(int(it.get("q_no")))
            except (TypeError, ValueError):
                pass
        if not batch_qns:
            return None
        candidate = max(batch_qns)
        rec = chapter_records.get(candidate, {})
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

def build_carry_context(carry, overlap_pages):
    """The actual text prepended to the next request."""
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
            f"The first {len(overlap_pages)} page image(s) (PDF page(s) "
            f"{', '.join(map(str, overlap_pages))}) are OVERLAP from the previous "
            "batch, provided as context only. Extract the new pages normally; if "
            "an item spans an overlap page into the new pages, combine both "
            "sides into ONE complete item under its printed q_no."
        )
    return "\n".join(lines)

def recover_orphans(orphans, chapter_records, subject, chapter_no, stats):
    """FEATURE 3 -- second-pass owner matching for q_no=null fragments.
    Confidence rules, in order:
      1. the carry-forward owner captured when the fragment arrived
      2. the highest-numbered question from the SAME batch window is missing
         exactly the field the orphan provides (solution/options/question)
    Recovered content is APPENDED (existing text is never overwritten).
    Whatever remains unmatched is returned for orphans.jsonl -- never
    silently discarded."""
    remaining = []
    for orph in orphans:
        item = orph["item"]
        owner, reason = None, None
        carry_qn = orph.get("carry_q_no")
        if carry_qn is not None and carry_qn in chapter_records:
            owner = carry_qn
            reason = f"{orph.get('cut_part') or 'content'} continuation (carry-forward)"
        else:
            last_qn = orph.get("last_qn_in_batch")
            rec = chapter_records.get(last_qn) if last_qn is not None else None
            if rec:
                if item.get("solution_text") and not rec.get("solution_text"):
                    owner, reason = last_qn, "solution continuation"
                elif item.get("options") and not rec.get("options"):
                    owner, reason = last_qn, "options continuation"
                elif item.get("question_text") and not rec.get("question_text"):
                    owner, reason = last_qn, "question continuation"
        page = (orph.get("new_pages") or orph.get("pdf_pages") or ["?"])[0]
        if owner is None:
            print(f"  [ORPHAN] Could not determine owner: page={page} kept in orphans.jsonl")
            remaining.append(orph)
            continue
        rec = chapter_records[owner]
        if item.get("solution_text"):
            frag = item["solution_text"].strip()
            if frag and frag not in (rec.get("solution_text") or ""):
                rec["solution_text"] = ((rec.get("solution_text") or "") + " " + frag).strip()
        if item.get("options"):
            rec["options"] = rec["options"] or {}
            for k, v in item["options"].items():
                rec["options"].setdefault(str(k).strip().upper(), v)
        if item.get("question_text") and not rec.get("question_text"):
            rec["question_text"] = item["question_text"]
        if item.get("correct_option") and not rec.get("correct_option"):
            rec["correct_option"] = str(item["correct_option"]).strip().upper()
        if item.get("tables"):
            have = {t.get("markdown") for t in rec["tables"]}
            for t in item["tables"]:
                if t.get("markdown") not in have:
                    rec["tables"].append(t)
                    have.add(t.get("markdown"))
        qid = f"{subject}-{chapter_no:03d}-{owner:03d}"
        print(f"  [ORPHAN] Recovered orphan: page={page} assigned_to={qid} reason={reason}")
        stats["orphans_recovered"] += 1
        if "carry-forward" in reason:
            stats["carry_merges"] += 1
    return remaining

# ============================================================
# STEP 4: merge partial results (a question's text might be on one
# page and its answer/solution on a later page) into final records
# ============================================================

def merge_question_records(existing, new_items, stats=None):
    """existing: dict keyed by q_no -> record (in progress for current chapter).
    stats: optional dict updated with "duplicates_merged"/"conflicts" counters.

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
        rec = existing.setdefault(qn, {
            "q_no": qn, "question_text": None, "options": None,
            "correct_option": None, "solution_text": None, "tables": [],
            "has_figure_in_question": False, "has_figure_in_solution": False,
        })
        # ---- duplicate / conflict classification for overlap pages ----
        old_q, new_q = rec.get("question_text"), item.get("question_text")
        if old_q and new_q:
            sim = difflib.SequenceMatcher(None, old_q, new_q).ratio()
            if sim >= 0.95:
                stats["duplicates_merged"] += 1   # expected overlap re-read
            else:
                a1, a2 = rec.get("correct_option"), item.get("correct_option")
                stats["conflicts"] += 1
                if a1 and a2 and a1 != a2:
                    print(f"  [WARN] conflicting re-extraction for q{qn} "
                          f"(similarity {sim:.2f}, answers {a1} vs {a2}) -- keeping first, dropping item")
                    continue
                print(f"  [WARN] question text for q{qn} differs between batches "
                      f"(similarity {sim:.2f}) -- merging non-conflicting fields")
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
            # Dedupe by markdown: overlap pages (BATCH_OVERLAP_PAGES) are
            # extracted twice, and blindly extending would duplicate tables.
            have = {t.get("markdown") for t in rec["tables"]}
            for t in item["tables"]:
                if t.get("markdown") not in have:
                    rec["tables"].append(t)
                    have.add(t.get("markdown"))
        rec["has_figure_in_question"] = rec["has_figure_in_question"] or item.get("has_figure_in_question", False)
        rec["has_figure_in_solution"] = rec["has_figure_in_solution"] or item.get("has_figure_in_solution", False)
    return existing, skipped

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

def claim_images_for_question(imgs, subject, chapter_no, chapter_records, image_files_by_q):
    """Attach one page's extracted images to the first question Gemini flagged
    as needing a figure (question OR solution) that doesn't have one of that
    kind yet. Returns True if claimed.

    NOTE: simple heuristic -- with multiple figures/flagged questions per page,
    review the mapping manually.

    LOCKED structure: ALL figure types live together in
    assets/questions/{SUBJECT}/ -- one folder per subject, type is told only
    by the filename suffix:
      {id}_Q_01.webp  {id}_SOL_01.webp  {id}_OPT_A_01.webp  {id}_TABLE_01.webp
    Do NOT create assets/solutions/, assets/options/ or assets/tables/ --
    the app relies on this convention."""
    for qn, rec in chapter_records.items():
        entry = image_files_by_q.setdefault(qn, {"question": [], "solution": []})
        needs_q_img = rec.get("has_figure_in_question") and not entry["question"]
        needs_sol_img = rec.get("has_figure_in_solution") and not entry["solution"]
        if not (needs_q_img or needs_sol_img):
            continue
        kind = "Q" if needs_q_img else "SOL"
        qid = f"{subject}-{chapter_no:03d}-{qn:03d}"
        renamed = []
        for old_rel in imgs:
            old_path = ASSETS_DIR / "questions" / old_rel
            if not old_path.exists():
                # never let one bad path kill the whole run
                print(f"  [WARN] {old_rel} missing at rename time "
                      f"-- skipping (leftover alias/dup ref)")
                continue
            idx = len(renamed) + 1
            new_name = f"{qid}_{kind}_{idx:02d}.webp"
            new_rel = f"{subject}/{new_name}"
            new_path = ASSETS_DIR / "questions" / subject / new_name
            old_path.rename(new_path)
            renamed.append(new_rel)
        entry["question" if kind == "Q" else "solution"] = renamed
        return True  # this page's image(s) assigned; don't hand to a later question
    return False

def _append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

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
        image_files_by_q = {}
        pages_imaged = set()       # overlap pages must not be image-extracted twice
        unmatched_images = []      # no claimant yet -- retried at chapter end
        orphans = []               # Gemini items with null/invalid q_no (RC-2)
        stats = {"batches": 0, "duplicates_merged": 0, "conflicts": 0,
                 "carry_used": 0, "carry_merges": 0,
                 "orphans_recovered": 0, "orphans_buffered": 0, "orphans_remaining": 0}
        carry_from_prev = None     # FEATURE 2 payload for the NEXT request
        prev_window_last_page = None

        overlap = max(0, min(BATCH_OVERLAP_PAGES, PAGES_PER_GEMINI_CALL - 1))
        batch_step = PAGES_PER_GEMINI_CALL - overlap
        for batch_start in range(0, len(page_files), batch_step):
            if batch_start and batch_start + overlap >= len(page_files):
                break  # trailing window would contain ONLY overlap pages
                       # (nothing new) -- don't spend a quota call on it
            reset_daily_counter_if_needed(state)
            if state["calls_today"] >= MAX_CALLS_PER_DAY:
                print("Daily Gemini call limit reached. Saving progress, exiting.")
                save_state(state)
                sys.exit(0)

            batch = page_files[batch_start:batch_start + PAGES_PER_GEMINI_CALL]
            window_pages = [int(p.stem.split("-")[-1]) for p in batch]
            overlap_pages = [pn for pn in window_pages
                             if prev_window_last_page is not None and pn <= prev_window_last_page]
            new_pages = [pn for pn in window_pages if pn not in overlap_pages]
            carry_in = carry_from_prev                      # context for THIS call
            context_str = build_carry_context(carry_in, overlap_pages)
            if carry_in:
                stats["carry_used"] += 1
            try:
                raw_items = call_gemini_on_pages(genai_model, batch, context=context_str)
                state["calls_today"] += 1
                save_state(state)
            except Exception as e:
                err_text = str(e)
                if "429" in err_text or "quota" in err_text.lower():
                    print(f"  [QUOTA] Gemini quota exhausted -- stopping run for now: {e}")
                    save_state(state)
                    sys.exit(0)
                print(f"  [WARN] Gemini call failed on {subject} ch{ch['chapter_no']} batch {batch_start}: {e}")
                # don't lose the whole batch over one bad page
                raw_items = retry_batch_page_by_page(genai_model, batch, state)
                if not raw_items:
                    continue

            items, batch_meta = extract_batch_meta(raw_items)
            chapter_records, skipped = merge_question_records(chapter_records, items, stats)
            try:
                last_qn_in_batch = max(int(it.get("q_no")) for it in items
                                       if it.get("q_no") is not None)
            except (ValueError, TypeError):
                last_qn_in_batch = None
            for it in skipped:
                # RC-2 salvage buffer: fragments (usually batch-boundary
                # continuations) carry real content -- keep with provenance
                # for the second-pass recovery at chapter end.
                orphans.append({
                    "chapter_id": chapter_id, "batch_start": batch_start,
                    "pdf_pages": window_pages, "new_pages": new_pages,
                    "carry_q_no": carry_in["last_open_question"] if carry_in else None,
                    "cut_part": carry_in.get("cut_part") if carry_in else None,
                    "last_qn_in_batch": last_qn_in_batch,
                    "item": it,
                })
            stats["orphans_buffered"] += len(skipped)
            stats["batches"] += 1
            prev_window_last_page = max(window_pages)
            carry_from_prev = compute_carry(batch_meta, items, chapter_records,
                                            prev_window_last_page)
            last_open = (f"q{carry_from_prev['last_open_question']}"
                         if carry_from_prev and carry_from_prev["last_open_question"] is not None
                         else ("open (no number)" if carry_from_prev else "-"))
            print(f"  [GEMINI] pages {window_pages[0]}-{window_pages[-1]}"
                  f" | overlap: {overlap_pages if overlap_pages else '-'}"
                  f" | carry-in: {('q' + str(carry_in['last_open_question'])) if carry_in and carry_in['last_open_question'] is not None else '-'}"
                  f" | last-open: {last_open}"
                  f" | items: {len(items)} | orphans buffered: {len(skipped)}")

            # extract real (non-watermark) images from this batch's pages.
            # pdftoppm names output files using the ACTUAL pdf page number
            # (e.g. page-005.jpg for real page 5) -- read it directly from
            # the filename, don't recompute it relative to ch["file_start"].
            for pf in batch:
                file_page_num = int(pf.stem.split("-")[-1])
                if file_page_num in pages_imaged:
                    continue  # overlap page -- images already extracted once
                pages_imaged.add(file_page_num)
                imgs = extract_real_images(pdf_path, file_page_num, watermark_id, subject, ASSETS_DIR / "questions")
                if not imgs:
                    continue
                if not claim_images_for_question(imgs, subject, ch["chapter_no"], chapter_records, image_files_by_q):
                    unmatched_images.append({"page": file_page_num, "files": imgs})
                    print(f"  [INFO] Page {file_page_num}: image(s) {imgs} unclaimed for now "
                          f"-- will retry after all batches (owner may be in a later batch)")

        # FEATURE 3 -- orphan recovery runs BEFORE image claiming and JSON
        # writing: recovered fragments can complete solutions/options, and
        # only genuinely ownerless orphans are persisted.
        orphans = recover_orphans(orphans, chapter_records, subject, ch["chapter_no"], stats)
        stats["orphans_remaining"] = len(orphans)
        for orph in orphans:
            _append_jsonl(DATA_DIR / "orphans.jsonl", orph)

        # SECOND PASS image claiming: a figure can be extracted BEFORE the
        # batch that introduces its owning question (plate printed just before
        # the question text, or owner arrived via an overlap window). Chapter
        # records are complete now -- retry every leftover once.
        n_unmatched = 0
        for um in unmatched_images:
            if claim_images_for_question(um["files"], subject, ch["chapter_no"], chapter_records, image_files_by_q):
                print(f"  [INFO] second pass: page {um['page']} image(s) matched to a question")
                um["matched"] = True
        for um in unmatched_images:
            if not um.get("matched"):
                n_unmatched += 1
                print(f"  [WARN] Page {um['page']}: extracted image(s) {um['files']} but no "
                      f"question/solution in this chapter claimed one -- left under its temp "
                      f"filename for manual review (see data/unmatched_images.jsonl).")
                _append_jsonl(DATA_DIR / "unmatched_images.jsonl",
                              {"subject": subject, "chapter_id": chapter_id,
                               "page": um["page"], "files": um["files"]})

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
        write_chapters(chapters_path, chapters_out)
        n_no_answer = sum(1 for r in chapter_records.values() if not r.get("correct_option"))
        n_no_solution = sum(1 for r in chapter_records.values() if not r.get("solution_text"))
        print(f"[{subject}] chapter {ch['chapter_no']} ({ch['chapter_title']}) done -> "
              f"{len(chapter_records)} questions ({n_no_answer} missing answer, {n_no_solution} missing solution)")
        print(f"[{subject}]   batches: {stats['batches']} | duplicates merged: {stats['duplicates_merged']}"
              f" | conflicts dropped: {stats['conflicts']} | carry-forward used: {stats['carry_used']}"
              f" | carry merges: {stats['carry_merges']}"
              f" | orphans: {stats['orphans_recovered']} recovered, {stats['orphans_remaining']} unresolved"
              f" | unmatched images: {n_unmatched}")

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
    print("All done (or paused at daily limit -- just re-run this script to resume).")

if __name__ == "__main__":
    main()
