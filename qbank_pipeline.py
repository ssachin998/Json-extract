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

def retry_batch_page_by_page(model, batch, state, ctx=None):
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
            items.extend(call_gemini_on_pages(model, [pf]))
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
                entry.update({"subject": ctx.get("subject"), "chapter_no": ctx.get("chapter_no"),
                              "chapter_id": ctx.get("chapter_id")})
            state.setdefault("failed_pages", []).append(entry)
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

def find_incomplete_records(chapter_records):
    """
    Returns [(q_no, missing_fields), ...] for records worth retrying.

    "answer" and "options" gaps are always retry-worthy: every real MCQ has
    4 options and one marked answer somewhere in the book.

    "solution" gaps are retry-worthy only when chapter-internal evidence
    says the book PRINTS explanations here (>=60% of questions already have
    solution text -- SOLUTION_GATE_MIN_SHARE). Chapters where the book
    genuinely prints no explanations (answer-key-only sections, RC-4) show
    ~0% coverage and stay protected: no quota is wasted chasing content
    that was never printed.
    """
    incomplete = []
    for qn, rec in chapter_records.items():
        if not (rec.get("question_text") or "").strip():
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
        if not rec.get("options") or len(rec["options"]) < 4:
            missing.append("options")
        if missing:
            incomplete.append((qn, missing))

    n = len(chapter_records)
    n_with_sol = sum(1 for r in chapter_records.values() if (r.get("solution_text") or "").strip())
    book_prints_solutions = n > 0 and n_with_sol / n >= SOLUTION_GATE_MIN_SHARE
    if book_prints_solutions:
        by_qn = {qn: missing for qn, missing in incomplete}
        for qn, rec in chapter_records.items():
            if rec.get("question_text") and not (rec.get("solution_text") or "").strip():
                if qn in by_qn:
                    by_qn[qn].append("solution")
                else:
                    incomplete.append((qn, ["solution"]))
    return incomplete


def build_targeted_retry_prompt(incomplete_items, chapter_records):
    lines = [
        "You already extracted most of this chapter's questions from these "
        "pages. A few specific pieces are still missing. Look at these SAME "
        "pages again, very carefully, and find ONLY the missing pieces listed "
        "below. Do not re-output anything else.",
        "",
        "Return a JSON array. Each element:",
        '{"q_no": <int>, "question_text": "..."|null, "correct_option": "A"|"B"|"C"|"D"|null, '
        '"options": {"A":"...","B":"...","C":"...","D":"..."} | null, '
        '"solution_text": "..." | null}',
        "Only fill the field(s) actually requested for that q_no; leave the "
        "other fields null. If you genuinely cannot find a piece anywhere in "
        "these pages, leave it null rather than guessing.",
        "",
        "MISSING PIECES TO FIND:",
    ]
    for qn, missing in incomplete_items:
        rec = chapter_records[qn]
        qtext = (rec.get("question_text") or "")[:120]
        block = [f"Question {qn} (\"{qtext}...\"):"]
        if "question" in missing:
            sol_anchor = (rec.get("solution_text") or "")[:150]
            ans_anchor = rec.get("correct_option")
            anchor_desc = (f'its printed solution begins "{sol_anchor}..."' if sol_anchor
                           else f"its marked correct option is {ans_anchor}")
            block.append(
                f"  - Find the FULL VERBATIM question stem AND all 4 options "
                f"(A/B/C/D) for question {qn}. Anchor: {anchor_desc}. Locate "
                f"the question that solution belongs to, on these SAME pages."
            )
        if "answer" in missing:
            block.append(
                f"  - Find the CORRECT OPTION LETTER for question {qn}. Check "
                f"any Answer Key table (a Question No. -> Correct Option table, "
                f"which may span two pages) for the row matching {qn}."
            )
        if "options" in missing:
            have = sorted((rec.get("options") or {}).keys())
            block.append(
                f"  - Find ALL 4 options (A/B/C/D) for question {qn}. "
                f"Already captured: {have or 'none'}. Find the missing letter(s)."
            )
        if "solution" in missing:
            block.append(
                f"  - Find the VERBATIM explanation/solution text printed for "
                f"question {qn}. It may sit in a 'Solutions'/'Explanations' block "
                f"near the questions, sometimes labelled 'Solution to Question "
                f"{qn}:'. Return the FULL text exactly as printed; if the page "
                f"genuinely shows none, leave it null."
            )
        lines.append("\n".join(block))
    return "\n".join(lines)


def targeted_retry(model, page_files, chapter_records, state, max_rounds=2):
    """
    Up to `max_rounds` small, focused re-asks for whatever answer/option
    fields are still missing after normal processing. Sends the chapter's
    full page set again each round (simple and robust -- we don't track
    per-field page provenance) but with a MUCH smaller ask, which is what
    actually improves accuracy, not the page count. Stops early if a round
    makes no progress (no point burning quota repeating the same miss).
    Returns the total number of fields filled.
    """
    total_fixed = 0
    first_check = True
    for round_no in range(1, max_rounds + 1):
        incomplete = find_incomplete_records(chapter_records)
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

        preview = ", ".join(f"q{qn}" for qn, _ in incomplete[:10])
        if len(incomplete) > 10:
            preview += ", ..."
        print(f"  [RETRY] round {round_no}: {len(incomplete)} question(s) still "
              f"incomplete ({preview}) -- sending targeted re-ask")

        prompt = build_targeted_retry_prompt(incomplete, chapter_records)
        try:
            parts = [prompt] + [Image.open(p) for p in page_files]
            resp = model.generate_content(
                parts, safety_settings=SAFETY_SETTINGS,
                request_options={"retry": None},
            )
            state["calls_today"] += 1
            save_state(state)
            if not resp.candidates:
                print("  [RETRY] empty/blocked response -- skipping this round")
                continue
            text = resp.text.strip()
            text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
            fixes = json.loads(text)
        except Exception as e:
            print(f"  [RETRY] call failed: {e} -- skipping this round")
            continue

        fixed_this_round = 0
        for fix in fixes:
            try:
                qn = int(fix.get("q_no"))
            except (TypeError, ValueError):
                continue
            rec = chapter_records.get(qn)
            if rec is None:
                continue
            if fix.get("question_text") and not (rec.get("question_text") or "").strip():
                rec["question_text"] = str(fix["question_text"]).strip()
                fixed_this_round += 1
            if fix.get("correct_option") and not rec.get("correct_option"):
                rec["correct_option"] = str(fix["correct_option"]).strip().upper()
                fixed_this_round += 1
            if fix.get("solution_text") and not (rec.get("solution_text") or "").strip():
                rec["solution_text"] = str(fix["solution_text"]).strip()
                fixed_this_round += 1
            if fix.get("options"):
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

    # whatever's STILL missing after all rounds -- log it, don't hide it
    still_incomplete = find_incomplete_records(chapter_records)
    if still_incomplete:
        path = DATA_DIR / "still_incomplete_after_retry.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)  # same guard as save_state/_append_jsonl (fresh volume)
        with open(path, "a", encoding="utf-8") as f:
            for qn, missing in still_incomplete:
                f.write(json.dumps({"q_no": qn, "missing": missing}, ensure_ascii=False) + "\n")
        print(f"  [RETRY] {len(still_incomplete)} question(s) still incomplete after "
              f"{max_rounds} round(s) -- logged to still_incomplete_after_retry.jsonl")

    return total_fixed

def pdftotext_page(pdf_path, true_page):
    out = subprocess.run(["pdftotext", "-f", str(true_page), "-l", str(true_page),
                          "-layout", str(pdf_path), "-"], capture_output=True, text=True)
    return out.stdout or ""


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
            raw = call_gemini_on_pages(model, [pf], context=RECOVERY_CONTEXT)
            state["calls_today"] += 1
            save_state(state)
        except Exception as e:
            print(f"  [DRAIN] {entry['page_file']} failed on second chance ({e}) -- kept in failed_pages queue")
            continue
        print(f"  [DRAIN] {entry['page_file']} recovered on second chance")
        items, _meta = extract_batch_meta(raw)
        chapter_records, skipped = merge_question_records(chapter_records, items, stats, fill_only=True)
        for it in skipped:
            new_orphans.append({"chapter_id": entry.get("chapter_id"), "batch_start": -1,
                                "pdf_pages": [int(entry["true_page"])], "new_pages": [],
                                "carry_q_no": None, "item": it})
        healed.append(entry)
    return chapter_records, new_orphans, healed

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

ANSWER_KEY_ROW_RE = re.compile(r"\|\s*(\d{1,3})\s*\|\s*([A-Da-d])\s*\|")
SOLUTION_TO_Q_RE = re.compile(r"Solution to Question\s+(\d{1,3})", re.IGNORECASE)

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
        filled_by_key = 0
        for t in item.get("tables") or []:
            if "answer" not in str(t.get("type", "")).lower() and "Correct Option" not in (t.get("markdown") or ""):
                continue
            for qn_s, letter in ANSWER_KEY_ROW_RE.findall(t.get("markdown") or ""):
                kqn = int(qn_s)
                rec = chapter_records.get(kqn)
                if rec and not rec.get("correct_option"):
                    rec["correct_option"] = letter.upper()
                    filled_by_key += 1
        if filled_by_key:
            stats["orphans_recovered"] += 1
            print(f"  [ORPHAN] Recovered orphan: page={page} answer-key table -> "
                  f"{filled_by_key} answer(s) filled deterministically")
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
                      and not _frag_mostly_present(frag, existing)):
                    owner, reason = last_qn, "solution continuation (PARTIAL owner append)"
                elif item.get("options") and not rec.get("options"):
                    owner, reason = last_qn, "options continuation"
                elif item.get("question_text") and not rec.get("question_text"):
                    owner, reason = last_qn, "question continuation"
        # ---- rule 4: positional certainty (Gap-1). An orphan carrying the
        # STEM (+options) can only belong to a record that is MISSING its
        # stem. Text-similarity between a stem and its own solution is
        # always ~0 (they never overlap lexically), so similarity-based
        # matching provably fails here (prod: PSY-001-003 stayed stemless
        # with answer+solution intact). When the chapter has EXACTLY ONE
        # stem-less record, position alone is the proof.
        if owner is None and item.get("question_text") and item.get("options"):
            stemless = [qn for qn, r in chapter_records.items()
                        if not (r.get("question_text") or "").strip()]
            if len(stemless) == 1:
                owner, reason = stemless[0], "question+options fallback (chapter's sole stem-less record)"
        if owner is None:
            print(f"  [ORPHAN] Could not determine owner: page={page} kept in orphans.jsonl")
            remaining.append(orph)
            continue
        rec = chapter_records[owner]
        if item.get("solution_text"):
            frag = item["solution_text"].strip()
            if frag and not _frag_mostly_present(frag, rec.get("solution_text") or ""):
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
        # ---- solution-style stem guard (backup net for the stale carry
        # bug): an unresolved carry context that survives into the Solutions
        # section can talk the model into CONTINUING the carried q_no with
        # solution PROSE as its question_text ("Option A: ...", "Answer: ...",
        # "Solution to Question 4: ..."). That text is not a stem -- reject
        # just this field (the item's real payload -- solution_text/options/
        # answer -- still merges below). A new record hit by this keeps its
        # other fields, stays stem-less, and becomes targeted-retry eligible
        # via the Gap-1 anchor rule instead of keeping a poisoned stem.
        if looks_like_solution_style_stem(item.get("question_text")):
            stats.setdefault("poison_stems_rejected", 0)
            stats["poison_stems_rejected"] += 1
            print(f"  [WARN] q{qn}: question_text is solution prose "
                  f"('{str(item['question_text'])[:60]}...') -- rejected as stem "
                  f"(stale carry-merge guard); other fields still merge")
            item = {**item, "question_text": None}
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
                print(f"  [WARN] question text for q{qn} differs between batches "
                      f"(similarity {sim:.2f}) -- merging non-conflicting fields")
        for k in ["question_text", "solution_text"]:
            if item.get(k):
                if fill_only and rec.get(k):
                    continue  # recovery: never overwrite existing content
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
                key = str(opt_id).strip().upper()
                if fill_only:
                    rec["options"].setdefault(key, opt_text)
                else:
                    rec["options"][key] = opt_text

        if item.get("correct_option"):
            if not (fill_only and rec.get("correct_option")):
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
    slots = pending_image_slots(chapter_records, image_files_by_q)
    if not slots:
        return list(imgs)
    if len(slots) == 1 or len(imgs) == 1:
        qn, kind = slots[0]
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
        carry_tracker = {}         # q_no -> batch-seq its UNRESOLVED carry opened
        carry_banned = set()       # expired q_nos: never respawn a carry this chapter
        answers_section_seen = False
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
                    # Free tier = ~1500 req/day PER DAY but also capped PER MINUTE
                    # (~15 RPM). A burst 429 is NOT the daily cap -- back off once
                    # and retry before declaring the whole day over.
                    print(f"  [429] rate limited on {subject} ch{ch['chapter_no']} batch {batch_start}"
                          f" -- backing off 65s (could be the per-minute cap, not the daily one)")
                    time.sleep(65)
                    try:
                        raw_items = call_gemini_on_pages(genai_model, batch, context=context_str)
                        state["calls_today"] += 1
                        save_state(state)
                    except Exception as e2:
                        t2 = str(e2)
                        if "429" in t2 or "quota" in t2.lower():
                            print(f"  [QUOTA] still limited after 65s backoff -- daily cap it is. "
                                  f"Saving progress, exiting: {e2}")
                            save_state(state)
                            sys.exit(0)
                        print(f"  [WARN] post-backoff call failed differently: {e2}")
                        raw_items = retry_batch_page_by_page(genai_model, batch, state, ctx={"subject": subject, "chapter_no": ch["chapter_no"], "chapter_id": chapter_id})
                        if not raw_items:
                            continue
                else:
                    print(f"  [WARN] Gemini call failed on {subject} ch{ch['chapter_no']} batch {batch_start}: {e}")
                    # don't lose the whole batch over one bad page
                    raw_items = retry_batch_page_by_page(genai_model, batch, state, ctx={"subject": subject, "chapter_no": ch["chapter_no"], "chapter_id": chapter_id})
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
            # stale-carry guard #1: expire carries whose own split never
            # resolved (ban blocks the compute_carry fallback respawn)
            carry_from_prev = enforce_carry_expiry(carry_from_prev, stats["batches"],
                                                   carry_tracker, carry_banned,
                                                   chapter_records, chapter_id)
            # stale-carry guard #2: questions->solutions SECTION boundary.
            # The first batch that shows the answers section hard-resets ALL
            # questions-section carry context -- resolved or not -- so it can
            # NEVER bleed into the Solutions section and cross-merge there.
            if not answers_section_seen:
                boundary = detect_section_boundary(items)
                if boundary:
                    answers_section_seen = True
                    had_pending = carry_from_prev is not None or bool(carry_tracker)
                    carry_from_prev = None
                    carry_tracker.clear()
                    print(f"  [SECTION] {boundary} first seen at pages "
                          f"{window_pages[0]}-{window_pages[-1]} -- answers "
                          f"section begins; questions-section carry context "
                          f"HARD-RESET{' (dropped pending context)' if had_pending else ''}")
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
                leftover = claim_page_images_one_to_one(imgs, pdf_path, file_page_num, subject,
                                                        ch["chapter_no"], chapter_records, image_files_by_q)
                if leftover:
                    unmatched_images.append({"page": file_page_num, "files": leftover})
                    print(f"  [INFO] Page {file_page_num}: image(s) {leftover} unclaimed for now "
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
            leftover2 = claim_page_images_one_to_one(um["files"], pdf_path, um["page"],
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
                other = "solution" if side == "question" else "question"
                kind = "Q" if side == "question" else "SOL"
                if not entry[side]:
                    qid = f"{subject}-{ch['chapter_no']:03d}-{qn:03d}"
                    renamed = []
                    ok = True
                    for i, old_rel in enumerate(um["files"], 1):
                        old_path = ASSETS_DIR / "questions" / old_rel
                        if not old_path.exists():
                            ok = False
                            break
                        new_name = f"{qid}_{kind}_{i:02d}.webp"
                        new_rel = f"{subject}/{new_name}"
                        old_path.rename(ASSETS_DIR / "questions" / subject / new_name)
                        renamed.append(new_rel)
                    if ok and renamed:
                        entry[side] = renamed
                        um["matched"] = True
                        print(f"  [INFO] third pass: page {um['page']} image(s) attached to {qid} "
                              f"(sole printed question on that page, {side} side)")
                    elif not ok:
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
            still, brake_hit = [], False
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
                    still.append(rel)
            um["files"] = still
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
                _append_jsonl(DATA_DIR / "unmatched_images.jsonl",
                              {"subject": subject, "chapter_id": chapter_id,
                               "page": um["page"], "files": um["files"]})

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

        # FEATURE: targeted gap-retry -- AFTER normal batches + orphan
        # recovery, BEFORE writing the chapter's questions to disk.
        n_fixed = targeted_retry(genai_model, page_files, chapter_records,
                                 state, max_rounds=TARGETED_RETRY_MAX_ROUNDS)
        if n_fixed:
            print(f"  [RETRY] closed {n_fixed} field(s) via targeted retry")

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
              f" | unmatched images: {n_unmatched}")

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
    }
    owned = {"question": [i["file"] for i in q["question"]["images"]],
             "solution": [i["file"] for i in q["solution"]["images"]]}
    return rec, owned

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
                 "carry_merges": 0, "orphans_recovered": 0}
        orphans = []
        unmatched_images = []

        for win_start in range(0, len(page_files), PAGES_PER_GEMINI_CALL):
            batch = page_files[win_start:win_start + PAGES_PER_GEMINI_CALL]
            window_pages = [int(p.stem.split("-")[-1]) for p in batch]
            try:
                raw = call_gemini_on_pages(model, batch, context=RECOVERY_CONTEXT)
            except Exception as e:
                print(f"  [WARN] recovery call failed for {chapter_id} pages "
                      f"{window_pages}: {e}")
                raw = retry_batch_page_by_page(model, batch,
                                               {"calls_today": 0, "day_stamp": ""})
                if not raw:
                    continue
            items, _meta = extract_batch_meta(raw)
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
                    rec_leftover = claim_page_images_one_to_one(imgs, pdf_path, file_page_num,
                                                                subject, chapter_no,
                                                                records, image_files_by_q)
                    if rec_leftover:
                        unmatched_images.append({"page": file_page_num, "files": rec_leftover})

        orphans = recover_orphans(orphans, records, subject, chapter_no, stats)
        for orph in orphans:
            _append_jsonl(DATA_DIR / "orphans.jsonl", orph)
        for um in unmatched_images:
            rec_leftover = claim_page_images_one_to_one(um["files"], pdf_path, um["page"],
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
              f" | rows kept: {len(records) - len(chapter_lines) + len(chapter_lines)}"
              f" | conflicts dropped: {stats['conflicts']}"
              f" | orphans unresolved: {len(orphans)}"
              f" | state.json untouched")

    print("[RECOVER] all planned chapters processed.")

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
    if len(sys.argv) > 1 and sys.argv[1] == "--recover":
        # targeted healing of already-written rows, e.g.:
        #   python3 qbank_pipeline.py --recover recovery_plan.json
        recover_pages(sys.argv[2] if len(sys.argv) > 2 else "recovery_plan.json")
    else:
        main()
