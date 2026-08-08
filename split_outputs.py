"""
split_outputs.py
================

Per-chapter SPLIT-OUTPUT LAYER (additive, observation-only).

Writes, after every existing in-pipeline reconciliation, three strictly
separate JSONL datasets joined by a stable q_id, plus a chapter
completeness summary, a chapter-scoped image manifest, a per-chapter
orphan view, and an unresolved-qids ledger:

  data/split/{subject}/{chapter_id}/
    questions.jsonl
    answers.jsonl
    solutions.jsonl
    unresolved_qids.jsonl
    orphans.jsonl
    chapter_completeness.json
    image_manifest.jsonl

The existing data/questions.jsonl and data/by_chapter/{chapter_id}.jsonl
are NOT touched by this module. The split is built from the same
chapter_records dict the master file is built from, so the two are
guaranteed consistent for the same chapter.

Inserted at one call site in process_pdf() (qbank_pipeline.py), AFTER
every existing check (batches, orphans, drain, sweep, retry, rescue,
anchorless drop, phantom drop, critique) and BEFORE the existing
build_final_question loop.

Public entry points
-------------------
- reconcile_qids(chapter_records, qn_source_pages, pdf_path, page_files,
                  subject, chapter_no) -> dict
      Walk the chapter's text layer ONCE, harvest every printed
      question/solution header, answer-key row, and block position for
      every q_no in chapter_records. Assign a 4-grade provenance label
      (RESOLVED_ANCHORED / RESOLVED / PROVISIONAL / UNRESOLVED). Populate
      a q_no_anchors vector on every record. Remove UNRESOLVED records
      from chapter_records and return them separately (the caller passes
      them to write_split_outputs -> unresolved_qids.jsonl).

- write_split_outputs(chapter_id, subject, chapter_no, chapter_records,
                      image_files_by_q, qn_source_pages, orphans,
                      chapter_unresolved_images, pdf_path, page_files,
                      reconciled) -> dict
      Write the seven per-chapter files atomically. chapter_completeness.json
      is written LAST as the "split is fully on disk" signal. Returns the
      chapter_completeness.json content as a dict for the caller's logs.

Provenance taxonomy (confirmed in design doc §2)
------------------------------------------------
- RESOLVED_ANCHORED  >=2 printed anchors agree + at least one of
                      printed_stem_match / printed_solution_header_match
                      is set
- RESOLVED           1 printed anchor only (Phase 1: the connected-run
                      and carry-forward origins are documented in the
                      Phase-2 hook plan and not yet observed; single
                      anchor stays RESOLVED, not PROVISIONAL)
- PROVISIONAL        only the model's q_no, no printed anchor
- UNRESOLVED         no anchor at all, OR two printed anchors disagree
                      (Case 2: missing_question_for_solution), OR
                      only foreign / hallucination-source q_no

Phase-2 plan (NOT implemented; documented for the next change)
--------------------------------------------------------------
The full design doc §3.1 also calls for two non-printed anchors
(neighbor_run from the run-18 GUARD's connected-run analysis, and
carry_forward_origin from compute_carry). Both live inside the
Q-pass block in process_pdf as transient local variables; capturing
them requires either: (a) capturing the batch_qnos / runs / carry_obj
into a per-chapter dict from inside the loop, or (b) re-running the
connected-run analysis from the page_ledger.jsonl. Option (a) is the
cleaner Phase-2 plan -- ~6 lines of read-only observation added to
the existing loop, no behavior change. Until that lands, the grader
distinguishes only the printed anchors above, which still covers
the most common 2-anchor case in MARROW/PSY-style books.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable


# Allowed values -- per the design contract. New grades are an explicit
# break of the contract and require a design-doc update.
ALLOWED_Q_ID_GRADES = frozenset(
    {"RESOLVED_ANCHORED", "RESOLVED", "PROVISIONAL", "UNRESOLVED"}
)
ALLOWED_EXTRACTION_STATUS = frozenset({"COMPLETE", "INCOMPLETE"})

# Reasons for unresolved_qids.jsonl (design doc §6). Kept short,
# future-extensible. Any new reason is fine; these are the v1 set.
UNRESOLVED_REASONS = frozenset({
    "no_anchor_at_all",
    "model_q_no_disagree",
    "conflicting_anchors",
    "hallucinated_q_no",
    "solution_q_no_not_in_printed_header",
    "answer_q_no_not_in_printed_key",
    "two_possible_questions",
    "question_continues_from_previous_page",
    "foreign_chapter_q_no",
    "missing_question_for_solution",
})

# Reasons for orphans.jsonl (design doc §7).
ORPHAN_REASONS = frozenset({
    "q_id_unresolved",
    "foreign_option_line_head",
    "first_line_in_sibling_solution",
    "owner_attached_but_speculative",
    "unconfirmed_discontinuous_qno",  # matches the existing pipeline
})


# ---------------------------------------------------------------------------
# 1. Atomic write helpers -- exactly the per-chapter pattern the existing
#    pipeline uses for questions.jsonl (see rewrite_questions_file in
#    qbank_pipeline.py). A crash mid-write leaves either the previous file
#    untouched or the new file complete.
# ---------------------------------------------------------------------------

def _atomic_jsonl_write(path: Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _atomic_json_write(path: Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# 2. Anchor harvesting -- pure read-only pass over the chapter's pages.
#    Uses the same zero-token text-layer + pypdf-visitor + answer-key
#    regexes the existing pipeline uses, so the anchors are exactly the
#    evidence the existing extraction loop already trusts.
# ---------------------------------------------------------------------------

_PRINTED_STEM_RE = re.compile(
    r"^\s*(?:Q(?:uestion)?\s*[.:]?\s*)?(\d{1,3})\s*[.:\-\u2013)]"
)
_PRINTED_SOL_HEADER_RE = re.compile(
    r"Solution\s+to\s+Question\s+(\d{1,3})", re.IGNORECASE
)
# Answer-key row: "| 13 | B |", "13. B", "13 - B". Same shape as
# locate_missing_record_pages in qbank_pipeline.py.
_ANSWER_KEY_ROW_RE = re.compile(
    r"(?m)^\s*\|\s*(\d{1,3})\s*\|\s*([A-Da-d])\s*\|"          # | 13 | B |
    r"|^\s*(\d{1,3})\s*[.)]\s*([A-Da-d])\s*$"                 # 13. B / 13) B
    r"|^\s*(\d{1,3})\s*[-\u2013]\s*([A-Da-d])\s*$"           # 13 - B
)
_ANSWER_KEY_PROBE_RE = re.compile(
    r"(question\s*no|q\.?\s*no)[^\n]{0,40}(correct\s*option|answer)"
    r"|answer\s*key",
    re.IGNORECASE,
)


def _pdftotext_page(pdf_path: str, true_page: int) -> str:
    """Zero-token text-layer read of a single page. Mirrors the existing
    pipeline's pdftotext_page (we can't import qbank_pipeline without
    pulling in google-generativeai, which we don't want as a hard
    dep for the synthetic harness)."""
    import subprocess
    try:
        out = subprocess.run(
            ["pdftotext", "-f", str(true_page), "-l", str(true_page),
             "-layout", str(pdf_path), "-"],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout or ""
    except Exception:
        return ""


def _page_word_lines(pdf_path: str, file_page: int):
    """[(y_baseline, line_text)] in PDF user space, top-first.
    Subset of the existing _page_word_lines in qbank_pipeline.py --
    only the joined-line text is needed by the grader (we don't need
    the per-word x positions). Falls back to empty if pypdf can't
    parse the page (scanned-only PDFs)."""
    try:
        from pypdf import PdfReader
        page = PdfReader(pdf_path).pages[file_page - 1]
    except Exception:
        return []
    words = []

    def _visitor(text, _cm, tm, _font_dict, _font_size):
        t = (text or "").strip()
        if t:
            words.append((round(float(tm[5]), 1), t))

    try:
        page.extract_text(visitor_text=_visitor)
    except Exception:
        return []
    if not words:
        return []
    lines: dict = {}
    for y, t in words:
        lines.setdefault(y, []).append(t)
    return [(y, " ".join(parts)) for y, parts in sorted(lines.items(), reverse=True)]


def _harvest_page(pdf_path: str, file_page: int, chapter_records: dict) -> dict:
    """Returns {qn: {anchor_name: {...}|None}} for a single page. Each
    anchor is captured at most once per page; later pages only add
    anchors when they discover new evidence."""
    found: dict = {}
    qn_set = set(chapter_records)
    # pypdf-visitor path (body pages: prints "1.", "2)", "Question 3:")
    for _y, line in _page_word_lines(pdf_path, file_page):
        m = _PRINTED_STEM_RE.match(line)
        if m and not line.lstrip().lower().startswith("solution to question"):
            try:
                qn = int(m.group(1))
            except (TypeError, ValueError):
                qn = None
            if qn in qn_set:
                found.setdefault(qn, {})["printed_stem_match"] = {
                    "page": file_page, "header_text": line.strip()[:80]
                }
        for sm in _PRINTED_SOL_HEADER_RE.finditer(line):
            try:
                qn = int(sm.group(1))
            except (TypeError, ValueError):
                qn = None
            if qn in qn_set:
                found.setdefault(qn, {})["printed_solution_header_match"] = {
                    "page": file_page, "header_text": line.strip()[:80]
                }
    # pdftotext path (answer-key rows, which sit in a table that the
    # pypdf visitor may format differently across PDFs)
    text = _pdftotext_page(pdf_path, file_page)
    if text.strip():
        for m in _ANSWER_KEY_ROW_RE.finditer(text):
            qn = int(m.group(1) or m.group(3) or m.group(5))
            letter = (m.group(2) or m.group(4) or m.group(6) or "").upper()
            if qn in qn_set:
                row_text = (m.group(0) or "").strip()[:80]
                # first-seen wins; later pages don't overwrite a confirmed row
                found.setdefault(qn, {}).setdefault("answer_key_row_match", {
                    "page": file_page, "row": row_text, "letter": letter
                })
    return found


def _harvest_anchors(chapter_records: dict, qn_source_pages: dict,
                     pdf_path: str, page_files) -> dict:
    """Per-q_no: {anchor_name: {page, ...}|None}. Walks the chapter's
    pages ONCE, reading each page's text layer (zero Gemini calls)
    and recording the strongest printed evidence found for each q_no.
    Pages come from page_files when given (the in-pipeline path),
    else from qn_source_pages (the synthetic harness path)."""
    pages: list
    if page_files:
        pages = []
        for pf in page_files:
            try:
                pages.append(int(pf.stem.split("-")[-1]))
            except (ValueError, IndexError):
                continue
    else:
        # Synthetic harness: union of qn_source_pages values
        pages = set()
        for sp in (qn_source_pages or {}).values():
            if isinstance(sp, (set, list, tuple)):
                pages.update(int(p) for p in sp)
            elif isinstance(sp, int):
                pages.add(sp)
        pages = sorted(pages)
    if not pages:
        return {qn: {} for qn in chapter_records}
    per_qn: dict = {qn: {} for qn in chapter_records}
    for p in pages:
        page_harvest = _harvest_page(pdf_path, p, chapter_records)
        for qn, anchors in page_harvest.items():
            for name, payload in anchors.items():
                per_qn[qn].setdefault(name, payload)
    return per_qn


# ---------------------------------------------------------------------------
# 3. Grader -- deterministic, 4-grade taxonomy.
# ---------------------------------------------------------------------------

def _grade_record(anchors: dict) -> str:
    """Map the printed anchors found for one record to a q_id_grade.

    The Phase-1 grader distinguishes:
      - RESOLVED_ANCHORED: >=2 printed anchors + at least one of the
        two high-confidence printed anchors (printed_stem_match or
        printed_solution_header_match) is set
      - RESOLVED:           exactly 1 printed anchor
      - PROVISIONAL:        no printed anchor at all (model-only q_no)
      - UNRESOLVED:         impossible from printed anchors alone in
        Phase 1; the caller (reconcile_qids) sets this when:
          * two printed anchors disagree on the q_no (impossible if
            anchors were harvested correctly per-q_no, but kept here
            for forward-compat)
          * the record is a Case 2 / missing_question_for_solution
            scenario (set explicitly by reconcile_qids)
    """
    matches = sum(bool(anchors.get(k)) for k in (
        "printed_stem_match",
        "printed_solution_header_match",
        "answer_key_row_match",
    ))
    if matches == 0:
        return "PROVISIONAL"
    if matches >= 2 and (anchors.get("printed_stem_match")
                         or anchors.get("printed_solution_header_match")):
        return "RESOLVED_ANCHORED"
    return "RESOLVED"


# ---------------------------------------------------------------------------
# 4. Per-record provenance collection
# ---------------------------------------------------------------------------

def _collect_provs(rec: dict) -> tuple:
    """Returns (model_q_no_provs, model_q_no, disagree).

    The existing pipeline's per-field _prov dict stores a string per
    populated field -- the values are typically 'Q_PASS', 'A_PASS',
    'S_PASS', 'Q_RETRY', 'A_RETRY', 'S_RETRY', 'RESCUE', 'RECOVER',
    'DRAIN_Q', 'OCR_S', 'DRAIN_S', etc. This is the ONLY deterministic
    way to know which extraction pass a record's content came from,
    and it's what the design's q_no_anchors.provenance_notes vector
    surfaces. Two passes that disagree on q_no is the disagreement
    signal (rare in the existing pipeline because the run-18 GUARD
    already filters unverified q_nos to orphans)."""
    provs: list = []
    prov = rec.get("_prov") or {}
    for _field, label in prov.items():
        if label:
            provs.append(str(label))
    # Fall back to the model-emitted q_no (always the integer)
    try:
        model_q_no = int(rec.get("q_no")) if rec.get("q_no") is not None else None
    except (TypeError, ValueError):
        model_q_no = None
    # Disagreement: not currently inferable from the per-field provs (the
    # pass name is a label, not a q_no). The existing pipeline captures
    # cross-pass disagreement via the unconfirmed_discontinuous_qno
    # guard, which is observable from the orphans list (not from
    # chapter_records). The caller threads that signal through
    # reconcile_qids -> write_split_outputs via `extra_reasons`.
    return sorted(set(provs)), model_q_no, False


def _build_q_no_anchors(rec: dict, qn: int, anchors: dict,
                        source_pages: list) -> dict:
    """Build the q_no_anchors vector for one record. Only fields that
    are actually populated are present; missing anchors are absent
    (not null), which matches the design's intent -- a consumer can
    distinguish 'no anchor' from 'anchor present but null'."""
    provs, model_q_no, disagree = _collect_provs(rec)
    if model_q_no is None:
        model_q_no = qn
    out = {
        "model_q_no": int(model_q_no),
        "model_q_no_provs": provs,
        "model_q_no_disagree": bool(disagree),
    }
    for name, payload in anchors.items():
        if payload:
            out[name] = payload
    if source_pages:
        out["section_position"] = {
            "kind": "page_set",
            "pages": sorted(set(int(p) for p in source_pages)),
        }
    out["provenance_notes"] = provs[:]  # short-form mirror for the design spec
    return out


# ---------------------------------------------------------------------------
# 5. reconcile_qids -- the chapter-close observation step.
# ---------------------------------------------------------------------------

def reconcile_qids(chapter_records: dict, qn_source_pages: dict,
                   pdf_path: str, page_files, subject: str,
                   chapter_no: int) -> dict:
    """Walk the chapter's pages, harvest every printed anchor, grade
    every record, and split the chapter_records dict into
    (kept_records, unresolved_records). Kept records have a non-null
    q_id_grade; unresolved records go to unresolved_qids.jsonl only.

    This is the OBSERVATION step: it does NOT call Gemini and does NOT
    modify any field that the extraction loop uses. It only:
      1. assigns q_id_grade (one of 4) to every record
      2. attaches a q_no_anchors dict to every record
      3. removes UNRESOLVED records from chapter_records (in-place,
         modifying the dict the caller passes in)
      4. returns a dict {qn: rec} for the UNRESOLVED records, so the
         caller can write them to unresolved_qids.jsonl with full
         provenance.
    """
    qn_set = set(chapter_records)
    per_qn_anchors = _harvest_anchors(chapter_records, qn_source_pages,
                                      pdf_path, page_files)
    kept: dict = {}
    unresolved: dict = {}
    for qn, rec in chapter_records.items():
        anchors = per_qn_anchors.get(qn, {})
        # Convert qn_source_pages[qn] (set in the live pipeline) -> list
        sp = qn_source_pages.get(qn) or []
        if isinstance(sp, set):
            sp = sorted(sp)
        anchors_full = _build_q_no_anchors(rec, qn, anchors, sp)
        grade = _grade_record(anchors)
        # The design lists 4 UNRESOLVED conditions; only one is
        # observable in Phase 1 without modifying the loop:
        #   no_anchor_at_all -- when a record is in chapter_records but
        #   the model-only q_no is non-printable AND no answer key row
        #   names it AND no solution header names it. To avoid false
        #   positives (a record whose stem is split across a page that
        #   the visitor can't decode), we mark such records UNRESOLVED
        #   ONLY when:
        #     * the record has zero printed anchors AND
        #     * the record is missing at least one of question_text,
        #       options, correct_option, solution_text
        #   (i.e. it's already on the export-gate's "incomplete" list)
        # This matches the design's spirit without flagging every
        # well-extracted record as UNRESOLVED just because the text
        # layer is silent on its q_no.
        if grade == "PROVISIONAL":
            if _record_mostly_empty(rec) and not anchors:
                rec["q_id_grade"] = "UNRESOLVED"
                rec["q_no_anchors"] = anchors_full
                rec.setdefault("_unresolved_reason", "no_anchor_at_all")
                unresolved[qn] = rec
                continue
        # Default: keep as the graded record
        rec["q_id_grade"] = grade
        rec["q_no_anchors"] = anchors_full
        kept[qn] = rec
    # In-place: the caller reads chapter_records after this call
    chapter_records.clear()
    chapter_records.update(kept)
    return {
        "unresolved": unresolved,
        "kept": kept,
        "per_qn_anchors": per_qn_anchors,
    }


def _record_mostly_empty(rec: dict) -> bool:
    """True when the record has none of the four content fields. Used
    to gate the "no_anchor_at_all" -> UNRESOLVED escalation (see
    reconcile_qids)."""
    return not any([
        (rec.get("question_text") or "").strip(),
        rec.get("options"),
        rec.get("correct_option"),
        (rec.get("solution_text") or "").strip(),
    ])


# ---------------------------------------------------------------------------
# 6. Per-file record builders -- emit one strictly-separated row per
#    (chapter, q_no) for the three split files, plus the support files.
# ---------------------------------------------------------------------------

def _source_pages_for(qn: int, qn_source_pages: dict) -> list:
    sp = qn_source_pages.get(qn) or []
    if isinstance(sp, set):
        return sorted(sp)
    return sorted(int(p) for p in sp)


def _build_question_row(qn: int, rec: dict, chapter_id: str, subject: str,
                        chapter_no: int, image_files: dict) -> dict:
    q_id = f"{subject}-{chapter_no:03d}-{int(qn):03d}"
    options = rec.get("options") or {}
    option_rows = []
    for letter in ("A", "B", "C", "D"):
        text = options.get(letter, "") if isinstance(options, dict) else ""
        opt_imgs = []
        if isinstance(image_files.get("option"), dict):
            opt_imgs = [
                {"file": f, "source_pages": []} for f in
                (image_files["option"].get(letter) or [])
            ]
        option_rows.append({
            "id": letter,
            "text": text or "",
            "images": opt_imgs,
        })
    question_images = [
        {"file": f, "source_pages": []} for f in
        (image_files.get("question") or [])
    ]
    tables = rec.get("tables") or []
    out = {
        "q_id": q_id,
        "chapter_id": chapter_id,
        "subject": subject,
        "chapter_no": int(chapter_no),
        "q_no": int(qn),
        "q_id_grade": rec.get("q_id_grade", "PROVISIONAL"),
        "q_no_anchors": rec.get("q_no_anchors", {}),
        "question_text": rec.get("question_text") or "",
        "options": option_rows,
        "question_images": question_images,
        "tables": tables,
        "source_pages": _source_pages_for(qn, rec.get("_qn_source_pages") or {}),
    }
    out["extraction_status"], missing = _classify_question_completeness(rec)
    if missing:
        out["missing_fields"] = missing
    return out


def _build_answer_row(qn: int, rec: dict, chapter_id: str, subject: str,
                      chapter_no: int) -> dict:
    q_id = f"{subject}-{chapter_no:03d}-{int(qn):03d}"
    correct = rec.get("correct_option")
    prov = (rec.get("_prov") or {}).get("correct_option")
    out = {
        "q_id": q_id,
        "chapter_id": chapter_id,
        "subject": subject,
        "chapter_no": int(chapter_no),
        "q_no": int(qn),
        "correct_option": correct,
        "correct_option_prov": prov,
        "q_id_grade": rec.get("q_id_grade", "PROVISIONAL"),
        "q_no_anchors": rec.get("q_no_anchors", {}),
        "source_pages": _source_pages_for(qn, rec.get("_qn_source_pages") or {}),
    }
    out["extraction_status"], missing = _classify_answer_completeness(rec)
    if missing:
        out["missing_fields"] = missing
    return out


def _build_solution_row(qn: int, rec: dict, chapter_id: str, subject: str,
                        chapter_no: int, image_files: dict) -> dict:
    q_id = f"{subject}-{chapter_no:03d}-{int(qn):03d}"
    tables = rec.get("tables") or []
    sol_imgs = [
        {"file": f, "source_pages": []} for f in
        (image_files.get("solution") or [])
    ]
    prov = (rec.get("_prov") or {}).get("solution_text")
    out = {
        "q_id": q_id,
        "chapter_id": chapter_id,
        "subject": subject,
        "chapter_no": int(chapter_no),
        "q_no": int(qn),
        "solution_text": rec.get("solution_text") or "",
        "tables": tables,
        "solution_images": sol_imgs,
        "solution_prov": prov,
        "q_id_grade": rec.get("q_id_grade", "PROVISIONAL"),
        "q_no_anchors": rec.get("q_no_anchors", {}),
        "source_pages": _source_pages_for(qn, rec.get("_qn_source_pages") or {}),
    }
    out["extraction_status"], missing = _classify_solution_completeness(rec)
    if missing:
        out["missing_fields"] = missing
    return out


def _classify_question_completeness(rec: dict) -> tuple:
    """(status, missing_fields). COMPLETE if stem + 4 options all
    populated; otherwise INCOMPLETE with a missing_fields list."""
    missing = []
    if not (rec.get("question_text") or "").strip():
        missing.append("question_text")
    opts = rec.get("options") or {}
    if not isinstance(opts, dict) or len(opts) < 4:
        missing.append("options")
    else:
        for letter in ("A", "B", "C", "D"):
            if not str(opts.get(letter, "") or "").strip():
                missing.append("options")
                break
    if not rec.get("tables") and not missing:
        # tables are not REQUIRED (per design), so don't flag
        pass
    if missing:
        return "INCOMPLETE", missing
    return "COMPLETE", []


def _classify_answer_completeness(rec: dict) -> tuple:
    if not (rec.get("correct_option") or "").strip():
        return "INCOMPLETE", ["correct_option"]
    return "COMPLETE", []


def _classify_solution_completeness(rec: dict) -> tuple:
    if not (rec.get("solution_text") or "").strip():
        return "INCOMPLETE", ["solution_text"]
    return "COMPLETE", []


# ---------------------------------------------------------------------------
# 7. Unresolved-qid and orphan row builders
# ---------------------------------------------------------------------------

def _build_unresolved_qid_row(qn: int, rec: dict, chapter_id: str,
                              subject: str, chapter_no: int) -> dict:
    q_id = f"{subject}-{chapter_no:03d}-{int(qn):03d}"
    reason = rec.get("_unresolved_reason", "no_anchor_at_all")
    if reason not in UNRESOLVED_REASONS:
        reason = "no_anchor_at_all"
    # available_passes: a snapshot of which pass-populated fields exist
    prov = rec.get("_prov") or {}
    available = {}
    field_to_pass = {
        "question_text": "Q_PASS",
        "options": "Q_PASS",
        "correct_option": "A_PASS",
        "solution_text": "S_PASS",
    }
    for field, default_pass in field_to_pass.items():
        present = bool({
            "question_text": rec.get("question_text"),
            "options": rec.get("options"),
            "correct_option": rec.get("correct_option"),
            "solution_text": rec.get("solution_text"),
        }.get(field))
        if present:
            available[default_pass.split("_")[0] + "_PASS"] = {
                "had_item": True,
                "q_no": int(qn) if rec.get("q_no") is None else
                (int(rec["q_no"]) if str(rec["q_no"]).isdigit() else None),
                "fields": [field],
            }
    return {
        "q_id": q_id,
        "chapter_id": chapter_id,
        "subject": subject,
        "chapter_no": int(chapter_no),
        "q_no": int(qn),
        "kind": "unresolved_qid",
        "reason": reason,
        "q_no_anchors": rec.get("q_no_anchors", {}),
        "available_passes": available,
        "source_pages": _source_pages_for(qn, rec.get("_qn_source_pages") or {}),
    }


def _build_orphan_row(orph: dict, chapter_id: str, subject: str,
                      chapter_no: int) -> dict:
    """orph is a dict the existing pipeline populates: chapter_id,
    pass, pdf_pages, new_pages, carry_q_no, cut_part, last_qn_in_batch,
    reason, item. We add the q_id-less fragment's content to a
    truncated snippet (never the full item) so a human reviewer can
    see what was at stake."""
    item = orph.get("item") or {}
    snippet = ""
    for f in ("question_text", "options", "correct_option", "solution_text"):
        v = item.get(f)
        if v:
            snippet = (str(v) if not isinstance(v, str) else v)[:600]
            if snippet:
                break
    return {
        "subject": subject,
        "chapter_id": chapter_id,
        "source_pages": sorted(set(
            int(p) for p in (orph.get("pdf_pages") or orph.get("new_pages") or [])
            if str(p).lstrip("-").isdigit()
        )),
        "pass": orph.get("pass"),
        "reason": orph.get("reason", "q_id_unresolved"),
        "fragment": snippet,
        "carry_q_no": orph.get("carry_q_no"),
        "cut_part": orph.get("cut_part"),
        "last_qn_in_batch": orph.get("last_qn_in_batch"),
    }


# ---------------------------------------------------------------------------
# 8. write_split_outputs -- the per-chapter writer.
# ---------------------------------------------------------------------------

def write_split_outputs(*, chapter_id: str, subject: str, chapter_no: int,
                        chapter_records: dict, image_files_by_q: dict,
                        qn_source_pages: dict, orphans: list,
                        chapter_unresolved_images: list,
                        pdf_path: str, page_files,
                        reconciled: dict,
                        output_root) -> dict:
    """Write all seven per-chapter files atomically. chapter_completeness.json
    is written LAST as the "this chapter's split is fully on disk" signal.
    Returns the chapter_completeness.json content (the per-chapter summary
    the design requires)."""
    output_root = Path(output_root)
    chapter_dir = output_root / "split" / subject / chapter_id
    chapter_dir.mkdir(parents=True, exist_ok=True)

    # Copy qn_source_pages into each record so the row builders can read it
    # without a separate lookup. This is a per-record decoration done here
    # (not in reconcile_qids) so the original chapter_records is untouched
    # when the live pipeline continues to use it for build_final_question.
    for qn, rec in chapter_records.items():
        rec.setdefault("_qn_source_pages", qn_source_pages.get(qn) or set())

    # Build the rows in deterministic order
    qns = sorted(chapter_records)
    image_files_by_q = image_files_by_q or {}
    question_rows = [
        _build_question_row(qn, chapter_records[qn], chapter_id, subject,
                            chapter_no, image_files_by_q.get(qn, {}))
        for qn in qns
    ]
    answer_rows = [
        _build_answer_row(qn, chapter_records[qn], chapter_id, subject,
                          chapter_no)
        for qn in qns
    ]
    solution_rows = [
        _build_solution_row(qn, chapter_records[qn], chapter_id, subject,
                            chapter_no, image_files_by_q.get(qn, {}))
        for qn in qns
    ]

    # Atomic writes (order: data files first, completeness.json LAST).
    # A crash mid-write leaves either the previous chapter's files or
    # a fresh partial set; the completeness.json absence is the
    # downstream signal "this chapter's split is not fully on disk yet".
    _atomic_jsonl_write(chapter_dir / "questions.jsonl", question_rows)
    _atomic_jsonl_write(chapter_dir / "answers.jsonl", answer_rows)
    _atomic_jsonl_write(chapter_dir / "solutions.jsonl", solution_rows)

    # Unresolved q_ids: records reconcile_qids removed
    unresolved_dict = (reconciled or {}).get("unresolved") or {}
    unresolved_rows = [
        _build_unresolved_qid_row(qn, rec, chapter_id, subject, chapter_no)
        for qn, rec in sorted(unresolved_dict.items())
    ]
    _atomic_jsonl_write(chapter_dir / "unresolved_qids.jsonl", unresolved_rows)

    # Orphans: chapter-scoped view of the existing in-memory orphan list
    orphan_rows = [
        _build_orphan_row(o, chapter_id, subject, chapter_no)
        for o in (orphans or [])
    ]
    _atomic_jsonl_write(chapter_dir / "orphans.jsonl", orphan_rows)

    # Image manifest: chapter-scoped cross-reference of every owned image
    # file. Built from the same image_files_by_q dict the master file uses,
    # so the chapter-scoped manifest is a VIEW, not a separate source of
    # truth (the existing data/image_ownership.jsonl remains the global
    # source of truth per the user's signed-off design decision).
    image_manifest_rows = []
    for qn in qns:
        entry = image_files_by_q.get(qn) or {}
        for f in (entry.get("question") or []):
            image_manifest_rows.append({
                "q_id": f"{subject}-{chapter_no:03d}-{int(qn):03d}",
                "type": "QUESTION",
                "option_letter": None,
                "file": f,
                "source_pages": _source_pages_for(qn, qn_source_pages),
            })
        for f in (entry.get("solution") or []):
            image_manifest_rows.append({
                "q_id": f"{subject}-{chapter_no:03d}-{int(qn):03d}",
                "type": "SOLUTION",
                "option_letter": None,
                "file": f,
                "source_pages": _source_pages_for(qn, qn_source_pages),
            })
        for letter, files in (entry.get("option") or {}).items():
            for f in (files or []):
                image_manifest_rows.append({
                    "q_id": f"{subject}-{chapter_no:03d}-{int(qn):03d}",
                    "type": "OPTION",
                    "option_letter": str(letter).upper(),
                    "file": f,
                    "source_pages": _source_pages_for(qn, qn_source_pages),
                })
    _atomic_jsonl_write(chapter_dir / "image_manifest.jsonl", image_manifest_rows)

    # Completeness summary -- the master per-chapter report
    grade_counts = {g: 0 for g in ALLOWED_Q_ID_GRADES}
    extraction_counts = {"COMPLETE": 0, "INCOMPLETE": 0}
    for r in question_rows + answer_rows + solution_rows:
        g = r.get("q_id_grade")
        if g in grade_counts:
            grade_counts[g] += 1
        es = r.get("extraction_status")
        if es in extraction_counts:
            extraction_counts[es] += 1
    pass_summary: dict = {}
    for r in question_rows + answer_rows + solution_rows:
        for prov_label in (r.get("q_no_anchors") or {}).get("model_q_no_provs") or []:
            pass_summary[prov_label] = pass_summary.get(prov_label, 0) + 1

    completeness = {
        "chapter_id": chapter_id,
        "subject": subject,
        "chapter_no": int(chapter_no),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),

        "question_records": len(question_rows),
        "answer_records": len(answer_rows),
        "solution_records": len(solution_rows),
        "image_manifest_records": len(image_manifest_rows),

        "incomplete_questions": sum(
            1 for r in question_rows
            if r.get("extraction_status") == "INCOMPLETE"
        ),
        "incomplete_answers": sum(
            1 for r in answer_rows
            if r.get("extraction_status") == "INCOMPLETE"
        ),
        "incomplete_solutions": sum(
            1 for r in solution_rows
            if r.get("extraction_status") == "INCOMPLETE"
        ),

        "unresolved_qid_count": len(unresolved_rows),
        "unresolved_qid_q_nos": sorted(int(qn) for qn in unresolved_dict),

        "orphan_count": len(orphan_rows),
        "unresolved_image_count": len(chapter_unresolved_images or []),

        "q_id_grade_counts": grade_counts,
        "extraction_status_counts": extraction_counts,
        "pass_provenance_summary": pass_summary,

        # Phase-2 hook plan: the missing anchors from design doc §3.1
        # that the Phase-1 grader does not yet observe (would require
        # read-only observation hooks in process_pdf). Documented here
        # so the next change can lift them without re-reading the
        # design doc.
        "phase2_pending_anchors": {
            "neighbor_run": "design doc §3.1: run-18 GUARD's connected-run analysis",
            "carry_forward_origin": "design doc §3.1: compute_carry() output per window",
            "ocr_stem_match": "design doc §3.1: only populated when text layer was garbled",
            "ocr_solution_header_match": "design doc §3.1: only populated when text layer was garbled",
        },
    }
    _atomic_json_write(chapter_dir / "chapter_completeness.json", completeness)
    return completeness
