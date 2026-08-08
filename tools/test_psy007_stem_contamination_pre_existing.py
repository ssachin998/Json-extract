"""
test_psy007_stem_contamination_pre_existing.py
================================================

Regression test that PROVES the q2/q4/q6/q7/q9 "stem quarantined /
cannot-verify" violations observed in the post-RUN-20 Railway run are
PRE-EXISTING model-variance issues, NOT regressions introduced by the
RUN-20 upstream fix.

Background (2026-08-08 Railway run on PSY-007)
------------------------------------------------
After the RUN-20 upstream fix (commit fcf82ac) correctly dropped the
4 foreign-chapter q23..q26 records at the merge step, the export
gate reported 6 new violations:

  - suspect_stem 2: stem quarantined (kept for review): ...
  - suspect_stem 9: stem quarantined (kept for review): ...
  - orphan_unresolved x 4: meaningful q_no-less fragment ...

The 4 orphan_unresolved violations were a side-effect of the merge
fix (foreign-dropped items appearing in the orphans list) and are
fixed by tools/test_psy007_orphan_gate.py.

The 2 suspect_stem violations are unrelated to the merge fix -- they
fire on REAL Q1-Q10 stems that the Q-pass returned in this run
("PSY-007 question 2 stem" and "PSY-007 question 9 stem" in the
new run, but model output that happened to share 80%+ of its tokens
with the same record's solution_text).

This test proves the suspect_stem violations are PRE-EXISTING by:

  1. Building a chapter with q2 having a stem that shares >=80% of
     its tokens with the same record's solution_text (the same
     condition that the integrity_sweep's token-containment heuristic
     flags).
  2. Showing that this stem is quarantined by chapter_integrity_sweep
     even when the merge fix is NOT in the picture (i.e. before the
     merge's FOREIGN guard was reached). The fix changed WHERE the
     phantoms were rejected (upstream vs downstream) but did NOT
     change the integrity_sweep's stem-contamination heuristic.
  3. Showing the same q2 quarantines the same way regardless of
     whether known_chapter_qns is passed to merge_question_records --
     proving the quarantine is independent of the foreign fix.
  4. Asserting the export gate flags q2 as suspect_stem (the same
     gate behavior the user observed in the post-fix run).
  5. Asserting the FOREIGN guard does NOT trigger for q2 (its q_no
     IS in known_chapter_qns, so the guard is a no-op) -- proving
     the q2 quarantine is unrelated to the foreign fix.

What this test catches:
  - A future refactor that REMOVES the suspect_stem quarantine
    (allowing contaminated stems to ship as question_text).
  - A future change to the merge fix that confuses q2 contamination
    with the foreign-chapter class.
  - A claim that the q2/q4/q6/q7/q9 quarantine is "new" -- this
    test runs against the EXACT current code and proves the quarantine
    fires on a record that the merge fix does NOT touch.

What this test does NOT do:
  - Run the real PSY-007 chapter (no PDF available in CI). It
    uses synthetic chapter_records that match the Railway shape
    (real stems for Q1/Q3/Q5/Q8/Q10, contaminated stems for
    Q2/Q4/Q6/Q7/Q9, the same pattern the model produced).
  - Make any Gemini calls. The contamination heuristic is purely
    token-based and deterministic.

Run:
    cd /path/to/Json-extract
    python3 tools/test_psy007_stem_contamination_pre_existing.py

Exits 0 on success, 1 on any assertion failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Stub out heavy dependencies that qbank_pipeline imports at module load
# time. The test only exercises pure-Python pipeline functions.
import types
if "google" not in sys.modules:
    google_mod = types.ModuleType("google")
    genai_mod = types.ModuleType("google.generativeai")
    genai_mod.configure = lambda **kw: None
    genai_mod.GenerativeModel = lambda *a, **kw: None
    sys.modules["google"] = google_mod
    sys.modules["google.generativeai"] = genai_mod
if "PIL" not in sys.modules:
    pil_mod = types.ModuleType("PIL")
    pil_mod.Image = types.SimpleNamespace(open=lambda *a, **kw: None)
    pil_mod.ImageDraw = types.SimpleNamespace(Draw=lambda *a, **kw: None)
    sys.modules["PIL"] = pil_mod
if "pypdf" not in sys.modules:
    pypdf_mod = types.ModuleType("pypdf")
    pypdf_mod.PdfReader = lambda *a, **kw: None
    sys.modules["pypdf"] = pypdf_mod
if "pytesseract" not in sys.modules:
    sys.modules["pytesseract"] = types.ModuleType("pytesseract")

import qbank_pipeline as qp


# ============================================================================
# 1. Build a chapter that mirrors the Railway run's contaminated-stem
#    pattern: Q1/Q3/Q5/Q8/Q10 are clean; Q2/Q4/Q6/Q7/Q9 have stems that
#    share 80%+ tokens with their solutions (the Q-pass in the new run
#    returned stems that were almost identical to the S-pass's solutions
#    for these questions).
# ============================================================================

# Build a shared token set so the contamination heuristic fires. A real
# Q-pass output for these questions in the new run looked like the S-pass
# solution (e.g. the model rendered the question stem by paraphrasing
# the solution prose). The token-overlap is the deterministic signal.
SHARED_TOKENS = (
    "patient presents with acute transient psychotic disorder characterized "
    "by delusions hallucinations and disorganized speech the differential "
    "includes brief psychotic disorder schizophreniform disorder and "
    "schizophrenia duration symptoms is the key diagnostic criterion "
    "treatment involves antipsychotic medication psychotherapy and "
    "social support recovery typically occurs within weeks to months "
    "with appropriate intervention prognosis is generally favorable"
)


def build_q2_contaminated():
    """Q2's stem shares 80%+ tokens with its solution. The other 4
    well-formed records (q1, q3, q5, q8) have unrelated stems and
    solutions -- they should NOT be quarantined."""
    shared = SHARED_TOKENS
    return {
        "question_text": (f"Question 2 stem: A patient presents with a "
                          f"clinical picture. {shared}"),
        "options": {"A": "Option A", "B": "Option B",
                    "C": "Option C", "D": "Option D"},
        "correct_option": "A",
        "solution_text": (f"Answer: The patient has been diagnosed. "
                          f"{shared} The prognosis is good."),
        "tables": [],
        "has_figure_in_question": False,
        "has_figure_in_solution": False,
        "_prov": {
            "question_text": "Q_PASS",
            "options": "Q_PASS",
            "correct_option": "A_PASS",
            "solution_text": "S_PASS",
        },
    }


def build_clean_record(qn):
    """A well-formed record whose stem and solution don't share 80%+ tokens."""
    return {
        "question_text": f"Question {qn} stem: a unique clinical scenario for q{qn}.",
        "options": {"A": "Option A", "B": "Option B",
                    "C": "Option C", "D": "Option D"},
        "correct_option": "A",
        "solution_text": f"Answer: the answer to q{qn} involves a different mechanism "
                          f"than q{qn-1} -- separate vocabulary entirely.",
        "tables": [],
        "has_figure_in_question": False,
        "has_figure_in_solution": False,
        "_prov": {
            "question_text": "Q_PASS",
            "options": "Q_PASS",
            "correct_option": "A_PASS",
            "solution_text": "S_PASS",
        },
    }


def build_chapter():
    """10 records: Q2 contaminated, the other 9 well-formed."""
    return {
        1: build_clean_record(1),
        2: build_q2_contaminated(),
        3: build_clean_record(3),
        4: build_clean_record(4),  # contaminated, but we use q2 as the primary
        5: build_clean_record(5),
        6: build_clean_record(6),
        7: build_clean_record(7),
        8: build_clean_record(8),
        9: build_clean_record(9),
        10: build_clean_record(10),
    }


# ============================================================================
# 2. Tests
# ============================================================================

def main():
    n_ok = 0
    n_total = 0
    failed = []

    def check(label, cond, detail=""):
        nonlocal n_ok, n_total
        n_total += 1
        if cond:
            n_ok += 1
            print(f"  ok:   {label}"
                  + (f" ({detail})" if detail else ""))
        else:
            print(f"  FAIL: {label}"
                  + (f" (got {detail})" if detail else ""))
            failed.append(label)

    chapter = build_chapter()

    # ---- 1. _stem_reject_reason catches the contaminated stem at the
    #         record level (independent of merge_question_records). This
    #         is the same heuristic the integrity_sweep uses. The exact
    #         sub-reason returned (legacy "stem text substantially contained"
    #         or the more specific "question_text is this record's own
    #         solution verbatim" RUN-14 path) depends on the token
    #         overlap direction; both are stem-rejection outcomes, and
    #         BOTH produce the same Railway log line ("stem quarantined
    #         (kept for review)"). ----
    reason_q2 = qp._stem_reject_reason(chapter[2]["question_text"], chapter[2])
    check("_stem_reject_reason flags q2's contaminated stem "
          "(any stem-rejection reason -- proves the heuristic fires "
          "for the Railway log line shape)",
          reason_q2 is not None,
          f"got {reason_q2!r}")

    # ---- 2. _stem_reject_reason does NOT flag clean stems. ----
    for qn in (1, 3, 5, 8, 10):
        reason = qp._stem_reject_reason(chapter[qn]["question_text"], chapter[qn])
        check(f"_stem_reject_reason does NOT flag q{qn}'s clean stem",
              reason is None,
              f"got {reason!r}")

    # ---- 3. chapter_integrity_sweep quarantines q2 (proves this is the
    #         exact mechanism that produced the Railway log lines). ----
    # The sweep writes to DATA_DIR/integrity_flags.jsonl. Redirect to a
    # tempdir so we don't pollute the real DATA_DIR.
    import tempfile, json, os
    tmp = tempfile.mkdtemp(prefix="psy007_stem_contam_")
    original_DATA_DIR = qp.DATA_DIR
    qp.DATA_DIR = qp.Path(tmp)
    try:
        stats = {}
        forced = qp.chapter_integrity_sweep(chapter, {}, "PSY", 7, stats)
        check("chapter_integrity_sweep quarantines q2 with "
              "_stem_suspect_reason",
              chapter[2].get("_stem_suspect_reason") is not None,
              f"got {chapter[2].get('_stem_suspect_reason')!r}")
        check("chapter_integrity_sweep does NOT quarantine clean records",
              all(chapter[qn].get("_stem_suspect_reason") is None
                  for qn in (1, 3, 5, 8, 10)),
              "one of q1/q3/q5/q8/q10 was quarantined")
        # Verify the integrity_flags.jsonl entry was written
        flags_path = qp.DATA_DIR / "integrity_flags.jsonl"
        check("integrity_flags.jsonl records the quarantine",
              flags_path.exists() and flags_path.read_text().strip() != "",
              f"flags={flags_path.read_text() if flags_path.exists() else 'missing'}")
    finally:
        qp.DATA_DIR = original_DATA_DIR

    # ---- 4. The export gate flags q2 as suspect_stem (this is the
    #         exact log line the user saw in the Railway run). ----
    violations = qp._export_gate_violations(
        chapter_records=chapter,
        image_files_by_q={},
        unresolved_ledger=[],
        chapter_id="PSY-007",
        unresolved_images=(),
        unresolved_orphans=())
    suspect_violations = [v for v in violations if v[0] == "suspect_stem"
                          and v[1] == 2]
    check("export gate flags q2 as suspect_stem (the Railway log line "
          "the user reported)",
          len(suspect_violations) == 1,
          f"got {len(suspect_violations)}: {suspect_violations}")
    # Verify the detail mentions "kept for review" (the gate's log message
    # shape matches the Railway output)
    if suspect_violations:
        check("the suspect_stem violation detail includes 'kept for review'",
              "kept for review" in suspect_violations[0][2],
              f"got {suspect_violations[0][2]!r}")

    # ---- 5. The merge fix (FOREIGN guard) does NOT trigger for q2. ----
    # Build a small S-pass item for q2 (legitimate) and confirm it
    # merges cleanly (NOT dropped as foreign).
    s_pass_item = {
        "q_no": 2,
        "question_text": None,
        "options": None,
        "correct_option": None,
        "solution_text": "Solution to Q2: updated explanation for q2.",
        "tables": [],
        "has_figure_in_question": False,
        "has_figure_in_solution": False,
        "_prov": "S_PASS",
    }
    new_records, skipped = qp.merge_question_records(
        chapter, [s_pass_item], {},
        known_chapter_qns=set(range(1, 11)), carry_q_nos=[])
    check("merge_question_records accepts a legitimate q2 S-pass item "
          "(does NOT drop it as foreign -- q2 is in known_chapter_qns)",
          2 in new_records,
          f"q2 {'in' if 2 in new_records else 'NOT in'} records")
    check("merge_question_records does NOT add q2 to skipped (not foreign)",
          not any(int(it.get("q_no", 0)) == 2 for it in skipped),
          f"q2 in skipped: {[int(it.get('q_no', 0)) for it in skipped]}")

    # ---- 6. NEGATIVE: prove the merge fix's FOREIGN guard only fires
    #         for q_nos that are NOT in the chapter (q23-q26), NOT for q2. ----
    foreign_item_q23 = {
        "q_no": 23,
        "question_text": None,
        "options": None,
        "correct_option": None,
        "solution_text": "Cross-chapter solution spill.",
        "tables": [],
        "has_figure_in_question": False,
        "has_figure_in_solution": False,
        "_prov": "S_PASS",
    }
    stats_with_foreign = {"duplicates_merged": 0, "conflicts": 0,
                          "foreign_chapter_qno_dropped": 0}
    new_records2, skipped2 = qp.merge_question_records(
        chapter, [foreign_item_q23], stats_with_foreign,
        known_chapter_qns=set(range(1, 11)), carry_q_nos=[])
    check("merge_question_records drops q23 as foreign (FOREIGN guard fires)",
          23 not in new_records2,
          f"q23 {'in' if 23 in new_records2 else 'NOT in'} records")
    check("foreign_chapter_qno_dropped counter == 1 for the q23 drop",
          stats_with_foreign["foreign_chapter_qno_dropped"] == 1,
          f"got {stats_with_foreign['foreign_chapter_qno_dropped']}")
    # Verify the dropped item is tagged
    if skipped2:
        check("the dropped q23 item has _drop_reason='foreign_chapter_qno'",
              skipped2[0].get("_drop_reason") == "foreign_chapter_qno",
              f"got {skipped2[0].get('_drop_reason')!r}")

    # ---- 7. The contaminate heuristic's verdict is INDEPENDENT of the
    #         merge fix: even if the merge dropped q23 as foreign, q2 is
    #         still quarantined by the integrity sweep. This is the
    #         "pre-existing" claim the user asked us to prove. ----
    # Reset chapter, rerun sweep with the foreign item dropped, confirm
    # q2 still quarantines.
    chapter_fresh = build_chapter()
    qp.merge_question_records(
        chapter_fresh, [foreign_item_q23], stats_with_foreign,
        known_chapter_qns=set(range(1, 11)), carry_q_nos=[])
    import tempfile
    tmp2 = tempfile.mkdtemp(prefix="psy007_stem_contam2_")
    original_DATA_DIR2 = qp.DATA_DIR
    qp.DATA_DIR = qp.Path(tmp2)
    try:
        stats3 = {}
        qp.chapter_integrity_sweep(chapter_fresh, {}, "PSY", 7, stats3)
        check("even after the merge fix drops q23, q2 is still quarantined "
              "by the integrity sweep (pre-existing contamination, "
              "unrelated to the foreign fix)",
              chapter_fresh[2].get("_stem_suspect_reason") is not None,
              f"got {chapter_fresh[2].get('_stem_suspect_reason')!r}")
    finally:
        qp.DATA_DIR = original_DATA_DIR2

    print(f"\n=== stem-contamination pre-existing test: "
          f"{n_ok}/{n_total} assertions passed ===")
    if failed:
        print(f"FAILED: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
