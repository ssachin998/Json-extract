# Phase 2 + 3 Test Report: Split-Output Anchors + Rollout Protocol

**Date:** 2026-08-08
**Branch:** `arena/019fe0d7-json-extract`
**Scope:** Phase 2 (the two pending non-printed anchors documented in the
Phase 1 report §6 and design doc §3.1) + Phase 3 (the rollout-to-all-chapters
playbook the Phase 1 report §9 deferred pending user sign-off).

---

## 0. Status summary

| Item | Status |
|---|---|
| Phase 1 reviewable on real PDF | ✅ Signed off in Phase 1 report §9 (deferred Phase 2 + 3) |
| Phase 2 anchor capture hooks in `process_pdf` | ✅ Read-only observation (~30 lines, no behavior change) |
| Phase 2 grader promotion (PROVISIONAL → RESOLVED) | ✅ Grader accepts `chapter_anchor_observations` |
| Phase 2 `q_no_anchors` injection (neighbor_run, carry_forward_origin) | ✅ Both keys surface in every split record's `q_no_anchors` |
| Phase 2 synthetic test | ✅ `tools/test_phase2_anchors.py` — **31/31 assertions** |
| All existing test suites still green | ✅ 9 suites, 171/171 assertions (no regression) |
| Phase 3 rollout protocol | ✅ This document, §4 below |
| Real Railway re-run on PSY-007 to confirm post-Phase-2 behavior | ⏳ Awaiting next deploy |

---

## 1. What Phase 2 implements

### 1.1 The two new anchors (design doc §3.1)

| Anchor | Shape | Source |
|---|---|---|
| `neighbor_run` | `{size, first, last, near_chapter_max}` | `run-18 GUARD`'s `trusted_qnos` set in the Q-pass block (the connected-run analysis at `qbank_pipeline.py` ~L5864) |
| `carry_forward_origin` | `{from_window: [int], cut_part: "question"/"options"/"solution"/null}` | `compute_carry()` output (per-pass carry-forward state at `qbank_pipeline.py` ~L2757) |

Both are read-only observations of transient local variables that
already existed in `process_pdf`; Phase 2 captures them into a
`chapter_anchor_observations` dict and threads it into
`split_outputs.reconcile_qids(chapter_anchor_observations=...)`.

### 1.2 Grader promotion (design doc §2 pseudocode)

Per the design doc §2, a single-anchor record is graded `RESOLVED`
(not `PROVISIONAL`) when EITHER:
- it is in a trusted connected-run from the run-18 GUARD, OR
- it has a valid carry-forward origin from `compute_carry()`.

Phase 2's `_grade_record(anchors, has_neighbor_run=..., has_carry_origin=...)`
implements this exactly. The pre-Phase-2 behavior is preserved when
the caller passes `chapter_anchor_observations=None` (Phase 1
callers) or `{}` (empty dict).

### 1.3 What Phase 2 does NOT do (Phase 2 is strictly additive)

- **No Gemini calls**: both anchors are derived from in-memory
  state that already existed in the Q-pass block.
- **No behavior change for records that already have ≥2 printed
  anchors** (they stay `RESOLVED_ANCHORED`).
- **No behavior change for records with 0 printed anchors and no
  Phase 2 origin** (they stay `PROVISIONAL`).
- **No change to the master `data/questions.jsonl`** — the split
  layer is still observation-only on the existing `chapter_records`
  dict (try/except at the call site in `qbank_pipeline.py`).
- **No change to the export gate**, the CRITIQUE pass, the rescue
  pass, or any other upstream/downstream machinery.

---

## 2. Code changes (commit `0540106` was the previous one; this
   Phase-2+3 commit follows)

### 2.1 `qbank_pipeline.py` (~30 lines added)

1. **`chapter_anchor_observations = {"neighbor_runs": [], "carry_forwards": []}`**
   initialised at the top of the per-chapter loop in `process_pdf`
   (right next to the existing `carry_by_pass` / `carry_trackers` /
   `ledger_rows` / `solutions_section_seen` block).
2. **Neighbor-run capture**: after the Q-pass GUARD builds
   `trusted_qnos`, the dict
   `{"window_pages": ..., "batch_qnos": ..., "runs": ..., "trusted_qnos": ..., "known_max": ...}`
   is appended to `chapter_anchor_observations["neighbor_runs"]`.
3. **Carry-forward capture**: after each `compute_carry()` call,
   `{"pass": ..., "window_pages": ..., "last_open_question": ..., "cut_part": ..., "ends_mid_content": ...}`
   is appended to `chapter_anchor_observations["carry_forwards"]`.
4. **Threading**: the new `chapter_anchor_observations` dict is passed
   to `split_outputs.reconcile_qids(chapter_records, qn_source_pages,
   pdf_path, page_files, subject, ch["chapter_no"],
   chapter_anchor_observations=chapter_anchor_observations)`.

### 2.2 `split_outputs.py` (~70 lines changed)

1. **Module docstring**: updated to mark the two Phase 2 anchors as
   now-populated, with a clear "what is still pending" note (the
   two OCR anchors, which only matter for books with garbled text
   layers — MARROW/PSY-style books do not exercise them).
2. **`reconcile_qids(...)`**: accepts the new
   `chapter_anchor_observations=None` parameter; when provided,
   precomputes `neighbor_trusted_qns` and `carry_origin` per-qn
   dicts that the loop uses to drive both the grader and the
   `q_no_anchors` injection.
3. **`_grade_record(...)`**: now takes `has_neighbor_run` and
   `has_carry_origin` flags; promotes 0-anchor + either-flag to
   `RESOLVED` (was `PROVISIONAL`), and 1-anchor + either-flag to
   `RESOLVED` (was `RESOLVED` already — no change for that case).
4. **`_build_q_no_anchors(...)`**: now takes `neighbor_run_obs` and
   `carry_origin_obs`; injects the two Phase 2 anchor dicts into
   `q_no_anchors` only when meaningful (empty observations stay
   absent — no `{}` placeholders in the consumer's diff).
5. **`phase2_pending_anchors` in `chapter_completeness.json`**:
   `neighbor_run` and `carry_forward_origin` removed (they are
   now populated); only `ocr_stem_match` and
   `ocr_solution_header_match` remain pending.

### 2.3 New test: `tools/test_phase2_anchors.py` (31 assertions, 10 sections)

| Section | What it proves | Assertions |
|---|---|---|
| A | `_grade_record`: 0 anchors + `neighbor_run` → RESOLVED | 1 |
| B | `_grade_record`: 0 anchors + `carry_origin` (with/without `neighbor_run`) → RESOLVED | 2 |
| C | `_grade_record`: 1 printed anchor + either origin → RESOLVED | 2 |
| D | `_grade_record`: 0 anchors + no origin → PROVISIONAL (Phase 1 baseline) | 1 |
| E | `_grade_record`: 1 printed anchor + no origin → RESOLVED (Phase 1 baseline) | 1 |
| F | `_build_q_no_anchors` injects `neighbor_run` (size, first, last, `near_chapter_max`) | 5 |
| G | `_build_q_no_anchors` injects `carry_forward_origin` (`from_window`, `cut_part`); no placeholder when not passed | 4 |
| H | `reconcile_qids` wires observations through to per-record grades and `q_no_anchors` | 9 |
| I | `chapter_completeness.json.phase2_pending_anchors` no longer lists the two Phase 2 anchors | 3 |
| J | `None` / `{}` observations preserve Phase 1 behavior (no surprise promotion) | 3 |

---

## 3. Test results (all suites)

```
$ for f in tools/test_*.py tools/run_split_psy007_synthetic.py; do
>   python3 "$f" 2>&1 | tail -3
> done

test_contamination_root_cause.py          17/17   (no regression)
test_phase2_anchors.py                    31/31   (NEW -- Phase 2 proof)
test_psy007_merge_q23_q26_phantom.py       4/4    (no regression)
test_psy007_orphan_gate.py                12/12   (no regression)
test_psy007_postfix_e2e.py                25/25   (no regression)
test_psy007_stem_contamination_pre_existing.py  20/20 (no regression)
test_rescue_page_routing.py               42/42   (no regression)
test_rescue_routing_integration.py        12/12   (no regression)
test_split_psy007_real_case.py            17/17   (no regression)
run_split_psy007_synthetic.py             22/22   (no regression)

TOTAL: 10 suites, 202/202 assertions
       (was 9 suites, 171/171 before Phase 2; +1 suite, +31 assertions)
```

Zero regression. The Phase 2 hook is observation-only and was designed
to preserve every Phase 1 codepath (sections D, E, J all prove the
Phase 1 baseline is intact for callers that don't pass observations).

---

## 4. Phase 3 — Rollout protocol (all 33 PSY chapters)

The Phase 1 report §9 deferred the rollout pending user sign-off on
the Phase 1 grader. Phase 2 lifts the final two anchors documented in
the design doc, and the user has now signed off on the Phase 1 results
in the Phase 2 sign-off. **Phase 3 is the playbook for going from
the single-chapter validated state (PSY-007) to all 33 chapters.**

### 4.1 Pre-flight: confirm the local environment matches the new code

```bash
cd /path/to/Json-extract
git checkout arena/019fe0d7-json-extract
git log --oneline -5
# Expected: HEAD is the Phase 2 commit (Phase 1 + Phase 2 fix +
# Phase 2 anchor test). All 10 test suites must be green:

for f in tools/test_*.py tools/run_split_psy007_synthetic.py; do
  python3 "$f" 2>&1 | tail -3 | grep -E "assertions|FAIL"
done
# Expected: every suite prints "X/X assertions passed" with no FAIL.
```

### 4.2 Per-chapter dry run (PSY-007 first, then expand)

The existing `test_v2_chapter.py` (already in the repo) is the
single-chapter smoke test. Run it on PSY-007 FIRST (it is the only
chapter with a real PDF on a local machine that has a real
`pdfs/Psychiatry_ed8.pdf`).

```bash
# 1. Single-chapter dry run on PSY-007 with the Phase 2 hooks active
python3 test_v2_chapter.py PSY 7

# 2. Verify the three contracts the Phase 2 rollout promises:
#    (a) the seven split files exist under data/split/PSY/PSY-007/
#    (b) every record's q_no_anchors has model_q_no + model_q_no_provs
#        + (optionally) neighbor_run / carry_forward_origin
#    (c) chapter_completeness.json q_id_grade_counts has no spurious
#        UNRESOLVED entries (UNRESOLVED only for the 4 documented
#        Case 2 reasons, NEVER because of a Phase 2 promotion bug)

ls qbank_output/data/split/PSY/PSY-007/
head -1 qbank_output/data/split/PSY/PSY-007/questions.jsonl | python3 -m json.tool
cat qbank_output/data/split/PSY/PSY-007/chapter_completeness.json | python3 -m json.tool
```

### 4.3 Expand to all 33 PSY chapters (per-chapter batch run)

The full pipeline (`qbank_pipeline.py` with `only_chapter_no=None`)
walks all 33 chapters of PSY in one invocation. The existing
`state.json` checkpoint + `rewrite_questions_file` crash-safe
rewrite means a partial run can be resumed at any chapter boundary
without re-doing work.

```bash
# 3. Full PSY book run -- only after PSY-007 dry run passes
python3 qbank_pipeline.py
# Expected: 33 chapters processed in order, each writing
#   data/split/PSY/PSY-NNN/{questions,answers,solutions,...}.jsonl
#   + chapter_completeness.json
#   + subjects/PSY/questions.jsonl (the bundled master)
```

### 4.4 Validate per-chapter split output (the post-Phase 2 rubric)

For every chapter, the following checks MUST hold:

```bash
# Per-chapter rubric
for ch in qbank_output/data/split/PSY/PSY-*/; do
  cid=$(basename "$ch")
  # 1. all 7 split files exist
  for f in questions.jsonl answers.jsonl solutions.jsonl \
           unresolved_qids.jsonl orphans.jsonl \
           chapter_completeness.json image_manifest.jsonl; do
    test -f "$ch/$f" || echo "MISSING: $cid/$f"
  done
  # 2. q_id_grade_counts has only the 4 documented grades
  python3 -c "
import json, sys
comp = json.load(open('$ch/chapter_completeness.json'))
grades = set(comp['q_id_grade_counts'].keys())
allowed = {'RESOLVED_ANCHORED','RESOLVED','PROVISIONAL','UNRESOLVED'}
extra = grades - allowed
if extra:
    print(f'  $cid: extra grades: {extra}')
print(f'  $cid: question_records={comp[\"question_records\"]} '
      f'grade_counts={comp[\"q_id_grade_counts\"]}')
  "
done
```

Expected: every chapter's split directory has all 7 files, only the
4 documented grades, and `q_id_grade_counts` sums to the chapter's
`question_records`.

### 4.5 Spot-check the Phase 2 anchors are populated

For chapters with Q-pass runs (which is most of them), the
synthetic test H already proved the population pattern. Spot-check
on one real chapter:

```bash
# Find a chapter with at least one RESOLVED_ANCHORED record
python3 -c "
import json
with open('qbank_output/data/split/PSY/PSY-007/questions.jsonl') as f:
    for line in f:
        r = json.loads(line)
        anchors = r.get('q_no_anchors', {})
        if 'neighbor_run' in anchors or 'carry_forward_origin' in anchors:
            print(json.dumps({
                'q_id': r['q_id'],
                'grade': r['q_id_grade'],
                'neighbor_run': anchors.get('neighbor_run'),
                'carry_forward_origin': anchors.get('carry_forward_origin'),
            }, indent=2))
            break
"
```

Expected: a record with `q_id_grade=RESOLVED` (not RESOLVED_ANCHORED)
that has at least one of the two Phase 2 anchors. The shape matches
the synthetic test H output:
- `neighbor_run.size` ≥ 1, `near_chapter_max` is True for runs that
  touch the chapter's known_max
- `carry_forward_origin.from_window` is a non-empty int list, and
  `cut_part` is one of `question` / `options` / `solution` / `null`

### 4.6 Phase 3 sign-off

The rollout is complete when:

1. ✅ All 33 PSY chapters processed (per `state.json`)
2. ✅ All 7 split files exist for every chapter
3. ✅ `q_id_grade_counts` sums to `question_records` for every chapter
4. ✅ `phase2_pending_anchors` in every `chapter_completeness.json`
   lists only the two OCR anchors (the design doc's "what is still
   pending" list)
5. ✅ The validator (`qbank_validator.py`) reports no new flag kinds
6. ✅ The dashboard / `app.py` reads the new split files without errors

If any chapter fails the rubric, the pipeline supports surgical
re-runs:

```bash
# Remove the chapter from chapters_done and re-run (state.json
# resume preserves all earlier chapters):
python3 -c "
import json
state = json.load(open('qbank_output/state.json'))
state['pdf_progress']['PSY']['chapters_done'] = [
    c for c in state['pdf_progress']['PSY']['chapters_done']
    if c != 'PSY-NNN'
]
json.dump(state, open('qbank_output/state.json','w'), indent=2)
"
python3 test_v2_chapter.py PSY NNN
```

### 4.7 What happens after the 33 PSY chapters

Once PSY is fully rolled out and signed off:

- The same protocol applies to the other 19 PDFs configured in
  `qbank_pipeline.py` (one `PDFS.append({...})` entry per subject,
  with the corresponding `pdfs/{Subject}.pdf` file).
- Each subject's `state["pdf_progress"]` key tracks its own
  `chapters_done` independently, so one subject's failures do not
  block another subject's progress.
- The split-layer is **strictly additive** per the design doc §11:
  the master `data/questions.jsonl` and `data/by_chapter/*.jsonl`
  are NEVER touched. Each subject's bundle goes under
  `qbank_output/subjects/{SUBJECT}/...` (the existing
  `build_subject_bundle` function handles this).

---

## 5. What's still pending (not Phase 2 / 3 — known future work)

These are explicitly out of scope for the current sign-off and
documented in `chapter_completeness.json.phase2_pending_anchors`:

1. **`ocr_stem_match`**: only populated when the text layer is garbled
   (scanned-only PDF). The live pipeline's OCR fallback
   (`_clean_ocr_text` / `_recover_ocr_solution_headers`) already
   handles those pages; the split layer's printed-anchor scan is the
   dominant signal for MARROW/PSY-style books.
2. **`ocr_solution_header_match`**: same as above, for the
   solution-header variant.
3. **Real-PDF PSY-007 re-run with `ec411ae` + this commit on Railway**:
   expected log signature:
   - pages sent to rescue: `[101, 102, 103]` (no 105/106/107) — routing fix preserved
   - `suspect_stem` gate violations: **1 of 5** (q1 only) — was 5 of 5 pre-fix
   - q2/q4/q6/q7/q9: well-formed records, no quarantine
   - every RESOLVED record's `q_no_anchors` has at least one of
     `neighbor_run` / `carry_forward_origin` for borderline cases
4. **Per-record `provenance_notes` cleanup** (Phase 4): currently
   `provenance_notes` is a short mirror of `model_q_no_provs`. A
   later change could add the per-field `Q_PASS` / `A_PASS` /
   `S_PASS` / `RESCUE` / `RECOVER` / `DRAIN_*` labels so consumers
   can filter by exact pass-of-origin without re-reading `_prov`.
5. **Multi-subject validation command** (Phase 4): a
   `tools/test_full_book_split.py` that walks all 33 PSY chapters
   and runs the per-chapter rubric in §4.4 against all of them in
   one go. Currently the rubric is hand-rolled per chapter.

---

## 6. Commit history (this branch, Phase 2 + 3)

```
<this-commit> Phase 2 + 3: lift neighbor_run + carry_forward_origin
            anchors (split_outputs), capture them in process_pdf,
            wire them into reconcile_qids grader, add Phase 2 test
            (31/31), and document the Phase 3 rollout playbook
0540106    Phase 2: end-to-end post-fix Railway log simulator (25/25)
ec411ae    stop using solution text to validate question stems
d55094c    Merge branch 'arena/019fe0d7-json-extract'
f5c8173    route rescue stem-recovery to QUESTION-side pages
3ba3249    Fix rescue routing (the routing fix above)
390a9fe    Fix rescue pass to use stem-only ask for contaminated stems
0a6262f    Add pre-fix 7a4bf810 Railway log analysis
b4ef628    Add RUN-20 post-fix log analysis confirming the fix
4145275    Tighten orphan-gate accounting for RUN-20 foreign-chapter drops
fcf82ac    Fix upstream root cause of PSY-007 Q23-Q26 phantom-record bug
04658ef    Fix integration bug: route solution-only Q23-Q26 to unresolved
e0e0e51    Fix Railway boot crash: add split_outputs.py to Dockerfile
6605cfd    Phase 1 test: synthetic harness with all 10 §12.1 assertions
bb84765    Phase 1: additive split-output layer
86c6692    Add split-output design doc + pipeline architecture analysis
```

---

*End of report. Awaiting real-PDF Railway re-run with this commit
on top of `ec411ae` to confirm the post-Phase-2 log signature in
production.*
