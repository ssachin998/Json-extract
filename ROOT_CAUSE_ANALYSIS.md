# Root-Cause Analysis — PSY production run (33 chapters, 434 questions)

Date: 2026-07-24. Evidence: Railway deploy log excerpt (12:40–13:11 UTC, chapters
1–21 + crash at 22), `qbank_pipeline.py` source at the time of the run, and the
reported ID lists. Every claim below is tagged **[PROVEN]** (log line and/or code
path demonstrates it), **[PATTERN]** (data signature strongly implies, needs one
visual check), or **[VERIFY]** (decisive check provided).

--------------------------------------------------------------------------------
## 1. Pipeline trace (as run)

```
PDF → pdftoppm (per chapter, 150dpi)
    → 6-page non-overlapping batches (process_pdf: range(0, len, PAGES_PER_GEMINI_CALL))
    → Gemini vision call per batch
    → JSON items
    → merge_question_records (chapter-wide dict, keyed by int(q_no))
    → image extraction per page + claim against records SO FAR (single pass)
    → build_final_question (IMG_PATH_RE validation, no content filtering)
    → questions.jsonl append
```

Field-level loss points inspected, in order:

| Stage | Can it lose `solution_text`? | Verdict |
|---|---|---|
| pdftoppm | no (rasterizes every page in range, filenames carry true page number) | clean |
| Batching | **YES** — 6-page windows have **no overlap**; Gemini sees only its own window | **LOSS POINT A** |
| Gemini response | yes (RECITATION aborts whole response; model may omit content) | **LOSS POINT B** |
| Parser (`json.loads`) | no — any parse failure raises → batch WARN (visible in log, none seen for ok batches) | clean |
| `q_no` guard in `merge_question_records` | **YES** — drops the WHOLE ITEM, including a solution-bearing fragment, when `q_no` is null/non-numeric | **LOSS POINT C** |
| Merge logic itself | no — chapter-global keyed by `q_no`; `solution_text` merges via truthy-overwrite; order-independent across batches | clean |
| Image attach | n/a for text fields | — |
| `build_final_question` / validation | no — passes `solution_text` through verbatim; only validates image *paths* | clean |
| JSONL write | no filter | clean |

--------------------------------------------------------------------------------
## 2. Root causes of the missing solutions

### RC-1 — Whole-batch RECITATION loss **[PROVEN]** → explains ch16 (and ch17's missing answers)

Log:
```
12:55:58 [WARN] Gemini call failed on PSY ch16 batch 12: Response did not finish
 normally (finish_reason=4). Likely safety-blocked or hit token limit
12:55:58 [PSY] chapter 16 ... done -> 17 questions (0 missing answer, 17 missing solution)
12:56:18 [WARN] Gemini call failed on PSY ch17 batch 0: ... finish_reason=4
12:56:24 [PSY] chapter 17 ... done -> 8 questions (8 missing answer, 0 missing solution)
```
`finish_reason=4` = RECITATION (verbatim text filter). Under the code that ran,
a failed batch was `continue`d with NO retry → all content on those chapter pages
lost. ch16 answers survived because they came from an answer-key table in a
different, surviving batch. Positions: ch16 `batch 12` ⇒ chapter pages 13–18
(absolute PDF pages = `ch16.file_start+12 .. +17`); ch17 `batch 0` ⇒ pages 1–6.
Your list has 15 IDs for ch16 while the log said 17 ⇒ 2 records were recovered by
a later reprocessing; **[VERIFY]** `grep -c '"PSY-016-' questions.jsonl` — duplicates
will confirm reprocessing happened.

### RC-2 — `q_no`-null fragment dropped WITH its content **[PROVEN]** → explains ≥1 ch4 loss, mechanism behind boundary splits

Log, immediately before "chapter 4 done (9 missing solution)":
```
12:51:21 [WARN] Gemini returned an item with no q_no, skipping:
 {'q_no': None, ..., 'solution_text': 'Option D: Displacement - Feelings that
  are connected with one person are displaced onto another person. ...
```
The item CARRIED REAL solution text and was discarded by the `q_no` guard
(`merge_question_records`). A second instance at 12:57:32 (a full "somnambulism"
QUESTION with `q_no: None`) proves it is a repeating failure mode, not a one-off.
(Per handoff rule #3 the skip itself was correct — never invent numbers — but the
content was dropped into a log line instead of being recovered.)

### RC-3 — Cross-batch split with no context **[PROVEN as mechanism]** → matches the singles (PSY-001-013; PSY-012-001 pending the ch12 anomaly check)

- Batches are hard windows `page_files[bs:bs+6]`, zero overlap ⇒ every page ≡ 6k
  boundary can split a question from its answer/solution.
- The continuation page IS sent — in the next request, WITHOUT the earlier pages.
  Gemini cannot number it ⇒ emits `q_no: null` (or omits it) ⇒ RC-2 discards it.
- The parser does NOT discard partial solutions; the merge is batch-agnostic.
  The single concrete loss point is: model-side context cut → fragment → guard.
- Singles like PSY-001-013 fit: one question sitting at a batch tail. **ch12
  anomaly:** the earlier run reported ch12 with 0 missing solutions, yet
  PSY-012-001 is missing now ⇒ its line in `questions.jsonl` was re-appended by a
  LATER run with null solution (reprocessing adds, never replaces).
  **[VERIFY]** `grep -c '"id": "PSY-012-001"' questions.jsonl` → >1 proves a
  duplicate; the older line likely still holds the solution.

### RC-4 — Source chapter gives answers without explanations **[PATTERN]** → most consistent for the consecutive blocks

PSY-004-001..009, PSY-006-001..006, PSY-011-001..005, PSY-032-001..008 are exact
consecutive runs starting at q001, and every one of those questions HAS its
correct answer (those chapters logged "0 missing answer"). Batch loss cannot
produce clean 1..N runs with answers intact; boundary splits don't prefer low
consecutive numbers. The consistent explanation: these MCQs are covered by an
answer-key table with no printed explanations (common in such books), i.e.
`solution_text` is null because THE BOOK has none — not because the pipeline lost
it. RC-2 proves ch4 also lost ≥1 REAL solution, so ch4 may be mixed.
**[VERIFY]** one look at the answer pages of ch4 q1–9 (or 1 manual Gemini call):
if no explanation text exists there, RC-4 confirmed and those IDs need NO
pipeline fix — they need "solution: none in source".

--------------------------------------------------------------------------------
## 3. Table 1 — missing solutions

| Question ID | PDF page | Gemini returned solution? | Lost stage | Root cause | Required fix |
|---|---|---|---|---|---|
| PSY-016-001..013, 016, 017 | chapter pp. 13–18 (= file_start+12..+17) | NO — response aborted (finish_reason=4) [PROVEN] | Gemini batch call | RC-1 whole-batch RECITATION, no retry in old code | Page-by-page retry (implemented) + reprocess ch16 |
| PSY-001-013 | single question at a batch tail (ch1) | partial/none surviving [PATTERN] | `q_no` guard at merge entry | RC-3 boundary split → RC-2 fragment drop | 2-page batch overlap + orphan salvage (implemented) + reprocess ch1 |
| PSY-012-001 | ch12 (see anomaly note) | unknown | — | **[VERIFY]** duplicate-line check above; likely overlap-loss in re-run | same as above |
| PSY-004-001..009 | answer section (~pp. 56–66 window) | YES for ≥1 fragment [PROVEN]; rest [PATTERN] | `q_no` guard (RC-2) + possibly source (RC-4) | RC-2 proven for the "Displacement" fragment; RC-4 suspected for the consecutive block | overlap + orphan salvage (implemented); visual check of ch4 answer pages decides RC-4 share |
| PSY-006-001..006 | answer-key area (ch6) | NO evidence any was emitted | likely nothing lost | RC-4 [PATTERN] | verify source pages; no code fix if confirmed |
| PSY-011-001..005 | answer-key area (ch11) | same | same | RC-4 [PATTERN] | same |
| PSY-032-001..008 | answer-key area (ch32, extracted in final run) | same | same | RC-4 [PATTERN] | same |

--------------------------------------------------------------------------------
## 4. Root causes of the unmatched images (Table 2)

Code facts [PROVEN]: (a) every non-watermark image ≥5000px on every page is
extracted — including illustrations that belong to NO question; (b) claiming is
single-pass: a page's images are offered ONLY to questions already in
`chapter_records` that Gemini flagged `has_figure_*` and that lack that kind yet;
(c) there is NO later re-check → a figure whose owning question is introduced by
a LATER batch can never be claimed; (d) if Gemini never sets the flag (imaging
questions it "read past"), no claimant can ever exist; (e) watermark detection
samples only the first 30 pages — a recurring decoration that starts late in the
book is never excluded.

| Image file | PDF page | Image type | Expected owner | Why unmatched | Required fix |
|---|---|---|---|---|---|
| PSY-p56-127.webp | 56 (ch4 window) | textbook illustration — likely the 720×980 grayscale portrait the handoff documented as a REAL (non-watermark) object [PATTERN, open file to confirm] | none — not an MCQ figure | extracted correctly; no flagged claimant exists by design | quarantine + manifest (implemented); no further fix |
| PSY-p62-141.webp | 62 (ch4/5 window) | textbook illustration [PATTERN] | none | same as above | same |
| PSY-p126-279.webp | 126 (ch9 Dementia window, log 12:53:31) | probable MCQ figure (imaging question) or chapter illustration [VERIFY: open file] | a ch9 question IF the file shows a scan | owner never received `has_figure_in_question=true` from Gemini (flag miss) OR illustration with no owner | manifest (implemented) + manual attach if it is an MCQ figure |
| PSY-p272-596.webp | 272 (ch22 window) | content image [VERIFY: open file] | a ch22 question if figure | claimant absent at extraction time (owner batch later) or flag miss | second-pass matching (implemented) covers the "owner later" case |
| PSY-p272-970.webp | 272 (ch22 window) | recurring object #970 — the same object aliased twice on p264 (earlier crash); now seen on ≥2 pages ⇒ recurring decorative element/stamp more likely than an MCQ figure [PATTERN] | likely none | recurs late in book ⇒ outside the 30-page watermark sample; extracted, never claimed | manifest (implemented); optional: extend watermark sampling beyond page 30 |

Component blame summary: parser NO, validation NO, merge NO (for images),
alias-dedupe NO (it prevented the crash; unrelated to ownership). YES:
claim-time ordering (single pass), Gemini flag misses, watermark-sample scope,
and content that simply has no owner (design gap = nowhere to file it).

--------------------------------------------------------------------------------
## 5. Cross-batch verdict (your 6 questions)

1. *Does batching assume atomic containment within one call?* YES at the
   model-call level (hard 6-page windows); the merge was batch-agnostic, but the
   model never saw both sides of a cut.
2. *Can a question/solution span two batches?* YES — any page ≡ 6k boundary.
3. *Is the continuation page ignored?* Not ignored — sent without context, so it
   comes back as `q_no: null` fragments.
4. *Does the parser discard incomplete solutions?* NO — truthy-merge keeps any
   partial text.
5. *Does merge fail on the second half?* It never receives it — dropped upstream
   at the `q_no` guard (RC-2).
6. *Could this explain ch1/4/6/11/12/16/32?* ch16 = RC-1 (batch abort).
   ch4 = RC-2 proven + RC-4 suspected. ch1-013 / ch12-001 = boundary-split
   signature. ch6/11/32 consecutive blocks = RC-4 signature (source has no
   explanations), not boundary loss.

--------------------------------------------------------------------------------
## 6. Implemented fixes (same commit as this document)

1. **2-page batch overlap** (`BATCH_OVERLAP_PAGES = 2`): windows step by 4, so
   every interior page appears in TWO calls with full context — any split ≤2
   pages is seen WHOLE in at least one call. Cost: ~+50% calls/book (~60→~90,
   far under the 950/day cap). Not "just bigger batches": page count per call
   stays 6.
2. **Idempotent merge for tables** (dedupe by markdown) — overlap re-extraction
   can no longer duplicate tables.
3. **Orphan salvage** — items Gemini returns with null/invalid `q_no` are no
   longer log-and-forget: they're appended to `data/orphans.jsonl` with their
   chapter, batch offset and exact PDF pages, and the prompt now explicitly asks
   Gemini to EMIT numberless continuation fragments as `q_no: null` items so they
   reach the salvage file instead of vanishing inside the model.
4. **Second-pass image claiming** at chapter end — figures extracted before
   their owner question's batch are retried against the complete chapter records;
   permanent leftovers are logged to `data/unmatched_images.jsonl` (the locked
   assets/questions/{SUBJECT}/ convention is unchanged).
5. Image extraction skips overlap pages (`pages_imaged`) — no duplicate work.

Recovery path for the already-damaged chapters (needs state surgery, do after
the book finishes): remove PSY-001/004/012/016 (and 006/011/032 ONLY if the
visual RC-4 check shows explanations DO exist) from `chapters_done`, delete their
lines from questions.jsonl, re-run. The new code re-extracts with overlap+retry.
