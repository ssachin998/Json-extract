# Json-extract — Full Pipeline Working Process
*(Hand-over document for technical/AI review. Written 2026-07-27, code @ commit `9dbadc6`, branch `arena/019f92d5-json-extract`. Every behavior below is verified from the actual code, not aspirational.)*

---

## 1. What the system does

Converts scanned/printed **medical MCQ textbook PDFs** (question booklets with questions, options, answer-key tables, and printed explanations) into a **single `questions.jsonl`** dataset using **Gemini Vision** (`gemini-3.1-flash-lite-preview`, free tier). One JSON object per question:

```json
{
  "id": "PSY-001-012",                       // SUBJECT-CHAPTER(3d)-Q_NO(3d)
  "subject": "PSY",
  "chapter_id": "PSY-001",
  "question":   { "text": "...", "images": [{"type": "figure", "file": "PSY/PSY-001-012_Q_01.webp"}] },
  "options":    [ {"id": "A", "text": "...", "images": []}, ... ],
  "correct_options": ["B"],
  "solution":   { "text": "...", "images": [...], "tables": [{"type": "...", "markdown": "|...|", "file": null}] },
  "tags": []
}
```

Golden rules the whole pipeline is built around:
- **Never paraphrase/summarize** — verbatim text only (enforced via prompt + post-checks).
- **Never invent a q_no** — fragments with no visible number go to an orphan pool, not guessed.
- **Never fabricate** missing values — leave null and re-ask with a small focused prompt.
- **Never silently discard** anything — every unresolved item lands in a sidecar JSONL for review.
- **Fill-only merges** — later/recovery passes may append/fill, never overwrite existing text (except provably-wrong content stripped by the integrity sweep with a ledger entry).

End goal: feed ~20 books through this, consume the JSONL in a separate exam/quiz app.

---

## 2. Components & deployment

| Component | File | Role |
|---|---|---|
| Extraction engine | `qbank_pipeline.py` (~2,580 lines) | All PDF→JSONL logic; also runnable standalone (`main()`) |
| Offline validator | `qbank_validator.py` (~1,020 lines) | Deterministic rule-based QA over the output; optional LLM-audit mode (flagged rows only) |
| One-shot healer | `fix_output.py` (~315 lines) | Evidence-gated patches P1–P13 for specific proven defects; idempotent |
| Dashboard server | `app.py` (~750 lines, Flask) | Non-technical user's ONLY interface: run/upload, validate, fix, recovery, backup/restore, zip download |
| Container | `Dockerfile` | python:3.11-slim + poppler-utils; `ENV OUTPUT_DIR=/data/qbank_output`, `CMD python3 app.py` |
| Hosting | Railway + **Volume mounted at `/data`** + public domain | State/output survive redeploys/restarts/crashes |
| LLM | google-generativeai SDK | `google-genai` style `model.generate_content(parts)`; **stateless** (continuity injected manually) |

Env vars: `GEMINI_API_KEY`, `OUTPUT_DIR` (default `./qbank_output`; `/data/qbank_output` in Docker), `PORT` (8080), optional `DRIVE_FOLDER_ID`, `DRIVE_API_KEY` (for restore-from-Drive), `GEMINI_MODEL` not env — model is a constant in `qbank_pipeline.py`.

**Hard volume guard**: if `OUTPUT_DIR` lives under `/data` but `/data` is not a real mount, the dashboard shows a red banner and **blocks** run/recover/fix/validate/restore (400) — never burn Gemini quota writing to ephemeral container fs.

---

## 3. End-to-end lifecycle (one book)

```
User taps "Run" → download PDF (Drive link or upload)
  → per chapter: [render pages → batch loop {Gemini call → merge → carry → images}
                  → orphan recovery → image claims passes 2/3/4 → failed-page drain
                  → orphan re-recovery → persist orphans → integrity sweep
                  → targeted retry → build final JSON rows → write questions.jsonl
                  → checkpoint state]
  → all chapters done → (user) 🔍 Validate → (optional) 🩹 Fix → 📦 zip download
Crash/quota-exit at ANY point → re-run RESUMES from state.json (per-chapter + calls_today)
```

---

## 4. Stage details

### 4.1 Input & orchestration (`app.py`)

- `POST /run-url`: accepts Google Drive share links (converted to `uc?export=download&id=...`), direct URLs; handles Drive's virus-scan interstitial (`confirm=` token); validates `%PDF` magic bytes; then starts the pipeline thread. Marks status=processing **before** download to prevent double-tap duplicate runs.
- `POST /run`: direct file upload.
- Both call `run_pipeline_thread(subject, pdf_path, page_offset)` which injects a one-PDF config into `PDFS` (`pipeline.PDFS[:] = [{...}]`) and calls `qbank_pipeline.main()` in a daemon thread. All progress goes to an in-memory log ring buffer shown on the dashboard (`GET /status` JSON polls).
- After `main()` returns: zero-token deterministic validation runs automatically (`run_hybrid(audit=False)`, summary logged) and `make_zip()` rebuilds the downloadable zip. Quota `SystemExit` → status "paused" (zip still rebuilt) so tomorrow's resume taps just Run.
- Other routes: `/recover` (recovery plan JSON), `/fix`, `/validate`, `/data-status` (deep fs scan), `/restore-drive` (pull a previous output folder back from Google Drive: webp→assets, listed data files→data/, state.json), `/restore-zip`, `/download` (zip of `/data` subtree).

### 4.2 PDF preparation (`process_pdf`)

1. **Watermark identification** — `find_watermark_object_id()`: scans sample pages' XObjects; an image object id reused across many pages = watermark → excluded from figure extraction.
2. **TOC parsing** — `extract_toc_chapters()`: reads printed TOC pages (default pages 1–3) via `pdftotext`, regexes chapter number/title/printed-page.
3. **Page ranges** — `compute_page_ranges()`: printed page + `page_offset` (per-book constant, typically −1 or −2 because PDF file pages lead printed pages by front matter) → contiguous `[file_start, file_end]` per chapter.
4. **Rendering** — `pdftoppm -jpeg -r 150` per chapter into `/tmp/{SUBJECT}_ch{NNN}/page-*.jpg`. 150 DPI chosen as readably-cheap; failed-page recovery re-renders targets ±1 neighbour at higher DPI.

### 4.3 Batching & continuity (`SCHEMA_PROMPT`, carry-forward)

- **Batch size** = 6 pages/call (`PAGES_PER_GEMINI_CALL`), **overlap** = 2 pages (`BATCH_OVERLAP_PAGES`) → step 4 new pages/call. Rationale: a question/solution split across a boundary is seen WHOLE (with its printed q_no) in at least one call; q_no-keyed merge makes re-extraction idempotent. Trailing all-overlap window is skipped (saves a call).
- **Stateless continuity**: every batch returns one extra control object `_batch_meta` `{last_q_no, ends_mid_content, cut_part, tail_text}`. `compute_carry()` turns that into a carry payload; `build_carry_context()` injects it as a text prefix into the NEXT call ("CONTEXT FROM PREVIOUS BATCH … use only to continue the referenced item under its original q_no").
- **Stale-carry guard #1** — `enforce_carry_expiry()`: a carry whose split never resolved within the chapter expires (banned from respawning).
- **Stale-carry guard #2** — `detect_section_boundary()`: first batch hitting the printed **Answers/Solutions section HARD-RESETS all carry context** so question-section text can never bleed into solutions and cross-merge.
- The main `SCHEMA_PROMPT` (see §6) additionally covers: one-JSON-entry-**per-row** for answer-key tables (the classic catastrophic failure: model summarizes a 20-row key as prose), null-q_no fragments ("never invent"), overlap handling, and verbatim-only rule.

### 4.4 Gemini call layer & retry ladders

Two ladders protecting two call sites:

**Ladder A — main batch loop (`call_gemini_on_pages` + `retry_batch_page_by_page`)**
1. Non-STOP finish_reason (safety/RECITATION=4/token limit) or exception → retry **each page ALONE** (a single poisoned page must not sink 6 pages' worth of data; proven in prod).
2. Page failing even alone → persisted to `state["failed_pages"]` with provenance (subject/chapter/true_page/reason) for the chapter-end drain.
3. 429/quota burst → 65 s backoff, one retry; still limited → clean save + `sys.exit(0)` (free tier: 15 RPM bursts ≠ daily cap; disambiguated by the backoff).

**Ladder B — focused asks (`gemini_json_call_splitting`)** used by targeted retry, drain, image attribution: whole set → halves → singles. Transient 500/503 → one 20 s-backoff retry; 429 → 65 s then save+exit; deterministic per-page failure costs only that page. *Origin: a 14-page whole-chapter retry failed as a unit in run-4 (recitation on one page wiped the ask).*

**Quota accounting**: `state["calls_today"]` incremented on EVERY call (main, retry ladder, drain, attribution, recovery) — a run-4 blind spot (recovery calls weren't counted) that was fixed. Daily brake `MAX_CALLS_PER_DAY = 1400` (self-imposed, ~7 % under the free 1500/day). On brake: save + exit; next run resumes mid-book. `reset_daily_counter_if_needed()` rolls the counter by server date.

**Safety settings**: `BLOCK_ONLY_HIGH` on all four categories — medical text routinely contains clinical violence/self-harm content; default thresholds false-blocked legitimate pages.

### 4.5 Merge engine (`merge_question_records`, q_no-keyed, fill-only)

- Record per q_no: `{question_text, options{A..D}, correct_option, solution_text, tables[], has_figure_*}`. New items fill nulls; non-null collisions resolved by rules:
  - **Exact/near-duplicate stems** (SequenceMatcher ≥ 0.95 on overlap re-reads) → counted, not duplicated.
  - **Answer conflicts** (same q_no, different correct_option, case-normalized) → keep FIRST, drop item, count conflict.
  - **Stem conflicts** (sim < 0.95): `_stem_payload_coherence()` scores how well each stem variant coheres with the record's own options+solution; winner kept; loser + both texts logged to `stem_conflicts.jsonl`. In fill-only (recovery) mode the existing stem NEVER loses.
  - **Stale carry-merge guard**: a "stem" that looks like solution-style prose (`looks_like_solution_style_stem`) is rejected as stem but its other fields still merge.
  - `_frag_mostly_present()` (0.85 containment) dedupes appended continuation fragments.
- **q_no = null items are NOT merged** — they go to the orphan pool with full provenance (batch window, carry owner, last q_no in batch).

### 4.6 Orphan recovery (`recover_orphans`) — confidence ladder

0. **Answer-key-table orphan** → parse markdown rows deterministically; fill-only `correct_option`s; rows that all match existing answers = consumed as "verified"; **disagreeing rows → `integrity_flags.jsonl`** (free wrong-answer alarm); key referencing only foreign q_nos → kept with `blocked_reason`, explicitly NOT merged (prevents foreign-key glue-onto-local bug).
1. **Self-labelled fragment** ("Solution to Question 3:") → parse owner from text.
2. **Carry-forward owner** captured when the fragment arrived.
3. **Last-q_no heuristic**: highest-numbered question in the same batch window missing exactly the field the fragment provides; partial owners get **append** of leading-new text.
4. Otherwise → persist to `orphans.jsonl` (AFTER the drain's second recovery pass, so healed fragments never get a stale "unresolved" entry).

### 4.7 Image/figure pipeline

- `extract_real_images()`: pypdf XObject walk per page, skip watermark id, save `webp` under `assets/questions/{SUBJECT}/` with temp names `PSY-p{page}-{seq}.webp`.
- `image_positions_on_page()`: 2D affine CTM composition (`_mat_mult`) → y-positions, so images on a page can be ordered against question numbers printed on the same page (`pdftotext`).
- **Claim pass 1** (`claim_page_images_one_to_one`, during batches): one-to-one assignment by page-order vs the questions already known; respects per-question caps.
- **Claim pass 2** (chapter end): retry leftovers now that all records exist (plates often print just before their question).
- **Claim pass 3** (0 tokens): if `pdftotext` shows EXACTLY ONE of this chapter's questions printed on the image's page → that question is owner (side picked by "fig/diagram" in stem vs solution presence); rename via `_rename_for_slot` (collision-proof `_Q_01/_Q_02…` suffixes, tiny-file guard, cap guard).
- **Claim pass 4** (Gemini, ONE image/call): `attribute_orphan_image()` returns `{q_no, slot}` or `decorative:true`; decorative → `decorative_images.jsonl`; undecided → stays unmatched; quota brake stops gracefully.
- **Guards everywhere**: `MIN_IMAGE_BYTES = 1500` (<1.5 KB webp ≈ broken crop — ref dropped everywhere, never shipped broken), `MAX_QUESTION_IMAGES = 3` (over-attribution sweep de-references extras to `unmatched` with reason — fired 4× in prod), `IMG_PATH_RE` filename validation, missing-file ref drop at build time.
- Naming convention (locked): `assets/questions/{SUBJECT}/{SUBJECT}-{CH:03d}-{Q:03d}_{Q|SOL|OPT_A..|TABLE}_{NN}.webp`.
- Still-unclaimed → `unmatched_images.jsonl` for human review (current known residual: 3 page-groups in the trial book).

### 4.8 Failed-page drain (`drain_failed_pages`)

Chapter-end second chance for recitation/refusal pages **before** targeted retry:
1. Re-render target ±1 neighbour at **higher DPI**.
2. Re-ask with a re-framed prompt (different phrasing than the first attempt — recitation triggers are prompt-sensitive).
3. If still blocked → **crop ladder**: page → horizontal halves → quarters (12 % overlap so clipped content is whole in ≥1 crop; overlap re-extraction is merge-safe because all consumers are fill-only/deduped).
4. Healed pages are removed from `state["failed_pages"]`; drained null-q_no fragments join the orphan pool and go through `recover_orphans` again.

### 4.9 Integrity sweep (`chapter_integrity_sweep`) — zero-token deterministic proofs

Runs before targeted retry so anything it strips is re-asked **in the same run**:
- **duplicated wrong-owner stems**: two records sharing a stem → coherence resolver strips the provably-foreign one.
- **foreign `Option` heads in solutions** (`_foreign_option_line`): a solution containing another question's "Option X:" rebuttal block is clipped at the foreign marker.
- **truncated solutions** (`looks_truncated_solution`): dangling-colon/mid-flow endings; **suppressed if the row has tables/images** that legitimately continue the text (book layout = "…listed below:" + table). Mid-flow truncation (trailing-space/joiner evidence) stays REAL.
- **over-attributed images** (>3 question-side) → de-reference extras with ledger entry.
- Outputs `forced_solution_qns` → targeted retry re-asks those solutions even if the 60 % gate wouldn't.
- Sweep findings → `integrity_flags.jsonl` with `kind`, evidence, `matched` bool.

### 4.10 Targeted retry (`find_incomplete_records` + `build_targeted_retry_prompt` + `targeted_retry`)

- Finds rows missing `correct_option`/`options`/`solution_text` **after** the whole chapter processed.
- **Solution gate** (`SOLUTION_GATE_MIN_SHARE = 0.6`): if ≥60 % of the chapter already has solutions, the book provably prints explanations here → the rest are extraction losses, retry-eligible. Below the gate, solution gaps are respected (answer-key-only chapters exist) — gate added after a run proved nondeterministic model drops on identical pages.
- Retry prompt = narrow, listing ONLY the specific missing fields with their page images (via Ladder B halves→singles), `TARGETED_RETRY_MAX_ROUNDS = 2` rounds.
- Fields filled are merged fill-only; sweep-forced q_nos bypass the gate.
- Remaining gaps → `still_incomplete_after_retry.jsonl` with `chapter_id` (added after the review found entries were chapter-less), consumed by the validator as `answer_key_only_suppressed` where the gate explains them.

### 4.11 Final build & persistence (`build_final_question` + chapter write)

- `sanitize_solution_text()`: strips leaked prompt/self-reference furniture, header junk; emits notes.
- `_is_printed_answer_key()`: root-level strip of printed answer-key markdown that leaked into a solution's `tables` (validator also flags leftovers as `stray_answer_key_table`; future books never carry them).
- `_dedupe_tables()`, table schema normalization, image ref validation (see §4.7).
- Rows appended to `data/questions.jsonl` (flushed per row); `chapters.json` rewritten incrementally per chapter; `state.json` saved after every batch & chapter (`chapters_done`, `calls_today`, `day_stamp`, `failed_pages`).
- **Resume semantics**: chapters in `chapters_done` are skipped entirely; a quota-exit mid-chapter redoes only that chapter under the SAME output file (chapter rows are written only after the chapter fully completes — no partial writes).

### 4.12 Recovery mode (`/recover` → `recover_pages`)

- Takes a JSON plan `{"PSY-016": {"pages": [217], "reason": "..."}}`, folds existing JSONL rows back into records (`final_q_to_record`), re-extracts ONLY those pages (high-DPI render, ±1 neighbour context), merges fill-only, then runs detection-only integrity sweep + targeted retry (including sweep-forced truncated solutions — added in the review; recovery previously could never heal truncations).
- Quota counted & braked like the main loop; stem resolver runs in fill-only mode (kept-old verdict logged to `stem_conflicts.jsonl`).
- Writes a NEW questions.jsonl atomically (backup of previous kept).

### 4.13 Offline validation (`qbank_validator.py`)

`run_hybrid(output_root, audit=False)`:
- **Deterministic layer (0 tokens)** over rows + chapters + sidecars. Flag kinds (severity high/low): `truncated_solution`, `suspect_truncated_table`, `foreign_solution_segment` (shingle-8 overlap ≥400 chars between sibling solutions → suspect = longer row), `option_solution_disagree`, `answer_mismatch`, `duplicate_text` (cross-row stem dup with coherence suspect), `duplicate_id`, `duplicate_table`, `empty_question`, `bad_options`, `missing_answer`, `missing_solution`, `numbering_gap`, `numbering_start`, `source_gap`, `short_bare_solution`, `solution_header_furniture`, `solution_recitation_dump`, `foreign_option_head`, `image_ref_missing`, `suspicious_tiny_image`, `over_attributed_images`, `image_unclaimed`, `stray_answer_key_table`, `orphan_unresolved`, `answer_key_only_suppressed`, `suspect_density` (per-chapter anomaly density).
- **Audit mode** (opt-in, capped calls): sends only flagged rows (+ context) to Gemini for semantic verdicts (`audit_missing_question`, `audit_ghost_question`, `audit_component_missing`, `verified_clean`, etc.) — human review funnel without full-dataset cost.
- Output: `data/validation_report.json` (generated_at, flags_total, by kind, by chapter) + dashboard log box rendering.

### 4.14 Healing (`fix_output.py`, dashboard 🩹 Fix)

- `patch_all(questions, ...) -> (rows, actions, archive)`: patches **P1–P13**, each **evidence-gated** (a patch only fires when its specific defect signature is present — safe to run on any dataset; skip ≠ failure), **idempotent**.
- Examples: P4/P10 strip stray printed Answer-Key tables/dupes from solutions; P11 completes a truncated Erikson-stages table **verbatim** from the sibling question that prints the same table; P12 trims a proven foreign solution tail; P13 relabels a mis-addressed "Option C:" line → "Option D:".
- Guardrails: timestamped `.bak-YYYYMMDD-HHMMSS` backup, `fix_output_archive.jsonl` ledger of every action, auto re-validate + zip rebuild after patching.

---

## 5. Data artifacts inventory (Volume `/data/qbank_output/`)

| Path | Writer | Content |
|---|---|---|
| `state.json` | pipeline | `calls_today`, `day_stamp`, `pdf_progress.{SUBJECT}.chapters_done/current`, `failed_pages[]` |
| `data/questions.jsonl` | pipeline | final dataset, one row/question |
| `data/chapters.json` | pipeline | chapter registry (incremental) |
| `data/orphans.jsonl` | pipeline | unresolved fragments with provenance (`batch_start`, `pdf_pages`, `carry_q_no`, `blocked_reason`) |
| `data/unmatched_images.jsonl` | pipeline | extracted figures no question claimed |
| `data/decorative_images.jsonl` | pipeline | model-confirmed decorative figures |
| `data/integrity_flags.jsonl` | pipeline+sweep | answer-key disagreements, sweep strip evidence, truncated retries |
| `data/stem_conflicts.jsonl` | merge engine | both stem variants + verdict when coherence couldn't decide |
| `data/still_incomplete_after_retry.jsonl` | targeted retry | gaps left after retry rounds, with `chapter_id` |
| `data/fix_output_archive.jsonl` | healer | every patch action |
| `data/validation_report.json` | validator | latest report |
| `assets/questions/{SUBJECT}/*.webp` | pipeline | all figures (single folder convention — no solutions/options/tables splits) |
| `data/*.bak-*` | healer | pre-patch backups |

---

## 6. Prompts (what Gemini is actually asked)

1. **Main extraction (`SCHEMA_PROMPT`)** — strict JSON array per question; VERBATIM; answer-key tables = one entry/row (worked 3-row example embedded); solutions-only pages → solution-only items; split options → only what's visible; markdown for any table; null-q_no fragments allowed ("never invent a number"); context-block semantics (continue-only, never re-emit); mandatory `_batch_meta` trailer `{last_q_no, ends_mid_content, cut_part, tail_text}`; bare-JSON output.
2. **Targeted retry** — "here are rows with specific missing fields; fill ONLY these" with the chapter's page images; small/narrow beats big/general (per-call error-rate finding).
3. **Failed-page drain** — re-framed extraction ask (different wording to dodge recitation triggers) + neighbour context.
4. **Image attribution** — single image + chapter's question list → `{q_no, slot}` / `decorative:true` / abstain.

---

## 7. Config constants (current, tuned on the PSY trial book)

| Const | Value | Why |
|---|---|---|
| `GEMINI_MODEL` | `gemini-3.1-flash-lite-preview` | free-tier, confirmed working |
| `PAGES_PER_GEMINI_CALL` | 6 | accuracy/cost balance |
| `BATCH_OVERLAP_PAGES` | 2 | boundary items seen whole ≥1× |
| `TARGETED_RETRY_MAX_ROUNDS` | 2 | diminishing returns after |
| `SOLUTION_GATE_MIN_SHARE` | 0.6 | distinguishes "prints solutions" vs answer-only chapters |
| `MAX_CALLS_PER_DAY` | 1400 | 7 % buffer under 1500/day free tier |
| `MAX_QUESTION_IMAGES` | 3 | >3 question-side figures ≈ mis-attribution |
| `MIN_IMAGE_BYTES` | 1500 | <1.5 KB webp ≈ broken crop |
| Render DPI | 150 (recovery: higher) | readability/cost |
| Backoffs | 65 s (429), 20 s (transient 5xx) | RPM-burst vs daily-cap disambiguation |
| Stem dedup sim | 0.95 SequenceMatcher | overlap re-read tolerance |
| Fragment containment | 0.85 | append dedupe |
| Foreign-segment detector | shingle-8, ≥400 chars | copy-paste tails between sibling solutions |
| Safety | BLOCK_ONLY_HIGH ×4 | clinical content must pass |

---

## 8. Failure & quota semantics (state machine)

- **429 (burst)** → 65 s → retry once → still 429 ⇒ daily cap ⇒ save + `sys.exit(0)`. Next container start resumes.
- **Transient 5xx / high-demand** → 20 s → retry once → fail ⇒ split halves → singles.
- **finish_reason ≠ STOP (RECITATION=4 etc.)** → page-by-page isolation → per-page persist to `failed_pages` → chapter-end drain (re-frame + crops) → recovery plan remains as last resort.
- **Any unexpected exception per batch** → page-by-page isolation (never lose 6 pages to 1 fault).
- **Daily brake / quota exit** → `state.json` + `chapters.json` + flushed `questions.jsonl` are always consistent; resume is idempotent per chapter.

---

## 9. Proven production behaviour (PSY trial book — 33 chapters, 434 questions)

- Full book extracted in 1 run + 1 same-day resume across a mid-run API-key swap; each chapter closed "0 missing answer, 0 missing solution".
- Guards observed firing in real logs: over-attribution (4×), stem-conflict kept-old (1), answer-key consume-verified (7 rows), foreign-key block, sweep forced-solution re-asks, 65 s 429-burst backoffs, RECITATION page quarantined instead of sinking batches.
- Post-run deterministic validation: 52 flags, all triaged — majority were **false positives later fixed in the validator** (dangling-colon + table rows), 39 cosmetic stray answer-key tables (root-fix shipped so future books never carry them), 3 real content issues healed by healer patches, 3 image-groups left for human screenshots.
- **Known residual**: page 217 (PSY-016) recitation-sensitive — reprompt+crops did not clear it; chapter is nonetheless complete (17/17 with solutions from neighbours/overlap), page sits in `failed_pages` for a future recovery attempt.

---

## 10. Known limitations / open design questions (for reviewers)

1. **RECITATION on dense verbatim textbook pages** — the crop ladder mitigates but doesn't eliminate; would Gemini 2.5/3.x non-lite, a "transcribe as data, document ID" framing, or a two-model fallback help?
2. **Chapter discovery depends on printed TOC** (pages 1–3) + a manual `page_offset` per book. Fragile for books without clean TOCs — worth a robust page-label/printed-number detector?
3. **Answer-key formats vary by publisher** — `ANSWER_KEY_ROW_RE` + prompt example handle the trial publisher; generalization strategy for the next ~19 books?
4. **Concurrency = 1 call at a time** (stateless sequential batches). RPM headroom exists (15 RPM free); a small parallel pool with per-call retry could cut wall-clock 3–4×, at the cost of reordering-safe merge (q_no-keyed, so mostly safe) — worth it?
5. **Detection-vs-repair asymmetry** — validator flags exceed healer patches (P1–P13 are defect-specific). A generic "flag→auto-re-ask" loop (v2 recovery driven by validator output) is an obvious next iteration.
6. **No offline regression dataset** — heuristics were tuned/verified on one book's outputs; a frozen mini-corpus (3 chapters ×3 books) with expected JSON would de-risk future prompt edits.
7. **Single-volume persistence** — restore-from-Drive exists but there's no automatic post-run Drive upload/backups.
8. **Image side-assignment heuristic** in claim pass 3 (`fig/diagram` keywords in stem vs solution presence) is coarse — better visual/layout priors possible?
9. **Token/context budget per call** is unmanaged — 6 page images + long context string per call; trimming carry context (tail_text 25 words) helps; measure actual utilization?
10. **Validator severity model is flat (high/low)** — a confidence×impact scoring would prioritize human review better at 20-book scale.

---

### 4.16 Changelog — 2026-07-27 (external-audit round 2, 26 tests green)

Triggered by an independent zip audit of a duplicate test-run dataset (same
trial book under two subject codes = 868 rows; 47 flags):

1. **Foreign dump-tail trim (sweep step 2b)** — `chapter_integrity_sweep` now
   trims an embedded `Solution to Question N:` tail when the numbered sibling
   record in the same chapter already owns a non-empty solution (redundancy
   proven). Donor-less tails kept + review-flagged. Origin: sanitize only
   trimmed tails duplicating the record's OWN text; neighbours' unique text
   was conservatively kept → the chapter's first solutions record could ship
   a whole-page blob (was 27/47 flags in the audit).
2. **still_incomplete ledger rewrite** — `_prune_still_incomplete(chapter_id)`
   runs before every chapter's ledger write; healed rows can no longer leave
   stale "missing" entries (was 12 stale entries in the audit).
3. **Zip hygiene** — `make_zip` excludes healer `.bak-*` snapshots and the
   `_archive/` tree.
4. **Healer generalized** — P12/P13 are now content-signature gated for ANY
   subject (not PSY-locked); NEW **P14** (generic embedded dump-tail trim with
   donor guard, any subject) and **P15** (generic same-header sibling-table
   completion, Erikson-class).
5. **Dashboard 🧹 /reset** — Danger-zone button (requires typing RESET):
   archives `data/`, `assets/`, `state.json` into `_archive/<ts>/` INSIDE the
   volume (nothing deleted), giving each new book a clean slate. Reset is
   400-blocked without a mounted Volume or while a run is processing.

---

## 11. What feedback is wanted from the reviewer

- Correctness bugs / race conditions / data-loss paths in `qbank_pipeline.py` (merge, resume, quota exits).
- Prompt-engineering improvements for `SCHEMA_PROMPT`/retry/drain prompts (verbatim fidelity vs recitation).
- Architecture: is the sequential state-machine + sidecar-JSONL design right for ~20 books, or should anything be restructured (DB? task queue?) — keep in mind the user is non-technical and deploys only through the dashboard.
- Validator: missing deterministic checks; better heuristics for truncation/foreign-content detection without false positives.
- Cost/quota strategy on the free tier (parallelism, caching, model choice).
- Any security/robustness issues in the Flask dashboard (path handling, uploads, restores).
