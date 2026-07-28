# V2 Handover Brief — SS/PST (paste this into the new Arena chat working on branch `V2`)

You are continuing work on a production **PDF→Gemini-Vision→JSONL MCQ extraction pipeline**.
Owner is NON-TECHNICAL, phone-only (cannot run terminals). Replies: short Hinglish, "bro", click-only instructions.

## Branch map
- `arena/019f92d5-json-extract` = v1 pipeline (PROVEN on trial book: 33 ch, 434 Qs) + `pipeline_v2/` folder = LATEST v2 code
- `V2` = YOUR branch. Root files = v2 (user manually uploaded; Railway deploys THIS branch).
  ⚠️ Your root `qbank_pipeline.py` lags arena's `pipeline_v2/qbank_pipeline.py` by one iteration.

## FIRST TASK before anything else
Port the two improvements from arena branch (commit `ddebb94`) onto YOUR root:
```
git fetch origin arena/019f92d5-json-extract && git checkout FETCH_HEAD -- pipeline_v2/qbank_pipeline.py pipeline_v2/V2_README.md
mv pipeline_v2/qbank_pipeline.py qbank_pipeline.py && mv pipeline_v2/V2_README.md README.md && rmdir pipeline_v2
```
Changes = RPM pacing (`MIN_SECONDS_BETWEEN_CALLS=5`, `_pace_gemini_call()` at the two call choke points — kills 15-RPM 429 bursts) + per-chapter files (`data/by_chapter/`) + subject bundle (`subjects/{SUBJECT}/`). Test with throwaway venv (see quirks), then commit/push to `V2`.

## Docs to read first (all present in YOUR V2 checkout)
- `README.md` — v2 architecture (3-pass Q/A/S, probes, clip guard)
- `PIPELINE_WORKFLOW.md` — full pipeline working-process handover (13 stages, config table, quota semantics, proven prod behavior, open questions)
- `FINAL_AUDIT_REPORT.md` + `ROOT_CAUSE_ANALYSIS.md` — why every guard exists (run-4 audit evidence)

## Non-negotiable rules (violating = data corruption)
1. Never paraphrase/summarize extracted text — verbatim only. 2. Never invent a q_no. 3. Never delete possibly-unique content: trims/clips ONLY with a donor proof (sibling item or chapter record owning the content). 4. Images: ALL under `assets/questions/{SUBJECT}/` with `_Q_01/_SOL_01/_OPT_A_01/_TABLE_01` suffixes. 5. Model locked: `gemini-3.1-flash-lite-preview`. 6. Work ONLY on branch `V2`; never touch `arena/…` or `main`. 7. Railway volume `/data` must be mounted before any run (dashboard has a hard block + red banner).

## Key facts (verified this project)
- Free tier: 1500 calls/day per project, ~15 RPM. Daily brake MAX_CALLS_PER_DAY=1400; 65s backoff disambiguates RPM-burst vs daily-cap. NEW: 5s pacing prevents bursts entirely.
- v2 architecture: same 6-page/2-overlap batches, but zero-token pdftotext probe activates only needed passes (Q-run before solutions section, S after sticky boundary, A only on key pages). Per-pass carry-forward with hard reset at section boundary. S-pass responses clipped via `clip_pass_solutions` (sibling-donor only).
- Output roots: v1 data `/data/qbank_output` (PSY+ACH trial data lives here — DO NOT modify), v2 `/data/qbank_output_v2`, smoke tests `…_v2test/`.
- Dashboard (app.py): violet V2 banner + 🧪 /v2-test card, run/fix/validate/data-status/restore/reset buttons. User operates via taps only.

## Trial-run data (read-only reference, Google Drive)
- Full trial output mirror: https://drive.google.com/drive/folders/1T4Wjucu_822tEgTZBNiJdqPRFj96MnVa
- Production run log: https://drive.google.com/file/d/1I-EnTyq4ugyLPa0KPWvaQworv9jSIQ2p/view
- Known benign flag: PSY-025-006 option_solution_disagree is content-verified OK (heuristic over-strict by design).

## Pending items
1. Port ddebb94 (FIRST TASK above), then user re-runs the 🧪 smoke test (ch 11 of the trial book, subject V T2 etc.) — ask for the black-log-box screenshot.
2. 6 unclaimed images in trial data (pages 37×2, 263×2, 272×2) — waiting on user screenshots from the Drive folder to decide attach vs decorative.
3. Page 217 (PSY-016) RECITATION-blocked twice (batch/alone/crops) — sits in state failed_pages; recover only on a fresh-quota day.
4. Claude (external AI) is writing improvement notes on PIPELINE_WORKFLOW.md — when the user pastes them, review skeptically vs actual code before committing anything (last external review had 1 correct fix, 1 data-loss-disguised-as-fix, 2 already-implemented suggestions).

## Sandbox quirks (save hours)
- Bash network is flaky (curl SSL errors): use fetch_page/web_search tools for GitHub raw, Railway URLs, Drive. Drive folder listing: `https://drive.google.com/embeddedfolderview?id={FOLDER_ID}#list`; file download: `uc?export=download&id={FILE_ID}`.
- Chat attachments don't sync into workspace — ask user for Drive links/screenshots instead.
- Sandbox git resets between turns: always `git fetch origin V2 -q && git reset FETCH_HEAD -q` before any git op.
- Tests: `python3 -m venv .venv-test` + `pip install pillow pypdf flask requests ruff`; stub `google`/`google.generativeai` in sys.modules before importing qbank_pipeline; DELETE venv + test file + __pycache__ after passing; never commit them.
