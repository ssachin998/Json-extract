#!/usr/bin/env python3
"""
One-shot, evidence-gated heal script for the run-4 PSY output.
Fixes the exact data defects found in the 2026-07-26 full audit
(FINAL_AUDIT_REPORT.md D1-D10). Code-level PREVENTION of every class
already lives in qbank_pipeline.py (integrity sweep + guards); this file
heals the rows that were WRITTEN BEFORE those guards existed.

Safety rules:
  * every patch checks the EXPECTED current content first; if the pattern
    is absent (file already healed / newer run), it reports SKIPPED and
    touches nothing -- safe to run repeatedly (idempotent).
  * never paraphrases: every replacement string is verbatim text taken
    from the SAME record or its sibling record (audit evidence).
  * default = DRY-RUN (prints what would change). Pass --apply to write.
  * on --apply: timestamped backup + atomic replace.

Run where the volume is mounted:
    python3 fix_output.py                 # dry-run preview
    python3 fix_output.py --apply         # heal data/questions.jsonl
"""

import json
import os
import re
import sys
import time
from pathlib import Path

OUTPUT_ROOT = Path(os.environ.get("OUTPUT_DIR", "./qbank_output"))
QUESTIONS = OUTPUT_ROOT / "data" / "questions.jsonl"
ARCHIVE_LOG = OUTPUT_ROOT / "data" / "fix_output_archive.jsonl"


def _norm_md(md):
    return re.sub(r"\s+", "", (md or "").lower())


def _coherence(stem, row):
    toks = [t for t in re.findall(r"\w+", (stem or "").lower()) if len(t) > 2]
    payload = " ".join(filter(None, [
        (row.get("solution") or {}).get("text") or "",
        " ".join(str(o.get("text") or "") for o in row.get("options") or []),
    ]))
    ptoks = set(re.findall(r"\w+", payload.lower()))
    if not toks or not ptoks:
        return 0.0
    return sum(1 for t in toks if t in ptoks) / len(toks)


def patch_all(rows, assets_q):
    """Returns (rows, actions). actions: list of (patch_id, status, detail)."""
    by_id = {r.get("id"): r for r in rows}
    actions, archive = [], []

    def act(pid, status, detail, archived=None):
        actions.append((pid, status, detail))
        if archived is not None:
            archive.append({"patch": pid, "archived": archived})

    # ---- P1: PSY-009-007 foreign "Option C: Catharsis" head -> PSY-009-006
    r6, r7 = by_id.get("PSY-009-006"), by_id.get("PSY-009-007")
    LINE = ("Option C: Catharsis is the expression of ideas, thoughts, and suppressed "
            "material that is accompanied by an emotional response that produces a "
            "state of relief in the patient.")
    if r7 and (r7["solution"]["text"] or "").strip().startswith(LINE[:40]):
        s7 = r7["solution"]["text"].strip()
        assert s7.startswith(LINE), "unexpected start"
        r7["solution"]["text"] = s7[len(LINE):].lstrip("\n ")
        if r6 and LINE not in (r6["solution"]["text"] or ""):
            r6["solution"]["text"] = ((r6["solution"]["text"] or "").rstrip() + "\n\n" + LINE).strip()
        act("P1", "APPLY", "moved 'Option C: Catharsis...' line PSY-009-007 -> PSY-009-006")
    else:
        act("P1", "SKIP", "PSY-009-007 no longer starts with the Catharsis line")

    # ---- P2: PSY-008-007 options A/B/D polluted by previous question's
    # explanations. Replacement texts come from THIS record's own solution
    # option-lines (verbatim print).
    r = by_id.get("PSY-008-007")
    if r and any("CAGE questionnaire is used" in (o.get("text") or "") for o in r.get("options", [])):
        fixed = {"A": "Chaotic (hypsarrhythmia)", "B": "3Hz spike and wave", "D": "Isoelectric EEG"}
        for o in r["options"]:
            if o["id"] in fixed:
                act("P2-old", "INFO", f"option {o['id']} was: {(o.get('text') or '')[:70]!r}")
                o["text"] = fixed[o["id"]]
        act("P2", "APPLY", "PSY-008-007 options A/B/D rebuilt from its own solution lines")
    else:
        act("P2", "SKIP", "PSY-008-007 options already sane")

    # ---- P3: PSY-032-003 whole recitation-block dump after the real solution
    r = by_id.get("PSY-032-003")
    if r:
        m = re.search(r"Solution\s+to\s+Question\s+3\s*", r["solution"]["text"] or "")
        if m and m.start() > 0:
            act("P3", "APPLY", f"PSY-032-003 truncated at embedded 'Solution to Question 3:' "
                               f"(cut {len(r['solution']['text']) - m.start()} chars)",
                archived=r["solution"]["text"][m.start():])
            r["solution"]["text"] = r["solution"]["text"][:m.start()].rstrip()
        else:
            act("P3", "SKIP", "PSY-032-003 has no embedded dump header")

    # ---- P4/P5/GENERIC: duplicate tables inside one solution (all rows)
    for r in rows:
        tbls = (r.get("solution") or {}).get("tables") or []
        seen, keep, dups = set(), [], 0
        for t in tbls:
            k = _norm_md(t.get("markdown"))
            if k and k in seen:
                dups += 1
                continue
            if k:
                seen.add(k)
            keep.append(t)
        if dups:
            r["solution"]["tables"] = keep
            act("P4", "APPLY", f"{r['id']}: dropped {dups} duplicate table(s)")

    # ---- P5b: PSY-009-005 raw markdown table pasted inside solution TEXT
    r = by_id.get("PSY-009-005")
    if r:
        st = r["solution"]["text"] or ""
        if "\n|" in st:
            kept = [ln for ln in st.splitlines() if not ln.strip().startswith("|")]
            stripped = "\n".join(kept).strip()
            if stripped and len(stripped) > 20:
                act("P5", "APPLY", "PSY-009-005 raw markdown table lines removed from solution text",
                    archived=st)
                r["solution"]["text"] = stripped

    # ---- P6: PSY-006-013/014 split table -> merge rows back into 013's table
    r13, r14 = by_id.get("PSY-006-013"), by_id.get("PSY-006-014")
    if r13 and r14:
        t13 = [t for t in (r13["solution"].get("tables") or [])
               if "D2" in (t.get("markdown") or "") or "generation" in (t.get("markdown") or "").lower()]
        t14_short = [t for t in (r14["solution"].get("tables") or [])
                     if str(t.get("type", "")).lower() == "short_label"
                     and "5HT2A" in (t.get("markdown") or "")]
        if t13 and t14_short:
            md13 = t13[0]["markdown"].rstrip()
            rows14 = [ln for ln in t14_short[0]["markdown"].splitlines()
                      if ln.strip().startswith("|") and "---" not in ln
                      and "Property" not in ln]
            t13[0]["markdown"] = md13 + "\n" + "\n".join(rows14)
            r14["solution"]["tables"] = [t for t in r14["solution"]["tables"]
                                         if t is not t14_short[0]]
            act("P6", "APPLY", f"merged {len(rows14)} table row(s) PSY-006-014 -> PSY-006-013 "
                               f"(antipsychotic classification table re-joined)")
        else:
            act("P6", "SKIP", "split-table pattern not found")

    # ---- P7: PSY-023-007 mangled+truncated solution -> verbatim comorbidity
    # paragraph (currently sitting at the END of PSY-023-006's solution)
    r6, r7 = by_id.get("PSY-023-006"), by_id.get("PSY-023-007")
    if r6 and r7:
        m = re.search(r"(Depression is the most common psychiatric comorbidity[^|]*)",
                      r6["solution"]["text"] or "", re.DOTALL)
        cur7 = r7["solution"]["text"] or ""
        if m and "Depression is the most common psychiatric comorbidity" not in cur7:
            para = m.group(1).strip()
            act("P7", "APPLY", "PSY-023-007 solution replaced with its verbatim explanation "
                               "(moved from PSY-023-006 tail); old mangled fragment archived",
                archived=cur7)
            r7["solution"]["text"] = para
            # trim the moved tail from 006 (its own lab content stays intact)
            r6["solution"]["text"] = (r6["solution"]["text"][:m.start()] +
                                      r6["solution"]["text"][m.end():]).rstrip()
        else:
            act("P7", "SKIP", "PSY-023-007 already carries the comorbidity explanation")

    # ---- P8: PSY-003-014 broken figure ref (414-byte webp cannot hold a TAT
    # card photo) -- de-reference so the app never renders a broken image.
    # Re-extraction of the real figure happens via the pipeline's image pass.
    r = by_id.get("PSY-003-014")
    if r:
        imgs = (r.get("question") or {}).get("images") or []
        bad = [i for i in imgs
               if not (assets_q / i["file"]).exists()
               or (assets_q / i["file"]).stat().st_size < 1500]
        if bad:
            r["question"]["images"] = [i for i in imgs if i not in bad]
            act("P8", "APPLY", f"PSY-003-014 de-referenced {len(bad)} broken/tiny image ref(s) "
                               f"(re-extract pending -- validator HIGH flag remains)",
                archived=bad)
        else:
            act("P8", "SKIP", "PSY-003-014 image refs look healthy")

    # ---- P9: PSY-012-001 wrong-owner stem (exact copy of PSY-012-013's
    # chart stem, no image, and its own solution describes a different
    # patient). Null the stem so it is visibly 'stem missing' -- the
    # pipeline's Gap-1 retry re-asks it from the pages; answers coherent
    # with its own solution stay.
    r1, r13 = by_id.get("PSY-012-001"), by_id.get("PSY-012-013")
    if r1 and r13:
        s1 = (r1["question"]["text"] or "").strip()
        s13 = (r13["question"]["text"] or "").strip()
        if s1 and s1 == s13 and _coherence(s1, r1) < 0.05:
            act("P9", "APPLY", "PSY-012-001 wrong-owner stem nulled (was identical to "
                               "PSY-012-013 with zero payload coherence; targeted retry refills it)",
                archived=s1)
            r1["question"]["text"] = None
        else:
            act("P9", "SKIP", "PSY-012-001 stem no longer a zero-coherence duplicate")

    return rows, actions, archive


def main():
    apply_mode = "--apply" in sys.argv
    assets_q = OUTPUT_ROOT / "assets" / "questions"
    if not QUESTIONS.exists():
        print(f"questions.jsonl not found at {QUESTIONS} (set OUTPUT_DIR)")
        sys.exit(1)
    rows = [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]

    rows, actions, archive = patch_all(rows, assets_q)
    mode = "APPLY" if apply_mode else "DRY-RUN"
    print(f"=== fix_output ({mode}) -- {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    n_apply = 0
    for pid, status, detail in actions:
        if status == "APPLY":
            n_apply += 1
        print(f"  [{status:5}] {pid}: {detail}")
    print(f"-- {n_apply} patch(es) {'written' if apply_mode else 'would be written (dry-run)'}")

    if apply_mode and n_apply:
        backup = QUESTIONS.with_suffix(f".bak-{time.strftime('%Y%m%d-%H%M%S')}")
        backup.write_text(QUESTIONS.read_text(encoding="utf-8"), encoding="utf-8")
        tmp = QUESTIONS.with_suffix(".tmp")
        tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                       encoding="utf-8")
        os.replace(tmp, QUESTIONS)
        if archive:
            with open(ARCHIVE_LOG, "a", encoding="utf-8") as f:
                for a in archive:
                    f.write(json.dumps(a, ensure_ascii=False) + "\n")
        print(f"backup -> {backup}")
        print(f"healed -> {QUESTIONS}")
        if archive:
            print(f"original fragments archived -> {ARCHIVE_LOG}")
        print("Re-run the validator for a fresh report:  python3 qbank_validator.py --report-only")


if __name__ == "__main__":
    main()
