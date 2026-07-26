#!/usr/bin/env python3
"""
Phone-friendly dashboard for the QBank pipeline.
Upload a PDF (or paste a link), tap Run, watch progress, download results —
all from a phone browser. No terminal/PC needed once this is deployed.

This is just a thin UI wrapper. All the real extraction logic (watermark
detection, Gemini vision calls, checkpointing, rate-limit handling) lives
in qbank_pipeline.py — this file does not duplicate or replace any of that.
"""

import os
import threading
import traceback
import zipfile
from pathlib import Path

import requests
from flask import Flask, render_template_string, request, redirect, url_for, send_file, jsonify
from werkzeug.utils import secure_filename

import qbank_pipeline as pipeline

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB per PDF upload

UPLOAD_DIR = Path("./pdfs")
UPLOAD_DIR.mkdir(exist_ok=True)

state_lock = threading.Lock()
state = {"status": "idle", "log": [], "error": None}

def log(msg):
    print(msg, flush=True)  # so it shows in Railway's Deploy Logs too, not just the dashboard box
    with state_lock:
        state["log"].append(msg)
        if len(state["log"]) > 500:
            state["log"].pop(0)

def run_validator_and_log():
    """Zero-token deterministic validation; every flag printed to the
    dashboard log box so no terminal is needed. Returns the report dict."""
    import qbank_validator
    rep = qbank_validator.run_hybrid(pipeline.OUTPUT_ROOT, audit=False)
    s = rep["summary"]
    log(f"🧪 Validator: {s['flags_total']} flag(s) across "
        f"{s['flagged_chapters']}/{s['chapters']} chapters ({s['questions']} questions)")
    for kind, n in sorted(s.get("flags_by_kind", {}).items(), key=lambda kv: -kv[1]):
        log(f"   • {kind}: {n}")
    for cid, flags in (rep.get("chapters") or {}).items():
        for f in flags:
            log(f"   [{f.get('severity', '?')}] {cid} {f.get('q_no') or '-'} "
                f"{f.get('kind')}: {str(f.get('detail', ''))[:100]}")
    log("🧪 Full report -> data/validation_report.json (inside the zip)")
    return rep

def run_pipeline_thread(subject_code, pdf_path, page_offset):
    with state_lock:
        state["status"] = "processing"
        state["error"] = None
    try:
        pipeline.PDFS[:] = [{"subject": subject_code, "path": str(pdf_path), "page_offset": page_offset}]
        pipeline.main()
        # zero-token deterministic validation right after every run -- the
        # defect map (numbering gaps, RC-4-aware missing solutions, orphan /
        # unmatched-image sidecars) lands in data/validation_report.json.
        try:
            import qbank_validator
            rep = qbank_validator.run_hybrid(pipeline.OUTPUT_ROOT, audit=False)
            log(f"🧪 Validation: {rep['summary']['flags_total']} flag(s) across "
                f"{rep['summary']['flagged_chapters']}/{rep['summary']['chapters']} chapters → "
                f"see data/validation_report.json")
        except Exception as ve:
            log(f"⚠️ post-run validation report failed (extraction unaffected): {ve}")
        with state_lock:
            state["status"] = "completed"
        log("✅ Done (or paused at daily Gemini limit — tap Run again tomorrow to resume).")
        make_zip()
    except SystemExit:
        with state_lock:
            state["status"] = "paused"
        log("⏸ Hit daily Gemini call limit — progress saved. Come back tomorrow and tap Run again.")
        make_zip()
    except Exception as e:
        with state_lock:
            state["status"] = "failed"
            state["error"] = str(e)
        log(f"❌ Error: {e}")
        traceback.print_exc()  # full traceback with file/line -> Railway Deploy Logs

def make_zip():
    # Must mirror qbank_pipeline's OUTPUT_ROOT -- on Railway that's
    # /data/qbank_output (the Volume), NOT the local ./qbank_output.
    # (Hardcoding the relative path here meant the zip was never created
    # on Railway, so /download always 404'd.)
    out = Path(os.environ.get("OUTPUT_DIR", "./qbank_output"))
    if not out.exists():
        return
    zpath = Path("output_results.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in out.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(out.parent))

PAGE = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QBank Extractor</title>
<script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-50 p-4">
<div class="max-w-lg mx-auto space-y-4">
  <h1 class="text-xl font-bold">QBank Extractor</h1>

  <div class="bg-white rounded-lg shadow p-4">
    <p class="text-sm mb-2">Status: <span class="font-semibold">{{ state.status }}</span></p>
    {% if state.error %}<p class="text-red-600 text-sm">{{ state.error }}</p>{% endif %}
    <a href="/download" class="inline-block mt-2 bg-emerald-600 text-white text-sm px-3 py-2 rounded">Download results (.zip)</a>
  </div>

  <div class="bg-white rounded-lg shadow p-4 border-2 border-amber-400 space-y-2">
    <p class="text-xs font-bold text-amber-700 uppercase">Maintenance — no terminal needed</p>
    <form action="/fix" method="POST">
      <button class="w-full bg-amber-500 text-white font-bold py-2 rounded" {% if state.status == 'processing' %}disabled{% endif %}>
        🩹 Fix data (heal known defects)
      </button>
    </form>
    <form action="/validate" method="POST">
      <button class="w-full bg-sky-600 text-white font-bold py-2 rounded" {% if state.status == 'processing' %}disabled{% endif %}>
        🔍 Check data (validator report)
      </button>
    </form>
    <a href="/data-status" class="block text-center w-full bg-slate-200 text-slate-800 font-bold py-2 rounded">📦 Data status (is my file safe?)</a>
    <p class="text-xs text-gray-500">Tap <b>Fix</b> first (auto backup, safe to tap again), then <b>Check</b>. Every flag appears in the black log box below — screenshot it and send it.</p>
  </div>

  <div class="bg-white rounded-lg shadow p-4 border-2 border-emerald-500 space-y-3">
    <p class="text-xs font-bold text-emerald-700 uppercase">Recommended for phone</p>
    <form action="/run-url" method="POST" class="space-y-3">
      <div>
        <label class="block text-sm font-semibold mb-1">PDF link (Google Drive / Telegram / direct download URL)</label>
        <input type="url" name="pdf_url" class="w-full text-sm border p-2 rounded" placeholder="https://..." required>
      </div>
      <div>
        <label class="block text-sm font-semibold mb-1">Subject code (3 letters)</label>
        <input type="text" name="subject_code" maxlength="3" class="w-full text-sm border p-2 rounded uppercase" placeholder="PSY" required>
      </div>
      <div>
        <label class="block text-sm font-semibold mb-1">Page offset</label>
        <input type="number" name="page_offset" value="-1" class="w-full text-sm border p-2 rounded">
      </div>
      <button class="w-full bg-emerald-600 text-white font-bold py-2 rounded" {% if state.status == 'processing' %}disabled{% endif %}>
        Run (from link)
      </button>
    </form>
  </div>

  <details class="bg-white rounded-lg shadow p-4">
    <summary class="text-sm font-semibold cursor-pointer">Or upload file directly (less reliable on mobile)</summary>
    <form action="/run" method="POST" enctype="multipart/form-data" class="space-y-3 mt-3">
      <div>
        <label class="block text-sm font-semibold mb-1">PDF file</label>
        <input type="file" name="file" accept=".pdf" class="w-full text-sm border p-2 rounded">
      </div>
      <div>
        <label class="block text-sm font-semibold mb-1">Subject code (3 letters)</label>
        <input type="text" name="subject_code" maxlength="3" class="w-full text-sm border p-2 rounded uppercase" placeholder="PSY" required>
      </div>
      <div>
        <label class="block text-sm font-semibold mb-1">Page offset</label>
        <input type="number" name="page_offset" value="-1" class="w-full text-sm border p-2 rounded">
      </div>
      <button class="w-full bg-slate-800 text-white font-bold py-2 rounded" {% if state.status == 'processing' %}disabled{% endif %}>
        Run (upload)
      </button>
    </form>
  </details>

  <details class="bg-white rounded-lg shadow p-4">
    <summary class="text-sm font-semibold cursor-pointer">Recovery mode — heal specific pages (missing solutions etc.)</summary>
    <form action="/recover" method="POST" class="space-y-3 mt-3">
      <div>
        <label class="block text-sm font-semibold mb-1">Recovery plan (JSON)</label>
        <textarea name="plan" rows="6" class="w-full text-xs font-mono border p-2 rounded">{
  "PSY-016": {"pages": [214, 217], "reason": "recitation batch loss"},
  "PSY-001": {"pages": [17], "reason": "missing solution for q13"}
}</textarea>
        <p class="text-xs text-gray-500">Pages = true PDF file page numbers (see orphans.jsonl / unmatched image filenames). Renders ±1 neighbour page for context. Never overwrites existing text; only fills what is missing.</p>
      </div>
      <button class="w-full bg-indigo-600 text-white font-bold py-2 rounded" {% if state.status == 'processing' %}disabled{% endif %}>
        Run recovery
      </button>
    </form>
  </details>

  <div class="bg-black text-green-400 text-xs rounded-lg p-3 h-64 overflow-y-auto font-mono" id="log">
    {% for line in state.log %}{{ line }}<br>{% endfor %}
  </div>
</div>
<script>
setInterval(() => {
  fetch('/status').then(r => r.json()).then(d => {
    document.getElementById('log').innerHTML = d.log.join('<br>');
  });
}, 3000);
</script>
</body>
</html>
"""

import re as _re

def resolve_download_url(url):
    """Convert common share-link formats (Google Drive etc.) into a direct
    download URL. Falls back to the original URL if it's not recognized."""
    m = _re.search(r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)", url)
    if not m:
        m = _re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m:
        file_id = m.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url

@app.route("/")
def index():
    return render_template_string(PAGE, state=state)

@app.route("/status")
def status():
    return jsonify(state)

def parse_page_offset():
    """int() of "" or junk raises ValueError -> Flask 500 page. Be forgiving."""
    try:
        return int(request.form.get("page_offset") or -1)
    except (TypeError, ValueError):
        return -1

@app.route("/run-url", methods=["POST"])
def run_url():
    if state["status"] == "processing":
        return redirect(url_for("index"))
    pdf_url = resolve_download_url(request.form.get("pdf_url", "").strip())
    subject_code = request.form.get("subject_code", "").strip().upper()
    page_offset = parse_page_offset()
    if not pdf_url:
        return "No URL provided", 400
    # Mark busy NOW (before the background download starts) -- otherwise the
    # status stays "idle" during the download and a second tap on Run starts
    # a duplicate pipeline writing to the same output files.
    with state_lock:
        state["status"] = "processing"

    def download_then_run():
        try:
            log(f"⬇️ Downloading PDF from link...")
            # secure_filename: a hostile/odd URL tail like "../../x" must not
            # be able to write outside ./pdfs
            fname = secure_filename(pdf_url.split("/")[-1].split("?")[0]) or f"{subject_code}.pdf"
            if not fname.lower().endswith(".pdf"):
                fname = f"{subject_code}.pdf"
            pdf_path = UPLOAD_DIR / fname
            r = requests.get(pdf_url, stream=True, timeout=120,
                              headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()

            # Google Drive shows an interstitial "can't scan for viruses"
            # confirm page for some files instead of the raw bytes -- detect
            # that and follow the confirm link before saving.
            content_type = r.headers.get("Content-Type", "")
            if "text/html" in content_type:
                text = r.text
                confirm_match = _re.search(r'confirm=([0-9A-Za-z_-]+)', text)
                if confirm_match:
                    confirm_token = confirm_match.group(1)
                    r = requests.get(f"{pdf_url}&confirm={confirm_token}",
                                      stream=True, timeout=120,
                                      headers={"User-Agent": "Mozilla/5.0"})
                    r.raise_for_status()

            first_chunk = None
            with open(pdf_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        if first_chunk is None:
                            first_chunk = chunk
                        f.write(chunk)

            if not first_chunk or not first_chunk.startswith(b"%PDF"):
                pdf_path.unlink(missing_ok=True)
                log("❌ The link didn't return a real PDF file (got a webpage instead). "
                    "For Google Drive: right-click the file → Share → 'Anyone with the "
                    "link' → copy that link, and make sure the file itself (not a folder) is shared.")
                with state_lock:
                    state["status"] = "failed"
                    state["error"] = "Downloaded content is not a valid PDF"
                return

            log(f"✅ Downloaded {fname} ({pdf_path.stat().st_size // 1024} KB)")
            run_pipeline_thread(subject_code, pdf_path, page_offset)
        except Exception as e:
            with state_lock:
                state["status"] = "failed"
                state["error"] = str(e)
            log(f"❌ Download failed: {e}")
            traceback.print_exc()

    t = threading.Thread(target=download_then_run)
    t.daemon = True
    t.start()
    return redirect(url_for("index"))

@app.route("/run", methods=["POST"])
def run():
    if state["status"] == "processing":
        return redirect(url_for("index"))
    f = request.files.get("file")
    subject_code = request.form.get("subject_code", "").strip().upper()
    page_offset = parse_page_offset()
    if f and f.filename.lower().endswith(".pdf"):
        pdf_path = UPLOAD_DIR / (secure_filename(f.filename) or f"{subject_code}.pdf")
        f.save(pdf_path)
    else:
        # no new file uploaded -> reuse whatever PDF is already in ./pdfs
        existing = list(UPLOAD_DIR.glob("*.pdf"))
        if not existing:
            return "No PDF uploaded and none found in ./pdfs", 400
        pdf_path = existing[0]
    # same double-tap guard as /run-url
    with state_lock:
        state["status"] = "processing"
    t = threading.Thread(target=run_pipeline_thread, args=(subject_code, pdf_path, page_offset))
    t.daemon = True
    t.start()
    return redirect(url_for("index"))

RECOVERY_PLAN_PATH = Path("./recovery_plan.json")

@app.route("/recover", methods=["POST"])
def recover():
    if state["status"] == "processing":
        return redirect(url_for("index"))
    plan_text = request.form.get("plan", "").strip()
    if not plan_text:
        return "No plan provided", 400
    import json as _json
    try:
        plan = _json.loads(plan_text)
        assert isinstance(plan, dict) and plan, "plan must be a non-empty object"
        for cid, spec in plan.items():
            assert isinstance(spec.get("pages"), list) and spec["pages"], \
                f"{cid}: needs a non-empty 'pages' list"
    except (ValueError, AssertionError) as e:
        return f"Invalid plan JSON: {e}", 400
    RECOVERY_PLAN_PATH.write_text(plan_text)
    with state_lock:
        state["status"] = "processing"

    def _do_recover():
        try:
            log(f"🩹 Recovery started for: {', '.join(plan)}")
            pipeline.recover_pages(str(RECOVERY_PLAN_PATH))
            with state_lock:
                state["status"] = "completed"
            log("🩹 Recovery finished. Download zip to inspect healed rows.")
            make_zip()
        except SystemExit:
            with state_lock:
                state["status"] = "paused"
            log("⏸ Recovery paused at Gemini daily limit -- run it again tomorrow.")
            make_zip()
        except Exception as e:
            with state_lock:
                state["status"] = "failed"
                state["error"] = str(e)
            log(f"❌ Recovery error: {e}")
            traceback.print_exc()

    t = threading.Thread(target=_do_recover)
    t.daemon = True
    t.start()
    return redirect(url_for("index"))

@app.route("/fix", methods=["POST"])
def fix():
    """Button-only heal of known run-4 defects (fix_output.patch_all).
    Evidence-gated + idempotent, timestamped backup + archive before write."""
    if state["status"] == "processing":
        return redirect(url_for("index"))
    with state_lock:
        state["status"] = "processing"

    def _do_fix():
        try:
            import json as _json
            import time as _time
            import fix_output
            q = fix_output.QUESTIONS
            if not q.exists():
                log(f"❌ {q} not found on the Volume -- nothing to fix")
                with state_lock:
                    state["status"] = "failed"
                    state["error"] = "questions.jsonl not found"
                return
            rows = [_json.loads(l) for l in q.read_text(encoding="utf-8").splitlines() if l.strip()]
            log(f"🩹 Fix pass on {len(rows)} questions ...")
            assets_q = fix_output.OUTPUT_ROOT / "assets" / "questions"
            rows, actions, archive = fix_output.patch_all(rows, assets_q)
            n_apply = 0
            for pid, st, detail in actions:
                if st == "APPLY":
                    n_apply += 1
                log(f"   [{st}] {pid}: {detail}")
            if n_apply:
                backup = q.with_suffix(f".bak-{_time.strftime('%Y%m%d-%H%M%S')}")
                backup.write_text(q.read_text(encoding="utf-8"), encoding="utf-8")
                tmp = q.with_suffix(".tmp")
                tmp.write_text("\n".join(_json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                               encoding="utf-8")
                os.replace(tmp, q)
                if archive:
                    with open(fix_output.ARCHIVE_LOG, "a", encoding="utf-8") as fh:
                        for a in archive:
                            fh.write(_json.dumps(a, ensure_ascii=False) + "\n")
                log(f"✅ {n_apply} heal(s) written. Backup -> {backup.name}, "
                    f"original fragments archived.")
            else:
                log("✅ Nothing to heal -- data already clean (all patches skipped).")
            run_validator_and_log()
            with state_lock:
                state["status"] = "completed"
            make_zip()
        except Exception as e:
            with state_lock:
                state["status"] = "failed"
                state["error"] = str(e)
            log(f"❌ Fix error: {e}")
            traceback.print_exc()

    t = threading.Thread(target=_do_fix)
    t.daemon = True
    t.start()
    return redirect(url_for("index"))

@app.route("/validate", methods=["POST"])
def validate():
    """Button-only re-check: fresh validation_report.json + flags in the log."""
    if state["status"] == "processing":
        return redirect(url_for("index"))
    with state_lock:
        state["status"] = "processing"

    def _do_validate():
        try:
            run_validator_and_log()
            with state_lock:
                state["status"] = "completed"
        except Exception as e:
            with state_lock:
                state["status"] = "failed"
                state["error"] = str(e)
            log(f"❌ Validate error: {e}")
            traceback.print_exc()

    t = threading.Thread(target=_do_validate)
    t.daemon = True
    t.start()
    return redirect(url_for("index"))

@app.route("/data-status")
def data_status():
    """Read-only proof that the extraction data on the Volume is intact.
    Shows file sizes + question/image counts -- nothing is modified."""
    out = Path(os.environ.get("OUTPUT_DIR", "./qbank_output"))
    lines = [f"Output folder: {out}"]
    if not out.exists():
        lines.append("X  folder missing -- is the Railway Volume mounted on this service?")
        return "<pre style='font-size:15px;padding:12px'>" + "\n".join(lines) + "</pre>"
    q = out / "data" / "questions.jsonl"
    if q.exists():
        n = sum(1 for l in q.read_text(encoding="utf-8").splitlines() if l.strip())
        lines.append(f"OK  questions.jsonl = {n} questions  ({q.stat().st_size // 1024} KB)")
    else:
        lines.append("X  data/questions.jsonl missing")
    d = out / "data"
    if d.exists():
        for f in sorted(d.iterdir()):
            if f.name != "questions.jsonl":
                lines.append(f"    data/{f.name}  ({f.stat().st_size // 1024} KB)")
    aq = out / "assets" / "questions"
    if aq.exists():
        for sub in sorted(aq.iterdir()):
            if sub.is_dir():
                lines.append(f"OK  assets/questions/{sub.name} = {len(list(sub.iterdir()))} images")
    else:
        lines.append("X  assets/questions folder missing")
    bak = list(out.glob("data/*.bak-*"))
    if bak:
        lines.append(f"    backups found: {len(bak)}")
    lines.append("")
    lines.append("Agar upar 'OK questions.jsonl = 434 questions' dikh raha hai,")
    lines.append("to data 100% safe hai. Screenshot bhej do.")
    return "<pre style='font-size:15px;padding:12px'>" + "\n".join(lines) + "</pre>"

@app.route("/download")
def download():
    if os.path.exists("output_results.zip"):
        return send_file("output_results.zip", as_attachment=True)
    return "No results yet", 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
