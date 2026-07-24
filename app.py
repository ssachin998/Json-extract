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

def run_pipeline_thread(subject_code, pdf_path, page_offset):
    with state_lock:
        state["status"] = "processing"
        state["error"] = None
    try:
        pipeline.PDFS[:] = [{"subject": subject_code, "path": str(pdf_path), "page_offset": page_offset}]
        pipeline.main()
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
    out = Path("qbank_output")
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

@app.route("/run-url", methods=["POST"])
def run_url():
    if state["status"] == "processing":
        return redirect(url_for("index"))
    pdf_url = resolve_download_url(request.form.get("pdf_url", "").strip())
    subject_code = request.form.get("subject_code", "").strip().upper()
    page_offset = int(request.form.get("page_offset", -1))
    if not pdf_url:
        return "No URL provided", 400

    def download_then_run():
        try:
            log(f"⬇️ Downloading PDF from link...")
            fname = pdf_url.split("/")[-1].split("?")[0] or f"{subject_code}.pdf"
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
    page_offset = int(request.form.get("page_offset", -1))
    if f and f.filename.lower().endswith(".pdf"):
        pdf_path = UPLOAD_DIR / f.filename
        f.save(pdf_path)
    else:
        # no new file uploaded -> reuse whatever PDF is already in ./pdfs
        existing = list(UPLOAD_DIR.glob("*.pdf"))
        if not existing:
            return "No PDF uploaded and none found in ./pdfs", 400
        pdf_path = existing[0]
    t = threading.Thread(target=run_pipeline_thread, args=(subject_code, pdf_path, page_offset))
    t.daemon = True
    t.start()
    return redirect(url_for("index"))

@app.route("/download")
def download():
    if os.path.exists("output_results.zip"):
        return send_file("output_results.zip", as_attachment=True)
    return "No results yet", 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
