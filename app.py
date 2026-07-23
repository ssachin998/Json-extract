import os
import sys
import re
import shutil
import zipfile
import threading
import subprocess
from flask import Flask, render_template_string, request, redirect, url_for, send_file, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '.'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max limit

# Global state to track processing
state = {
    'status': 'idle',        # idle, processing, completed, failed
    'current_pdf': None,
    'subject_code': None,
    'logs': [],
    'error': None
}

state_lock = threading.Lock()

# Auto-detect subject code from filename
def detect_subject_code(filename):
    # Match 3-letter prefix at the start, e.g., PSY_Psychology.pdf or PSY-Psychology.pdf
    match = re.match(r'^([a-zA-Z]{3})[\s_\-]', filename)
    if match:
        return match.group(1).upper()
    
    # Otherwise, try to find any 3-letter sequence or just take the first 3 letters of the filename
    name_only = os.path.splitext(filename)[0]
    clean_name = re.sub(r'[^a-zA-Z]', '', name_only)
    if len(clean_name) >= 3:
        return clean_name[:3].upper()
    
    return "SUB"

# Find any PDF files in the current folder
def find_pdf_files():
    pdfs = []
    for f in os.listdir('.'):
        if f.lower().endswith('.pdf'):
            # Ignore intermediate temp PDFs if any
            if not f.startswith('temp_') and os.path.isfile(f):
                pdfs.append(f)
    return pdfs

# Worker function to run the pipeline
def run_pipeline_worker(pdf_path, subject_code):
    global state
    with state_lock:
        state['status'] = 'processing'
        state['current_pdf'] = pdf_path
        state['subject_code'] = subject_code
        state['logs'] = [f"🚀 Starting pipeline for {pdf_path} with subject code {subject_code}...\n"]
        state['error'] = None

    try:
        # Run qbank_pipeline.py as a subprocess
        cmd = [sys.executable, 'qbank_pipeline.py', pdf_path, subject_code]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Read output in real-time
        for line in iter(process.stdout.readline, ''):
            with state_lock:
                state['logs'].append(line)
                # Keep logs bounded in memory if too large (e.g., last 2000 lines)
                if len(state['logs']) > 2000:
                    state['logs'].pop(0)

        process.stdout.close()
        return_code = process.wait()

        with state_lock:
            if return_code == 0:
                state['status'] = 'completed'
                state['logs'].append("🎉 Pipeline completed successfully!\n")
                # Create zip archive of results
                create_results_zip()
            else:
                state['status'] = 'failed'
                state['error'] = f"Pipeline process exited with code {return_code}"
                state['logs'].append(f"❌ Pipeline failed with exit code {return_code}\n")

    except Exception as e:
        with state_lock:
            state['status'] = 'failed'
            state['error'] = str(e)
            state['logs'].append(f"❌ Exception occurred: {str(e)}\n")

def create_results_zip():
    zip_path = 'output_results.zip'
    # Remove existing zip if any
    if os.path.exists(zip_path):
        try:
            os.remove(zip_path)
        except:
            pass

    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Zip data folder
            if os.path.exists('data'):
                for root, dirs, files in os.walk('data'):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.relpath(file_path, '.'))
            # Zip assets folder
            if os.path.exists('assets'):
                for root, dirs, files in os.walk('assets'):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.relpath(file_path, '.'))
        print("✅ output_results.zip created successfully!")
    except Exception as e:
        print(f"❌ Failed to create zip file: {e}")

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QBank PDF Extraction Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .terminal {
            background-color: #1e1e1e;
            color: #d4d4d4;
            font-family: 'Courier New', Courier, monospace;
        }
    </style>
</head>
<body class="bg-gray-50 min-h-screen text-gray-800">
    <nav class="bg-blue-600 text-white py-4 px-6 shadow-md">
        <div class="max-w-6xl mx-auto flex justify-between items-center">
            <h1 class="text-2xl font-bold tracking-tight">📚 QBank Extraction Pipeline</h1>
            <span class="bg-blue-700 px-3 py-1 rounded text-sm font-semibold">Railway Deployment</span>
        </div>
    </nav>

    <main class="max-w-6xl mx-auto px-4 py-8">
        
        <!-- Status Panel -->
        <div class="bg-white rounded-lg shadow p-6 mb-8 border border-gray-100">
            <h2 class="text-xl font-bold mb-4 flex items-center">
                <span class="mr-2">Status:</span>
                {% if state.status == 'idle' %}
                <span class="px-3 py-1 text-sm rounded bg-gray-200 text-gray-800 font-medium">Idle</span>
                {% elif state.status == 'processing' %}
                <span class="px-3 py-1 text-sm rounded bg-amber-500 text-white font-medium animate-pulse">Processing...</span>
                {% elif state.status == 'completed' %}
                <span class="px-3 py-1 text-sm rounded bg-green-500 text-white font-medium">Completed Successfully</span>
                {% elif state.status == 'failed' %}
                <span class="px-3 py-1 text-sm rounded bg-red-500 text-white font-medium">Failed</span>
                {% endif %}
            </h2>

            {% if state.current_pdf %}
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 bg-gray-50 p-4 rounded mb-4">
                <div>
                    <p class="text-sm text-gray-500">Processing PDF</p>
                    <p class="font-semibold text-lg break-all">{{ state.current_pdf }}</p>
                </div>
                <div>
                    <p class="text-sm text-gray-500">Subject Code</p>
                    <p class="font-semibold text-lg">{{ state.subject_code }}</p>
                </div>
            </div>
            {% endif %}

            <!-- Result Downloads -->
            {% if state.status == 'completed' or results_exist %}
            <div class="mt-6 border-t pt-4">
                <h3 class="text-lg font-semibold mb-3 text-gray-700">Download Results</h3>
                <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <a href="/download/chapters" class="flex items-center justify-center bg-blue-50 hover:bg-blue-100 text-blue-700 font-semibold py-3 px-4 rounded border border-blue-200 transition">
                        📄 chapters.json
                    </a>
                    <a href="/download/questions" class="flex items-center justify-center bg-blue-50 hover:bg-blue-100 text-blue-700 font-semibold py-3 px-4 rounded border border-blue-200 transition">
                        📄 questions.jsonl
                    </a>
                    <a href="/download/zip" class="flex items-center justify-center bg-emerald-600 hover:bg-emerald-700 text-white font-bold py-3 px-4 rounded shadow-sm transition">
                        📦 Download All (.ZIP)
                    </a>
                </div>
                <p class="text-xs text-gray-500 mt-2">The ZIP file contains the structured JSON files and all extracted and cropped images organized by subject.</p>
            </div>
            {% endif %}
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <!-- Left Side: Source PDFs & Upload -->
            <div class="lg:col-span-1 space-y-6">
                <!-- PDF Files List -->
                <div class="bg-white rounded-lg shadow p-6 border border-gray-100">
                    <h3 class="text-lg font-bold mb-3 text-gray-800">Detected PDF Files</h3>
                    <p class="text-sm text-gray-500 mb-4">These are PDF files found in your workspace or GitHub repo. Click run to process them.</p>
                    
                    {% if pdf_files %}
                    <div class="space-y-3">
                        {% for pdf in pdf_files %}
                        <div class="p-3 bg-gray-50 rounded border flex flex-col space-y-2">
                            <span class="font-medium text-sm break-all text-gray-700">{{ pdf }}</span>
                            <form action="/run" method="POST" class="flex gap-2">
                                <input type="hidden" name="pdf_path" value="{{ pdf }}">
                                <div class="flex-1">
                                    <input type="text" name="subject_code" value="{{ detect_subject_code(pdf) }}" class="w-full text-xs border rounded px-2 py-1 uppercase" placeholder="SUB" max="3" required>
                                </div>
                                <button type="submit" class="bg-blue-500 hover:bg-blue-600 text-white text-xs px-3 py-1 rounded font-semibold transition" {% if state.status == 'processing' %}disabled{% endif %}>
                                    Run
                                </button>
                            </form>
                        </div>
                        {% endfor %}
                    </div>
                    {% else %}
                    <div class="text-center p-6 bg-gray-50 border border-dashed rounded text-gray-500">
                        No PDF files found. Upload a PDF using the form below or commit a PDF to your GitHub repository!
                    </div>
                    {% endif %}
                </div>

                <!-- Manual Upload -->
                <div class="bg-white rounded-lg shadow p-6 border border-gray-100">
                    <h3 class="text-lg font-bold mb-3 text-gray-800">Upload New PDF</h3>
                    <form action="/upload" method="POST" enctype="multipart/form-data" class="space-y-4">
                        <div>
                            <label class="block text-sm font-semibold text-gray-600 mb-1">Select PDF File</label>
                            <input type="file" name="file" accept=".pdf" class="w-full text-sm border p-2 rounded bg-gray-50" required>
                        </div>
                        <div>
                            <label class="block text-sm font-semibold text-gray-600 mb-1">Subject Code (3 letters)</label>
                            <input type="text" name="subject_code" class="w-full text-sm border p-2 rounded uppercase" placeholder="e.g. PSY, BIO, CHE" maxlength="3">
                            <p class="text-xs text-gray-400 mt-1">If blank, it will be automatically guessed from the file name.</p>
                        </div>
                        <button type="submit" class="w-full bg-slate-800 hover:bg-slate-900 text-white font-bold py-2 px-4 rounded text-sm transition" {% if state.status == 'processing' %}disabled{% endif %}>
                            Upload and Run
                        </button>
                    </form>
                </div>
            </div>

            <!-- Right Side: Terminal / Logs -->
            <div class="lg:col-span-2">
                <div class="bg-white rounded-lg shadow overflow-hidden border border-gray-100 h-full flex flex-col">
                    <div class="bg-gray-800 px-4 py-3 text-white flex justify-between items-center">
                        <span class="font-semibold text-sm tracking-wide">Pipeline Console Logs</span>
                        <div class="flex items-center space-x-2">
                            <span class="h-2 w-2 rounded-full {% if state.status == 'processing' %}bg-amber-400 animate-pulse{% elif state.status == 'completed' %}bg-green-400{% else %}bg-gray-400{% endif %}"></span>
                            <span class="text-xs text-gray-300 uppercase">{{ state.status }}</span>
                        </div>
                    </div>
                    <div id="logs" class="terminal p-4 overflow-y-auto text-xs flex-1 h-[450px] whitespace-pre-wrap">{% for line in state.logs %}{{ line }}{% endfor %}</div>
                </div>
            </div>
        </div>
    </main>

    <footer class="bg-gray-100 text-center py-6 border-t mt-12 text-sm text-gray-500">
        <p>QBank PDF to JSON Automatic Extraction Pipeline Dashboard</p>
        <p class="text-xs text-gray-400 mt-1">Deploy on Railway • Auto-Detect Mode</p>
    </footer>

    <script>
        // Auto-refresh the page logs if pipeline is processing
        {% if state.status == 'processing' %}
        setInterval(function() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    const logsDiv = document.getElementById('logs');
                    const wasAtBottom = logsDiv.scrollHeight - logsDiv.clientHeight <= logsDiv.scrollTop + 20;
                    
                    logsDiv.textContent = data.logs.join('');
                    
                    if (wasAtBottom) {
                        logsDiv.scrollTop = logsDiv.scrollHeight;
                    }
                    
                    if (data.status !== 'processing') {
                        location.reload();
                    }
                });
        }, 2000);
        {% endif %}

        // Scroll logs to bottom initially
        const logsDiv = document.getElementById('logs');
        logsDiv.scrollTop = logsDiv.scrollHeight;
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    pdf_files = find_pdf_files()
    results_exist = os.path.exists('data/chapters.json') and os.path.exists('data/questions.jsonl')
    return render_template_string(
        HTML_TEMPLATE,
        state=state,
        pdf_files=pdf_files,
        detect_subject_code=detect_subject_code,
        results_exist=results_exist
    )

@app.route('/api/status')
def api_status():
    return jsonify({
        'status': state['status'],
        'current_pdf': state['current_pdf'],
        'subject_code': state['subject_code'],
        'logs': state['logs'],
        'error': state['error']
    })

@app.route('/run', methods=['POST'])
def trigger_run():
    global state
    if state['status'] == 'processing':
        return redirect(url_for('index'))

    pdf_path = request.form.get('pdf_path')
    subject_code = request.form.get('subject_code', 'SUB').upper()

    if not pdf_path or not os.path.exists(pdf_path):
        return redirect(url_for('index'))

    # Start pipeline in a background thread
    t = threading.Thread(target=run_pipeline_worker, args=(pdf_path, subject_code))
    t.daemon = True
    t.start()

    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload_file():
    global state
    if state['status'] == 'processing':
        return redirect(url_for('index'))

    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if file and file.filename.lower().endswith('.pdf'):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        # Determine subject code
        subject_code = request.form.get('subject_code')
        if not subject_code or len(subject_code.strip()) != 3:
            subject_code = detect_subject_code(filename)
        else:
            subject_code = subject_code.strip().upper()

        # Automatically start processing
        t = threading.Thread(target=run_pipeline_worker, args=(filename, subject_code))
        t.daemon = True
        t.start()

    return redirect(url_for('index'))

@app.route('/download/chapters')
def download_chapters():
    path = 'data/chapters.json'
    if os.path.exists(path):
        return send_file(path, as_attachment=True, download_name='chapters.json')
    return "File not found", 404

@app.route('/download/questions')
def download_questions():
    path = 'data/questions.jsonl'
    if os.path.exists(path):
        return send_file(path, as_attachment=True, download_name='questions.jsonl')
    return "File not found", 404

@app.route('/download/zip')
def download_zip():
    path = 'output_results.zip'
    # Generate the zip file on demand if not already generated
    if not os.path.exists(path):
        create_results_zip()
    
    if os.path.exists(path):
        return send_file(path, as_attachment=True, download_name='output_results.zip')
    return "File not found", 404

# Background startup scanner
def auto_start_scanner():
    """Scans for PDF files on startup and runs processing automatically for the first one found."""
    # Let the app initialize first
    import time
    time.sleep(2)
    
    pdfs = find_pdf_files()
    if pdfs and state['status'] == 'idle':
        # Select the first PDF
        target_pdf = pdfs[0]
        code = detect_subject_code(target_pdf)
        print(f"📡 [Auto-Scanner] Automatically detected PDF file: {target_pdf}")
        print(f"📡 [Auto-Scanner] Inferred Subject Code: {code}")
        print(f"📡 [Auto-Scanner] Starting background pipeline execution...")
        t = threading.Thread(target=run_pipeline_worker, args=(target_pdf, code))
        t.daemon = True
        t.start()

if __name__ == '__main__':
    # Start auto-scanner in background thread
    t = threading.Thread(target=auto_start_scanner)
    t.daemon = True
    t.start()

    # Get port from environment variable for Railway
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
