# 🤖 Railway Automation & Web Dashboard

You wanted a way for **Railway to automatically detect your uploaded PDF files on GitHub**, run the pipeline in the background without any manual commands, and make it super easy to view and download your results. 

We have upgraded your repository with a **fully automated web dashboard** (`app.py`), a modern `Dockerfile`, and updated dependencies!

---

## 🌟 How It Works

Instead of running manual commands in the Railway CLI or shell, your app now boots as a **dynamic Flask Web Application** with a beautiful UI.

1. **Automatic Detection:** On startup, the web app scans the repository for any `.pdf` files.
2. **Auto-Inference:** It automatically detects the subject code based on the filename (e.g. `PSY_Psychology.pdf` $\rightarrow$ `PSY`, `BIO_Biology_QBank.pdf` $\rightarrow$ `BIO`). If no code is prefixed, it uses the first three characters in uppercase.
3. **Background Execution:** It starts the extraction pipeline on a background thread so Railway doesn't time out.
4. **Live Logs:** It streams stdout logs from `qbank_pipeline.py` directly to a live-updating terminal on your dashboard.
5. **Easy Downloads:** Once completed, it generates a single **`output_results.zip`** file containing the complete `data/` and `assets/` folders, ready for download in one click!

---

## 📁 Upgraded Repository Structure

These new/modified files have been added/updated:
- **`app.py`**: The Flask server that provides the live-logging web dashboard, auto-detection logic, and download endpoints.
- **`Dockerfile`**: Upgraded to expose port `8080` and run `app.py` as the default startup script.
- **`requirements.txt`**: Added `Flask` and `gunicorn` for a production-ready, stable web server on Railway.

---

## 🚀 Setup & Deployment Guide

To deploy this updated code to your Railway app:

### 1. Copy the Files to Your GitHub Repo
Commit the updated/new files from this workspace into your GitHub repository:
- `app.py`
- `Dockerfile`
- `requirements.txt`

### 2. Configure Railway to Expose Your Web Dashboard
To see your dashboard in your browser, you need to expose a public domain on Railway:
1. Go to your **Railway Dashboard** and click on your service.
2. Go to **Settings**.
3. Under **Networking**, click **Generate Domain** (or set up a custom domain). 
4. Railway will automatically inject the `PORT` environment variable (the app is configured to listen on it).

### 3. Processing a PDF (Fully Automated)

#### Method A: Via GitHub Push (What you requested!)
1. Rename your PDF to start with a 3-letter uppercase subject code, for example:
   - `PSY_Psychology_QBank.pdf`
   - `BIO_Biology_QBank.pdf`
2. Commit and push the PDF file to the root of your GitHub repository.
3. **Railway will automatically detect the push**, rebuild the project using the upgraded `Dockerfile`, and deploy.
4. On startup, the app finds the PDF, automatically starts processing in the background, and streams the logs.
5. Open your Railway public domain in your browser. You will see the **live terminal** running the OCR in real-time!
6. Once the status changes to **"Completed"**, click **"Download All (.ZIP)"** to get your complete JSON files and cropped images in one package!

#### Method B: Via the Web Dashboard (Even easier!)
If you don't want to bloat your GitHub repository with heavy PDF files:
1. Push your updated code to GitHub (without any PDF files).
2. Once Railway deploys, open your public domain in your browser.
3. You will see a clean, empty state saying "No PDF files found."
4. Use the **"Upload New PDF"** form on the left:
   - Choose your PDF file.
   - Enter your 3-letter subject code (or leave it blank to auto-detect).
   - Click **"Upload and Run"**.
5. The PDF will upload directly to Railway and instantly start the extraction process while you watch the live logs!

---

## 🎯 Where Do I Receive My Results?

On your Railway Web Dashboard, you will have three instant download buttons as soon as processing completes:

1. 📄 **`chapters.json`**: Subject chapter metadata.
2. 📄 **`questions.jsonl`**: Every extracted question with its text, options, answers, solution, and image references.
3. 📦 **`Download All (.ZIP)`**: A single ZIP file containing the complete structured folder structure:
   ```
   data/
   ├── chapters.json
   └── questions.jsonl
   assets/
   ├── questions/{SUBJECT}/...
   ├── options/{SUBJECT}/...
   ├── solutions/{SUBJECT}/...
   └── tables/{SUBJECT}/...
   ```

No more digging around in Railway's shell or files tab! Everything is accessible through your custom web page in one click.
