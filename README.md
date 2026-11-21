# Product Manual Assistant

A beginner-friendly Streamlit chatbot that answers appliance troubleshooting questions from an uploaded PDF manual. It extracts text with PyMuPDF, keeps each chunk's original 1-based PDF page number, embeds chunks with Sentence Transformers, searches them with a FAISS cosine-similarity index, and sends only the retrieved excerpts to Google Gemini for a grounded answer.

## What happens when you ask a question?

1. PyMuPDF reads text from every PDF page and records that page's number.
2. The app divides each page's text into overlapping word chunks. The overlap helps keep instructions that cross chunk boundaries together.
3. Sentence Transformers turns each chunk into a numerical vector. These embeddings run locally; the first run downloads the model.
4. FAISS searches the vectors for the chunks closest to your question.
5. Gemini receives the question and only those retrieved manual excerpts. Its instructions tell it to say when the manual does not contain the answer.
6. The UI displays the retrieved source pages so you can check the manual yourself.

The index exists in memory for the current Streamlit session. Uploading a different manual creates a new index. No manual or index is saved to disk by this app.

## Requirements

- Python 3.10 or newer
- A Google Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
- Internet access for installing packages, downloading the embedding model the first time, and calling Gemini

## Setup on Windows (PowerShell)

Open PowerShell in this project folder, then run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

In `.env`, replace `your_gemini_api_key_here` with your key and save. Do not add quotes. `.env` is ignored by Git so your key is not included in commits.

If PowerShell prevents environment activation, you can skip activation and use `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`, then launch with `.\.venv\Scripts\python.exe -m streamlit run app.py`.

## Run the app

```powershell
streamlit run app.py
```

Your browser should open the app. Upload a text-based appliance manual PDF, choose **Process manual**, then ask a question in the chat box. Processing a large PDF can take a little time, especially the first time the embedding model is downloaded.

## Notes and limitations

- Scanned PDFs containing only page images are not OCR processed. Use a PDF with selectable text.
- Source pages are PDF page numbers, starting at 1. Retrieved pages are shown alongside each answer; they are the evidence sent to Gemini, not a guarantee that every answer is correct.
- The assistant is designed to stay within manual content. Verify important repair and safety steps in the cited manual and follow its warnings.
- Gemini usage may be subject to Google AI Studio quotas and billing settings.
- On systems where `faiss-cpu` has no compatible wheel, use a supported Python version and consult the official FAISS installation instructions.

## Project files

- `app.py` — Streamlit layout, upload flow, chat display, and user-facing error handling.
- `rag_engine.py` — PDF parsing, page-aware chunking, local embeddings, FAISS retrieval, and Gemini generation.
- `requirements.txt` — Python packages used by the project.
- `.env.example` — safe template for the private Gemini API key file.
- `.gitignore` — prevents keys, virtual environments, caches, and PDFs from being committed.
