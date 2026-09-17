# LexBrief judge and lawyer PDF-analysis backend

This backend is separate from `collector/`. The collector continues to build the
legal dictionary; the API reads that dictionary without changing it.

## Files

- `app.py` – Flask entry point and upload API.
- `core/pdf_processing.py` – complete text extraction with `pypdf`.
- `core/analysis.py` – pipeline orchestration and dictionary retrieval.
- `core/LLM_judge.py` – chunked Ollama fact extraction.
- `core/lawyer_LLM.py` – independent, generalized lawyer strategy extraction prompt.
- `core/lawyer_analysis.py` – lawyer pipeline orchestration using the shared PDF and response layers.
- `core/response_parser.py` – JSON parsing, validation, de-duplication, fallback,
  and the final frontend contract.
- `core/storage.py` – content-hash upload/result persistence and cache reuse.
- `core/chat.py` – LangChain loading/chunking, persistent Chroma retrieval, Llama query rewriting, and grounded answer synthesis.
- `index_saved_analyses.py` – command-line index builder for every existing saved-analysis JSON.

## Saved-analysis chatbot

Every completed judge or lawyer analysis is converted into field-aware LangChain documents and embedded into a persistent Chroma database under `saved_analysis/chroma_db`. Chat retrieval is filtered to the exact pipeline and `document_id` selected by the frontend, so two case files cannot leak into each other's answers. Llama first rewrites follow-up questions into standalone retrieval queries, then stitches the retrieved JSON chunks into a curated answer without adding outside information.

Install and prepare the two local Ollama models:

```powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Build the Chroma index for all JSON files already present in `saved_analysis/outputs`:

```powershell
cd backend
python index_saved_analyses.py
```

New analyses are indexed automatically by the upload endpoint. To rebuild every entry, run the command with `--force` or call `POST /api/chat/reindex` with `{ "force": true }`.

## Run on Windows PowerShell

From the project root:

```powershell
cd backend
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
ollama pull llama3.2:3b
ollama serve
```

Open a second PowerShell window in the project root:

```powershell
cd backend
.venv\Scripts\Activate.ps1
python app.py
```

The API runs at `http://localhost:5000`.

## API

`POST /api/judge/analyze` accepts `multipart/form-data` with a field named
`file`. Only PDFs are accepted. The response contains `analysis.facts` with at
least one fact and no application-level maximum number of fact items.

`POST /api/lawyer/analyze` accepts the same upload contract and returns a
source-grounded legal strategy brief in `analysis`. The Judge and Lawyer caches
are separate. Uploaded PDFs are stored under `backend/saved_analysis/uploads`,
and completed JSON results under `backend/saved_analysis/outputs/<pipeline>`.
An identical PDF uploaded again to the same pipeline reuses the saved output
without calling the LLM again. Set `ANALYSIS_DATA_DIR` to override the directory.

Text-based PDFs are supported. Image-only scans must be OCRed before upload.

The LLM receives smaller document chunks so CPU-hosted Ollama models do not have
to classify the entire file in one oversized request. A timeout or invalid JSON
affects only that chunk. Completed LLM chunks are retained, while failed chunks
receive multi-fact extractive recovery. Processing failures are reported under
analysis limitations and are never inserted as missing case information.

The Lawyer pipeline uses page-aware chunks capped at 3,000 characters by
default and a separate 1,200-token generation budget. Configure these with
`LAWYER_CHUNK_CHARACTERS`, `LAWYER_NUM_PREDICT`, `LAWYER_NUM_CTX`, and
`LAWYER_TIMEOUT_SECONDS`. After the configured number of consecutive failures,
remaining chunks immediately use structured extractive recovery instead of
waiting through repeated timeouts.

## Test

From the project root with the backend environment active:

```powershell
python -m unittest backend.tests.test_chat backend.tests.test_judge_pipeline -v
```
