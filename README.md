# LexBrief

LexBrief contains two independent PDF workflows selected from the frontend:

- **Judge** extracts neutral, decision-relevant facts and evidence issues.
- **Lawyer** extracts source-grounded facts and builds a working legal-strategy brief.

Only the selected role's upload and output components are mounted. Both backend
pipelines use the same page-aware PDF extraction and response post-processing
module, but have separate LLM prompts and separate saved outputs.

## Start the backend

```powershell
cd backend
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python app.py
```

Run `ollama serve` separately and pull both the chat and embedding models:

```powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

## Start the frontend

From the project root in a second PowerShell window:

```powershell
npm install
npm start
```

The frontend uses `http://localhost:5000` by default. To override it, create a
root `.env` containing `REACT_APP_API_BASE_URL=http://your-backend-host:5000`.

## Saved analysis

The backend saves uploads and JSON results beneath `backend/saved_analysis`.
Uploading identical PDF bytes to the same pipeline reuses the saved result;
Judge and Lawyer outputs are cached independently. Configure a different durable
location with `ANALYSIS_DATA_DIR` when deploying.

The chatbot indexes these saved JSON outputs as LangChain documents in a
persistent Chroma database. To index existing saved analyses before starting
the API, run `python index_saved_analyses.py` from the `backend` folder.
