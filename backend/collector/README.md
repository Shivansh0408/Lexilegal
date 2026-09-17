# LexBrief Drishti Bare Act PDF Collector

This collector creates an **India-only**, LLM-ready legal library from the
Drishti Judiciary Bare Acts catalogue. You enter one law type, such as
`Business Law`. The pipeline then:

1. opens the supplied Drishti catalogue URL with Beautiful Soup;
2. follows every discovered Bare Act category page;
3. extracts PDF links from the site's JavaScript `onclick` attributes;
4. selects canonical categories deterministically (for example, Criminal Law
   uses only Criminal Law and New Criminal Laws);
5. normalizes Act names and removes duplicate catalogue entries before selection;
6. skips an Act before downloading when a complete section index already exists;
7. downloads each remaining unique PDF only from `vault.drishtijudiciary.com`;
8. extracts every PDF page and rejects incomplete/blank-page extraction;
9. saves the complete extracted text beside the PDF;
10. parses the Act body into ordered statutory section numbers;
11. sends section-based batches to Llama and generates `what` for every section;
12. stitches the section summaries into the overall Act-level `what`; and
13. saves a final JSON record containing `law`, `what`, numbered sections and source metadata.

Large sections are never silently truncated. They are split into parts and
recombined. Progress is saved in `llm_section_checkpoint.json`, so a retry can
continue from completed sections instead of restarting the entire Act.

Acts with more than 160 parsed sections automatically use large-Act mode. This
uses smaller Llama batches, pauses between batches and retries temporary HTTP,
timeout and invalid-JSON failures. The collector remains locked on that Act and
resumes its checkpoint; it will not move to the next Act until the current Act
is complete. If recovery is exhausted, the run stops safely and can be resumed
with the same command.

## 1. Install

Run these commands from the project root (the directory containing `backend`).

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend/collector/requirements.txt
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/collector/requirements.txt
```

## 2. Local Llama setup

```bash
ollama pull llama3.2:3b
ollama serve
```

The crawler can still discover catalogue entries if Ollama is temporarily
unavailable, using deterministic category/name matching. Final `law`/`what`
records require Ollama because the full extracted PDF is processed by the LLM.

## 3. Interactive run

```bash
python backend/collector/collector.py
```

The first prompt now lets you keep the existing law-type pipeline or search for
one specific Act:

```text
1. Existing pipeline: collect every Act related to a law type
2. Specific Act search: collect and index only one typed Act
Choose collection mode (1 or 2) [1]:
```

Choose `1` to receive the existing `Law type` and `Maximum Acts` prompts.
Choose `2`, then type an Act name such as `Indian Penal Code`. The catalogue is
searched directly and only the single matched Act is downloaded and indexed.
An ambiguous or unknown name stops with matching names/suggestions so the
collector never silently indexes a different Act.

There is no country or state prompt. The jurisdiction is always India.

## 4. Non-interactive run

```bash
python backend/collector/collector.py --law-type "Business Law" --max-acts 20
```

To collect only one Act without prompts, use `--act-name`:

```bash
python backend/collector/collector.py --act-name "Indian Penal Code"
```

Aliases such as `IT Act` are supported. `--max-acts` is ignored in this mode
because a specific-Act run always has exactly one candidate.

By default, every discovered candidate is processed. You can state that
explicitly with `--max-acts 0`:

```bash
python backend/collector/collector.py --law-type "Business Law" --max-acts 0
```

Llama settings can be placed in `backend/collector/.env`:

```env
LLAMA_API_URL=http://localhost:11434/api/generate
LLAMA_MODEL=llama3.2:3b
```

The Llama service must be available for final `law`/`what` generation.

The default read timeout is 300 seconds. Recovery behaviour can be tuned when
needed:

```bash
python backend/collector/collector.py --law-type "Criminal Law" --max-acts 5 \
  --llama-timeout 300 --llama-max-retries 4 --act-max-retries 3
```

Do not delete the download folders or `llm_section_checkpoint.json` files before
rerunning. Complete Acts will be skipped and partial Acts will resume from their
last successfully indexed section.

## 5. Output structure

For `Business Law`, final dictionaries are written to:

```text
backend/collector/dictionary/india/business-law/
```

Example:

```text
business-law/
├── discovery_manifest.json
├── index.json
├── the-companies-act-2013.json
├── the-indian-contract-act-1872.json
└── the-sale-of-goods-act-1930.json
```

Downloaded and extracted evidence is stored separately under:

```text
backend/collector/downloads/india/business-law/<act-name>/
├── bare_act.pdf
├── extracted_text.txt
├── parsed_sections.json
├── llm_section_checkpoint.json
└── extraction_report.json
```

Each Act JSON has this structure:

```json
{
  "country": "India",
  "law_type": "Business Law",
  "law": "The Indian Contract Act, 1872",
  "what": "Comprehensive LLM-generated explanation grounded in the complete PDF...",
  "section_count": 238,
  "sections": {
    "1": {
      "section": "1",
      "title": "Short title",
      "what": "This section states...",
      "chapter": "CHAPTER I"
    }
  },
  "key_points": ["Important statutory rule..."],
  "source": {
    "name": "Drishti Judiciary Bare Acts",
    "url": "Drishti vault PDF URL",
    "format": "pdf",
    "content_sha256": "..."
  }
}
```

`index.json` lists every successfully collected Act and every failed candidate.
`discovery_manifest.json` keeps the selected candidates and the fully crawled
Drishti catalogue.

## Safeguards and limitations

- Catalogue pages are restricted to `drishtijudiciary.com`.
- PDF downloads are restricted to `vault.drishtijudiciary.com`.
- Downloads are size-limited and content-hashed.
- Duplicate Act titles and aliases are collapsed to one canonical Act before
  selection, download and final index generation.
- Existing complete Act records are skipped before the download stage; existing
  matching PDFs and extracted text for partial Acts are reused.
- Ollama requests retry transient connection errors, read timeouts and malformed
  JSON responses with exponential backoff.
- A failed partial Act stops the run instead of allowing the collector to move
  on and leave gaps in the requested sequence.
- Every PDF page must yield machine-readable text; incomplete/scanned PDFs are
  reported as failures instead of silently producing partial legal data.
- The catalogue manifest makes the complete crawl and selection auditable.
- This output is a research dataset, not legal advice. Always verify current
  amendments, commencement notifications, repeal status, and formatting with
  the cited official source.
