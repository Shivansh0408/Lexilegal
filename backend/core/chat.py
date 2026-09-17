"""LangChain + Chroma RAG chatbot over persisted case-analysis JSON files."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .storage import DATA_DIR, OUTPUT_DIR, load_saved_analysis

LOGGER = logging.getLogger("lexbrief.rag")
CHROMA_DIR = DATA_DIR / "chroma_db"
MANIFEST_PATH = DATA_DIR / "rag_manifest.json"
COLLECTION_NAME = "lexbrief_saved_analyses"
_SERVICE = None
_SERVICE_LOCK = threading.Lock()


class RagError(RuntimeError):
    """Raised when indexing, retrieval, or grounded generation fails."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ollama_base_url() -> str:
    configured = os.getenv("OLLAMA_BASE_URL", "").strip()
    if configured:
        return configured.rstrip("/")
    generate_url = os.getenv("LLAMA_API_URL", "http://localhost:11434/api/generate").rstrip("/")
    if "/api/" in generate_url:
        return generate_url.split("/api/", 1)[0]
    return generate_url


def _message_text(message) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        pieces = []
        for block in content:
            if isinstance(block, str):
                pieces.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                pieces.append(block["text"])
        return "\n".join(pieces).strip()
    return str(content).strip()


def _history_messages(history: Iterable[dict], maximum: int = 8):
    messages = []
    for item in list(history)[-maximum:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()[:2_000]
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def _json_value_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def analysis_documents(record: dict, source_path: Path) -> list[Document]:
    """Convert one saved result JSON into field-aware, traceable LangChain documents."""

    document_id = str(record.get("document_id", "")).strip().lower()
    pipeline = str(record.get("pipeline", "")).strip().lower()
    analysis = record.get("analysis")
    if pipeline not in {"judge", "lawyer"} or len(document_id) != 64 or not isinstance(analysis, dict):
        raise ValueError(f"Invalid saved analysis record: {source_path.name}")

    case_title = str(analysis.get("case_title") or "Untitled case").strip()
    scope = f"{pipeline}:{document_id}"
    base_metadata = {
        "document_id": document_id,
        "pipeline": pipeline,
        "scope": scope,
        "case_title": case_title[:500],
        "source_file": source_path.name,
    }
    raw_documents: list[Document] = []

    for field, value in analysis.items():
        if value in (None, "", [], {}):
            continue
        label = field.replace("_", " ").title()
        items = value if isinstance(value, list) else [value]
        for index, item in enumerate(items):
            path = f"analysis.{field}[{index}]" if isinstance(value, list) else f"analysis.{field}"
            content = (
                f"Case: {case_title}\n"
                f"Analysis type: {pipeline}\n"
                f"Section: {label}\n"
                f"JSON path: {path}\n"
                f"Saved analysis content:\n{_json_value_text(item)}"
            )
            raw_documents.append(Document(
                page_content=content,
                metadata={**base_metadata, "section": field, "json_path": path},
            ))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max(600, int(os.getenv("RAG_CHUNK_SIZE", "1400"))),
        chunk_overlap=max(50, int(os.getenv("RAG_CHUNK_OVERLAP", "180"))),
        separators=["\n\n", "\n", ". ", "; ", ", ", " "],
    )
    return splitter.split_documents(raw_documents)


class SavedAnalysisRagService:
    """Persistent JSON indexing, scoped retrieval, query understanding, and synthesis."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._embedding_model = os.getenv("RAG_EMBEDDING_MODEL", "nomic-embed-text").strip()
        self._chat_model = os.getenv("LLAMA_MODEL", "llama3.2:3b").strip()
        self._base_url = _ollama_base_url()
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self._embeddings = OllamaEmbeddings(model=self._embedding_model, base_url=self._base_url)
        self._vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self._embeddings,
            persist_directory=str(CHROMA_DIR),
            collection_metadata={"hnsw:space": "cosine"},
        )
        self._llm = ChatOllama(
            model=self._chat_model,
            base_url=self._base_url,
            temperature=0,
            num_ctx=max(4_096, int(os.getenv("RAG_NUM_CTX", "8192"))),
            num_predict=max(256, int(os.getenv("RAG_NUM_PREDICT", "900"))),
        )
        self._splitter_version = "json-fields-v1"

    @staticmethod
    def _read_manifest() -> dict:
        try:
            payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _write_manifest(payload: dict) -> None:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = MANIFEST_PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, MANIFEST_PATH)

    def index_file(self, path: Path, force: bool = False) -> int:
        """Embed one saved JSON when its bytes or indexing configuration changed."""

        path = path.resolve()
        try:
            raw = path.read_bytes()
            record = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RagError(f"Could not read saved analysis {path.name}: {exc}") from exc

        document_id = str(record.get("document_id", "")).strip().lower()
        pipeline = str(record.get("pipeline", "")).strip().lower()
        scope = f"{pipeline}:{document_id}"
        fingerprint = hashlib.sha256(
            raw + self._embedding_model.encode("utf-8") + self._splitter_version.encode("utf-8")
        ).hexdigest()
        manifest_key = str(path.relative_to(DATA_DIR)).replace("\\", "/")

        with self._lock:
            manifest = self._read_manifest()
            current = manifest.get(manifest_key, {})
            if not force and current.get("fingerprint") == fingerprint:
                return int(current.get("chunk_count", 0))

            documents = analysis_documents(record, path)
            if not documents:
                raise RagError(f"Saved analysis {path.name} contains no indexable content.")
            ids = [
                hashlib.sha256(f"{scope}:{index}:{fingerprint}".encode("utf-8")).hexdigest()
                for index in range(len(documents))
            ]
            try:
                self._vectorstore.delete(where={"scope": scope})
                self._vectorstore.add_documents(documents=documents, ids=ids)
            except Exception as exc:
                raise RagError(
                    f"Chroma indexing failed. Confirm Ollama is running and the embedding model "
                    f"'{self._embedding_model}' is installed. Details: {exc}"
                ) from exc

            manifest[manifest_key] = {
                "document_id": document_id,
                "pipeline": pipeline,
                "scope": scope,
                "fingerprint": fingerprint,
                "chunk_count": len(documents),
                "indexed_at": _utc_now(),
            }
            self._write_manifest(manifest)
            LOGGER.info("Indexed %s into %d Chroma chunk(s)", manifest_key, len(documents))
            return len(documents)

    def index_analysis(self, pipeline: str, document_id: str, force: bool = False) -> int:
        load_saved_analysis(pipeline, document_id)
        path = OUTPUT_DIR / pipeline / f"{document_id}.json"
        return self.index_file(path, force=force)

    def sync_all(self, force: bool = False) -> dict:
        """Index every valid JSON currently present under saved_analysis/outputs."""

        files = sorted(OUTPUT_DIR.glob("*/*.json"))
        indexed = 0
        chunks = 0
        errors = []
        for path in files:
            try:
                chunks += self.index_file(path, force=force)
                indexed += 1
            except (RagError, ValueError) as exc:
                LOGGER.error("RAG sync skipped %s: %s", path, exc)
                errors.append({"file": path.name, "error": str(exc)})
        return {"files": len(files), "indexed": indexed, "chunks": chunks, "errors": errors}

    def _rewrite_query(self, question: str, history: Iterable[dict]) -> str:
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Rewrite the latest user question as a short standalone semantic-search query for a legal "
                "case-analysis vector database. Resolve pronouns and follow-up references only from the chat "
                "history. Preserve names, dates, document labels, statutory references, evidence terms, and "
                "whether the user asks for judge facts or lawyer strategy. Return only the search query.",
            ),
            ("placeholder", "{history}"),
            ("human", "{question}"),
        ])
        try:
            message = (prompt | self._llm).invoke({
                "history": _history_messages(history),
                "question": question,
            })
            rewritten = _message_text(message).strip('"').strip()
            return rewritten[:2_000] or question
        except Exception as exc:
            LOGGER.warning("Query rewriting failed; using the original question: %s", exc)
            return question

    def _retrieve(self, scope: str, query: str) -> list[Document]:
        count = max(3, min(20, int(os.getenv("RAG_RETRIEVAL_K", "10"))))
        fetch_k = max(count, min(60, int(os.getenv("RAG_FETCH_K", "30"))))
        try:
            retriever = self._vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": count,
                    "fetch_k": fetch_k,
                    "lambda_mult": 0.65,
                    "filter": {"scope": scope},
                },
            )
            return retriever.invoke(query)
        except Exception as exc:
            raise RagError(f"Chroma retrieval failed: {exc}") from exc

    def answer(
        self,
        pipeline: str,
        document_id: str,
        question: str,
        history: Iterable[dict] = (),
    ) -> dict:
        chunk_count = self.index_analysis(pipeline, document_id)
        scope = f"{pipeline}:{document_id}"
        retrieval_query = self._rewrite_query(question, history)
        documents = self._retrieve(scope, retrieval_query)
        if not documents:
            return {
                "answer": "I could not find information relevant to that question in the saved analysis.",
                "source": "rag",
                "retrieval_query": retrieval_query,
                "sources": [],
            }

        context_blocks = []
        sources = []
        for index, document in enumerate(documents, 1):
            section = str(document.metadata.get("section", "unknown"))
            json_path = str(document.metadata.get("json_path", "analysis"))
            context_blocks.append(
                f"[Retrieved context {index} | section={section} | path={json_path}]\n{document.page_content}"
            )
            sources.append({
                "context": index,
                "section": section,
                "json_path": json_path,
                "document_id": document_id,
                "pipeline": pipeline,
            })

        answer_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are LexBrief's saved-analysis RAG assistant. Understand the user's actual question and "
                "stitch a clear, curated answer using ONLY the retrieved saved-analysis context. Combine related "
                "facts across chunks, remove repetition, preserve the difference between allegations, disputed "
                "facts, supported material, missing information, and strategy options. Never add outside legal "
                "knowledge, invent facts, decide guilt, or claim a law applies unless the retrieved analysis says "
                "so. If retrieval does not contain the answer, state that it is not available in the saved "
                "analysis. When useful, cite retrieved blocks as [Context 1], [Context 2]. Answer the question "
                "directly and use concise headings or bullets only when they improve clarity."
            ),
            ("placeholder", "{history}"),
            (
                "human",
                "Original question:\n{question}\n\nRetrieval query used:\n{retrieval_query}\n\n"
                "Retrieved saved-analysis context:\n{context}",
            ),
        ])
        try:
            response = (answer_prompt | self._llm).invoke({
                "history": _history_messages(history),
                "question": question,
                "retrieval_query": retrieval_query,
                "context": "\n\n".join(context_blocks),
            })
        except Exception as exc:
            raise RagError(
                f"Llama could not generate the RAG answer. Confirm Ollama is running and model "
                f"'{self._chat_model}' is installed. Details: {exc}"
            ) from exc

        answer = _message_text(response)
        if not answer:
            raise RagError("Llama returned an empty RAG answer.")
        return {
            "answer": answer,
            "source": "langchain_chroma_rag",
            "retrieval_query": retrieval_query,
            "indexed_chunks": chunk_count,
            "sources": sources,
        }


def get_rag_service() -> SavedAnalysisRagService:
    global _SERVICE
    if _SERVICE is None:
        with _SERVICE_LOCK:
            if _SERVICE is None:
                _SERVICE = SavedAnalysisRagService()
    return _SERVICE


def index_saved_analysis(pipeline: str, document_id: str, force: bool = False) -> int:
    try:
        return get_rag_service().index_analysis(pipeline, document_id, force=force)
    except (RagError, ValueError, FileNotFoundError):
        raise
    except Exception as exc:
        raise RagError(f"The RAG index could not be initialized: {exc}") from exc


def sync_saved_analyses(force: bool = False) -> dict:
    try:
        return get_rag_service().sync_all(force=force)
    except RagError:
        raise
    except Exception as exc:
        raise RagError(f"The RAG index could not be initialized: {exc}") from exc


def answer_saved_analysis_question(
    pipeline: str,
    document_id: str,
    question: str,
    history: Iterable[dict] = (),
) -> dict:
    try:
        return get_rag_service().answer(pipeline, document_id, question, history)
    except (RagError, ValueError, FileNotFoundError):
        raise
    except Exception as exc:
        raise RagError(f"The saved-analysis RAG request failed: {exc}") from exc
