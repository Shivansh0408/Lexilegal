"""Tests for the LangChain and Chroma saved-analysis RAG integration."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.core import chat
from langchain_core.embeddings import DeterministicFakeEmbedding


SAMPLE_ID = "a" * 64
SAMPLE_RECORD = {
    "document_id": SAMPLE_ID,
    "pipeline": "judge",
    "analysis": {
        "case_title": "State v. Example",
        "case_overview": "A fictional prosecution based on a disputed recovery.",
        "facts": [{
            "fact": "The recovery is alleged to have occurred at 20:15.",
            "status": "disputed",
            "support": "Seizure memo",
        }],
        "evidence": [{"item": "Seizure memo", "reliability_note": "Signature is disputed."}],
        "missing_information": ["Independent witness statement"],
    },
}


class AnalysisDocumentTests(unittest.TestCase):
    def test_json_is_converted_to_scoped_traceable_documents(self):
        documents = chat.analysis_documents(SAMPLE_RECORD, Path("saved.json"))
        self.assertGreaterEqual(len(documents), 4)
        self.assertTrue(all(item.metadata["scope"] == f"judge:{SAMPLE_ID}" for item in documents))
        self.assertTrue(any(item.metadata["section"] == "facts" for item in documents))
        self.assertTrue(any("20:15" in item.page_content for item in documents))
        self.assertTrue(any(item.metadata["json_path"] == "analysis.evidence[0]" for item in documents))

    def test_invalid_saved_record_is_rejected(self):
        with self.assertRaises(ValueError):
            chat.analysis_documents({"analysis": {}}, Path("bad.json"))

    def test_identity_question_uses_saved_party_role_without_generation(self):
        analysis = {
            "case_title": "State v. Arjun Mehra",
            "parties": [
                {"name": "Arjun Mehra", "role": "Accused", "description": "husband of the complainant"},
                {"name": "Naina Mehra", "role": "Complainant", "description": ""},
            ],
        }
        result = chat._identity_answer(analysis, "Who is Arjun Mehra?")
        self.assertIsNotNone(result)
        self.assertIn("Accused", result["answer"])
        self.assertNotIn("judge", result["answer"].lower())

    def test_previous_assistant_output_is_not_reused_as_evidence(self):
        messages = chat._history_messages([
            {"role": "user", "content": "Who is the accused?"},
            {"role": "assistant", "content": "John Smith is the judge."},
        ])
        self.assertEqual(1, len(messages))
        self.assertEqual("Who is the accused?", messages[0].content)

    def test_unknown_person_is_not_invented(self):
        analysis = {"case_title": "State v. Arjun Mehra", "parties": []}
        result = chat._identity_answer(analysis, "Who is John Smith?")
        self.assertIsNotNone(result)
        self.assertIn("not available", result["answer"])


class RagFacadeTests(unittest.TestCase):
    def test_saved_json_is_persisted_and_retrievable_in_chroma(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved_path = root / "outputs" / "judge" / f"{SAMPLE_ID}.json"
            saved_path.parent.mkdir(parents=True)
            saved_path.write_text(json.dumps(SAMPLE_RECORD), encoding="utf-8")

            service = object.__new__(chat.SavedAnalysisRagService)
            service._lock = threading.RLock()
            service._embedding_model = "deterministic-test-embedding"
            service._splitter_version = "test-v1"
            service._vectorstore = chat.Chroma(
                collection_name="lexbrief_test_collection",
                embedding_function=DeterministicFakeEmbedding(size=64),
                persist_directory=str(root / "chroma"),
            )

            with patch.object(chat, "DATA_DIR", root), patch.object(chat, "MANIFEST_PATH", root / "rag_manifest.json"):
                chunk_count = service.index_file(saved_path)
                documents = service._retrieve(f"judge:{SAMPLE_ID}", "disputed seizure memo signature")
                unchanged_count = service.index_file(saved_path)

            self.assertGreater(chunk_count, 0)
            self.assertEqual(chunk_count, unchanged_count)
            self.assertTrue(documents)
            self.assertTrue(all(item.metadata["document_id"] == SAMPLE_ID for item in documents))

    @patch("backend.core.chat.get_rag_service")
    def test_answer_is_delegated_to_rag_service(self, get_service):
        get_service.return_value.answer.return_value = {
            "answer": "The signature is disputed [Context 1].",
            "source": "saved_analysis",
            "sources": [{"context": 1, "section": "evidence"}],
        }
        result = chat.answer_saved_analysis_question(
            "judge",
            SAMPLE_ID,
            "Was the signature accepted?",
            [{"role": "user", "content": "Tell me about the seizure memo."}],
        )
        self.assertEqual("saved_analysis", result["source"])
        get_service.return_value.answer.assert_called_once()

    @patch("backend.core.chat.get_rag_service")
    def test_index_initialization_failure_becomes_rag_error(self, get_service):
        get_service.side_effect = RuntimeError("Chroma unavailable")
        with self.assertRaises(chat.RagError):
            chat.index_saved_analysis("judge", SAMPLE_ID)


class ChatRouteTests(unittest.TestCase):
    @patch("backend.app.index_saved_analysis", return_value=3)
    @patch("backend.app.answer_saved_analysis_question")
    def test_chat_route_returns_rag_answer(self, answer, _index):
        import backend.app as app_module

        answer.return_value = {
            "answer": "Curated RAG answer",
            "source": "saved_analysis",
            "retrieval_query": "disputed signature seizure memo",
            "sources": [],
        }
        client = app_module.create_app().test_client()
        response = client.post(
            "/api/chat",
            data=json.dumps({
                "pipeline": "judge",
                "document_id": SAMPLE_ID,
                "question": "What about its signature?",
                "history": [{"role": "user", "content": "Explain the seizure memo."}],
            }),
            content_type="application/json",
        )
        self.assertEqual(200, response.status_code)
        self.assertEqual("saved_analysis", response.get_json()["source"])

    @patch("backend.app.index_saved_analysis", return_value=3)
    @patch("backend.app.answer_saved_analysis_question", side_effect=chat.RagError("Ollama unavailable"))
    def test_rag_failure_returns_service_unavailable(self, _answer, _index):
        import backend.app as app_module

        client = app_module.create_app().test_client()
        response = client.post("/api/chat", json={
            "pipeline": "judge",
            "document_id": SAMPLE_ID,
            "question": "What evidence exists?",
            "history": [],
        })
        self.assertEqual(503, response.status_code)
        self.assertIn("Ollama unavailable", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
