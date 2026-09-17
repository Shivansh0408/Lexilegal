import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app import create_app
from backend.core.LLM_judge import analyse_with_llm
from backend.core.analysis import analyse_pdf, find_relevant_sections, load_dictionary_sections
from backend.core.lawyer_analysis import analyse_pdf_for_lawyer
from backend.core.lawyer_LLM import _chunk_lawyer_text, analyse_with_lawyer_llm
from backend.core.response_parser import (
    extractive_lawyer_analysis,
    fallback_analysis,
    format_for_frontend,
    parse_and_merge,
    parse_and_merge_lawyer,
)
from backend.core.storage import analyse_pdf_with_cache


class ResponseParserTests(unittest.TestCase):
    def test_merges_without_fact_limit_and_deduplicates(self):
        first = {
            "case_title": "State v A",
            "case_overview": "A test file.",
            "facts": [{"fact": "Fact one."}, {"fact": "Fact two."}],
        }
        second = {
            "case_title": "State v A",
            "case_overview": "A test file.",
            "facts": [{"fact": "Fact two."}, {"fact": "Fact three."}],
        }
        result = parse_and_merge([json.dumps(first), json.dumps(second)])
        self.assertEqual(3, len(result["facts"]))

    def test_fallback_always_contains_one_fact(self):
        result = fallback_analysis("[Page 1]\nThe complainant filed a written report at the police station.", "offline")
        self.assertGreaterEqual(len(result["facts"]), 1)
        self.assertNotIn("Full LLM-based classification could not be completed.", result["missing_information"])

    def test_fallback_curates_multiple_material_facts(self):
        text = (
            "[Page 1]\nThe complainant alleged that the accused struck her arm on 18 June 2023. "
            "A doctor examined the complainant and recorded a contusion on the left forearm. "
            "The accused denies the assault and states that only a verbal argument occurred. "
            "The investigating officer did not obtain CCTV footage and did not seize the accused's phone."
        )
        result = fallback_analysis(text, "offline")
        self.assertGreaterEqual(len(result["facts"]), 3)
        self.assertTrue(result["missing_information"])

    def test_frontend_formatter_hides_section_identifiers(self):
        analysis = fallback_analysis("[Page 1]\nA sufficiently long factual sentence appears in the file.", "offline")
        result = format_for_frontend(analysis, {"filename": "case.pdf"}, ["The Indian Penal Code, 1860"])
        self.assertEqual(["The Indian Penal Code, 1860"], result["matched_laws"])
        self.assertNotIn("matched_sections", result)


class DictionaryTests(unittest.TestCase):
    def test_dictionary_is_loaded_from_collector_without_mutation(self):
        entries = load_dictionary_sections()
        self.assertGreater(len(entries), 100)
        matches = find_relevant_sections("The accused allegedly committed cheating and dishonestly took property.")
        self.assertTrue(matches)
        self.assertIn("law", matches[0])
        self.assertIn("section", matches[0])


class PipelineIntegrationTests(unittest.TestCase):
    @patch("backend.core.analysis.analyse_with_llm")
    def test_real_pdf_extraction_reaches_frontend_contract(self, llm_mock):
        llm_mock.return_value = ([json.dumps({
            "case_title": "Dictionary source test",
            "case_overview": "A source PDF was processed.",
            "facts": [{
                "fact": "The uploaded source contains statutory text.",
                "status": "supported",
                "confidence": "high",
            }],
        })], [])
        project_root = Path(__file__).resolve().parents[2]
        pdf_path = project_root / "backend" / "collector" / "downloads" / "india" / "criminal-law" / "indian-penal-code-act-1860" / "official_source.pdf"
        with pdf_path.open("rb") as source:
            result = analyse_pdf(source, "official_source.pdf")
        self.assertEqual(1, result["fact_count"])
        self.assertGreater(result["document"]["character_count"], 0)


class LlmResilienceTests(unittest.TestCase):
    @patch("backend.core.LLM_judge._call_ollama")
    def test_invalid_chunk_is_isolated_for_recovery(self, call_mock):
        valid = json.dumps({"facts": [{"fact": "A valid fact."}]})
        call_mock.side_effect = [valid, "not valid JSON"]
        long_document = ("The complainant reported an assault and supplied medical evidence. " * 220)
        outputs, failures = analyse_with_llm(long_document, [])
        self.assertEqual(1, len(outputs))
        self.assertGreaterEqual(len(failures), 1)
        self.assertIn("invalid JSON", failures[0]["error"])


class LawyerPipelineTests(unittest.TestCase):
    def test_page_aware_lawyer_chunks_are_small_and_keep_page_markers(self):
        text = "[Page 1]\n" + ("First page fact. " * 260) + "\n\n[Page 2]\n" + ("Second page fact. " * 220)
        chunks = _chunk_lawyer_text(text, 3000)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(len(chunk) <= 3000 for chunk in chunks))
        self.assertTrue(all(chunk.startswith("[Page ") for chunk in chunks))

    @patch("backend.core.lawyer_LLM._call_ollama")
    def test_lawyer_uses_small_generation_budget_and_stops_repeated_failures(self, call_mock):
        from backend.core.LLM_judge import LlmJudgeError

        call_mock.side_effect = LlmJudgeError("timed out")
        document = "\n\n".join(f"[Page {page}]\n" + ("Material allegation and evidence. " * 100) for page in range(1, 5))
        with patch.dict("os.environ", {"LAWYER_MAX_CONSECUTIVE_FAILURES": "1"}):
            outputs, failures = analyse_with_lawyer_llm(document, [])
        self.assertEqual([], outputs)
        self.assertEqual(1, call_mock.call_count)
        self.assertEqual(len(_chunk_lawyer_text(document, 3000)), len(failures))
        self.assertEqual(1200, call_mock.call_args.kwargs["num_predict"])
        self.assertEqual(8192, call_mock.call_args.kwargs["num_ctx"])
        self.assertEqual(180, call_mock.call_args.kwargs["timeout_seconds"])

    def test_lawyer_parser_merges_all_strategy_facts(self):
        first = {"strategy_facts": [{"fact": "Fact A"}], "next_actions": [{"action": "Inspect A"}]}
        second = {"strategy_facts": [{"fact": "Fact B"}], "next_actions": [{"action": "Inspect B"}]}
        result = parse_and_merge_lawyer([json.dumps(first), json.dumps(second)])
        self.assertEqual(2, len(result["strategy_facts"]))
        self.assertEqual(2, len(result["next_actions"]))

    def test_lawyer_fallback_produces_strategy_not_empty_message(self):
        text = (
            "[Page 1]\nThe complainant alleged that the accused received a package on 18 June 2023. "
            "The accused denies knowing its contents. The investigating officer seized the package and phone."
        )
        result = extractive_lawyer_analysis(text, "offline")
        self.assertGreaterEqual(len(result["strategy_facts"]), 1)
        self.assertTrue(result["next_actions"])
        self.assertTrue(result["ethics_and_safety"])

    @patch("backend.core.lawyer_analysis.analyse_with_lawyer_llm")
    def test_lawyer_pipeline_uses_shared_real_pdf_extraction(self, llm_mock):
        llm_mock.return_value = ([json.dumps({
            "case_title": "Strategy source test",
            "case_overview": "A source PDF was processed.",
            "client_position": "The represented side disputes the allegation.",
            "strategy_facts": [{"fact": "A source-grounded fact.", "status": "supported"}],
            "next_actions": [{"priority": "high", "action": "Inspect originals", "why": "Verify the record"}],
        })], [])
        project_root = Path(__file__).resolve().parents[2]
        pdf_path = project_root / "backend" / "collector" / "downloads" / "india" / "criminal-law" / "indian-penal-code-act-1860" / "official_source.pdf"
        with pdf_path.open("rb") as source:
            result = analyse_pdf_for_lawyer(source, "official_source.pdf")
        self.assertEqual(1, result["fact_count"])
        self.assertEqual("official_source.pdf", result["document"]["filename"])


class AnalysisStorageTests(unittest.TestCase):
    def test_identical_pdf_reuses_saved_pipeline_output(self):
        calls = []

        def analyser(_stream, filename):
            calls.append(filename)
            return {"document": {"filename": filename}, "facts": [{"fact": "Saved fact"}]}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("backend.core.storage.DATA_DIR", root), patch("backend.core.storage.UPLOAD_DIR", root / "uploads"), patch("backend.core.storage.OUTPUT_DIR", root / "outputs"):
                first, first_cached, _ = analyse_pdf_with_cache("judge", b"%PDF-test", "case.pdf", analyser)
                second, second_cached, _ = analyse_pdf_with_cache("judge", b"%PDF-test", "case.pdf", analyser)
        self.assertFalse(first_cached)
        self.assertTrue(second_cached)
        self.assertEqual(first["facts"], second["facts"])
        self.assertEqual(["case.pdf"], calls)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config.update(TESTING=True)
        self.client = self.app.test_client()

    def test_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(200, response.status_code)
        self.assertEqual("ok", response.get_json()["status"])

    def test_rejects_non_pdf(self):
        response = self.client.post(
            "/api/judge/analyze",
            data={"file": (io.BytesIO(b"not a pdf"), "case.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(400, response.status_code)

    @patch("backend.app.analyse_pdf")
    def test_returns_validated_analysis(self, analyse_pdf_mock):
        analyse_pdf_mock.return_value = {
            "facts": [{"fact": "One supported fact."}],
            "fact_count": 1,
            "document": {"filename": "case.pdf"},
        }
        response = self.client.post(
            "/api/judge/analyze",
            data={"file": (io.BytesIO(b"%PDF-pretend"), "case.pdf")},
            content_type="multipart/form-data",
        )
        self.assertEqual(200, response.status_code)
        self.assertEqual(1, response.get_json()["analysis"]["fact_count"])

    @patch("backend.app.analyse_pdf_with_cache")
    def test_lawyer_route_returns_saved_analysis_contract(self, cache_mock):
        cache_mock.return_value = (
            {"strategy_facts": [{"fact": "One strategy fact."}], "fact_count": 1, "document": {"filename": "case.pdf"}},
            True,
            {"document_id": "abc123", "saved_filename": "case.pdf", "result_file": "abc123.json"},
        )
        response = self.client.post(
            "/api/lawyer/analyze",
            data={"file": (io.BytesIO(b"%PDF-pretend"), "case.pdf")},
            content_type="multipart/form-data",
        )
        self.assertEqual(200, response.status_code)
        self.assertTrue(response.get_json()["cached"])
        self.assertEqual(1, response.get_json()["analysis"]["fact_count"])


if __name__ == "__main__":
    unittest.main()
