import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from backend.collector.collector import find_specific_catalog_act
from backend.collector.llama_processor import LlamaProcessor
from backend.collector.dictionary_builder import IndianActDictionaryBuilder
from backend.collector.drishti_crawler import DrishtiBareAct, DrishtiBareActCrawler
from backend.collector.pdf_py import DrishtiPdfProcessor
from backend.collector.section_parser import build_act_record, parse_ordered_bare_act_sections, parse_sections


SAMPLE_ACT = """
THE EXAMPLE ACT, 2026
ACT NO. 7 OF 2026
ARRANGEMENT OF SECTIONS
1. Short title.
2. Definitions.

CHAPTER I PRELIMINARY
1. Short title.—This Act may be called the Example Act, 2026.
2. Definitions.—In this Act, unless the context otherwise requires, “company” means an example company.
CHAPTER II CONTRACTS
3. Valid contracts.—A contract is valid when the stated statutory conditions are satisfied.
3A. Electronic contracts.—An electronic contract shall not be denied validity solely because it is electronic.
"""


class SectionParserTests(unittest.TestCase):
    def test_arrangement_is_deduplicated_and_body_is_kept(self):
        sections = parse_sections(SAMPLE_ACT)
        self.assertEqual(["1", "2", "3", "3A"], list(sections))
        self.assertIn("may be called", sections["1"]["what"])
        self.assertEqual("Electronic contracts", sections["3A"]["title"])

    def test_record_has_required_key_value_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "act.html"
            source.write_text(f"<main><pre>{SAMPLE_ACT}</pre></main>", encoding="utf-8")
            record = build_act_record(
                source,
                "html",
                "Business Law",
                {"act_name": "The Example Act, 2026", "year": "2026", "reason": "test", "discovery_method": "test"},
                {"source_name": "India Code", "source_url": "https://www.indiacode.nic.in/example", "content_sha256": "abc"},
            )
        self.assertEqual("The Example Act, 2026", record["law"])
        self.assertEqual(4, record["section_count"])
        self.assertIn("what", record["sections"]["2"])

    def test_ordered_parser_ignores_toc_and_numbered_footnotes(self):
        text = """
ARRANGEMENT OF SECTIONS
1. Short title.
2. Definitions.
3. Main duty.

CHAPTER I
1. Short title.—This Act may be called the Example Act.
1. Ins. by Act 2 of 2025.
2. Definitions.—In this Act, company means an incorporated company.
3. Main duty.—Every company shall maintain the required records.
SCHEDULE I
1. Form A.
"""
        sections = parse_ordered_bare_act_sections(text)
        self.assertEqual(["1", "2", "3"], list(sections))
        self.assertIn("required records", sections["3"]["text"])


class DiscoveryFallbackTests(unittest.TestCase):
    def test_evidence_regex_does_not_keep_introductory_words(self):
        text = "Major laws include Indian Contract Act, 1872 and the Companies Act, 2013."
        names = LlamaProcessor._extract_act_names(text)
        self.assertIn("Indian Contract Act, 1872", names)

    def test_business_seeds_are_available_when_llama_is_offline(self):
        processor = LlamaProcessor(api_url="http://127.0.0.1:1", timeout=1)
        acts = processor.discover_indian_acts("Business Law", [])
        names = {item["act_name"] for item in acts}
        self.assertIn("The Companies Act, 2013", names)
        self.assertIn("The Indian Contract Act, 1872", names)


class DrishtiPipelineTests(unittest.TestCase):
    def test_specific_act_search_returns_only_exact_alias_match(self):
        catalog = [
            {"title": "Information Technology Act 2000", "category": "Criminal Law"},
            {"title": "Indian Penal Code 1860", "category": "Criminal Law"},
        ]
        selected = find_specific_catalog_act("IT Act", catalog)
        self.assertEqual("Information Technology Act 2000", selected["title"])

    def test_specific_act_search_accepts_unique_name_without_year(self):
        catalog = [
            {"title": "The Indian Penal Code, 1860", "category": "Criminal Law"},
            {"title": "The Indian Evidence Act, 1872", "category": "Criminal Law"},
        ]
        selected = find_specific_catalog_act("Indian Penal Code", catalog)
        self.assertEqual("The Indian Penal Code, 1860", selected["title"])

    def test_specific_act_search_rejects_ambiguous_name(self):
        catalog = [
            {"title": "Consumer Protection Act 1986", "category": "Civil Laws"},
            {"title": "Consumer Protection Act 2019", "category": "Civil Laws"},
        ]
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            find_specific_catalog_act("Consumer Protection Act", catalog)

    def test_specific_act_search_rejects_unknown_name(self):
        catalog = [
            {"title": "The Indian Penal Code, 1860", "category": "Criminal Law"},
        ]
        with self.assertRaisesRegex(ValueError, "No Drishti Bare Act matched"):
            find_specific_catalog_act("Completely Unknown Law", catalog)

    def test_onclick_pdf_url_is_extracted(self):
        value = "javascript:location.href='https://vault.drishtijudiciary.com/english_file_uploads/example.pdf'"
        self.assertEqual(
            "https://vault.drishtijudiciary.com/english_file_uploads/example.pdf",
            DrishtiBareActCrawler._onclick_url(value),
        )

    def test_only_drishti_vault_pdf_is_allowed(self):
        self.assertTrue(
            DrishtiPdfProcessor._is_allowed_pdf_url(
                "https://vault.drishtijudiciary.com/english_file_uploads/example.pdf"
            )
        )
        self.assertFalse(DrishtiPdfProcessor._is_allowed_pdf_url("https://example.com/example.pdf"))

    def test_catalog_deduplicates_same_act_identity(self):
        self.assertEqual(
            DrishtiBareActCrawler.canonical_act_key("The IT Act"),
            DrishtiBareActCrawler.canonical_act_key("IT Act"),
        )

    def test_llm_chunking_preserves_every_character(self):
        text = ("Section 1. Example text.\n" * 1000) + "THE END"
        chunks = LlamaProcessor._split_all_text(text, 4000)
        self.assertEqual(text, "".join(chunks))
        self.assertGreater(len(chunks), 1)

    def test_law_what_record_confirms_complete_llm_input(self):
        processor = LlamaProcessor(max_text_chars=9000)
        prompts = []

        def fake_request(prompt):
            prompts.append(prompt)
            if "FULL CHUNK TEXT:" in prompt:
                return {"law": "Example Act, 2026", "what": "Chunk rule", "key_points": ["Rule"], "section_references": ["1"]}
            return {"law": "Example Act, 2026", "what": "Complete law summary", "key_points": ["Rule"], "section_references": ["1"]}

        processor._request_json = fake_request
        text = ("Section text and obligations.\n" * 700) + "END"
        record = processor.generate_law_key_value_record(
            "Business Law",
            "Example Act, 2026",
            text,
            {"source_name": "Drishti Judiciary Bare Acts", "extraction": {"page_count": 4}},
            chunk_chars=4000,
        )
        self.assertEqual("Example Act, 2026", record["law"])
        self.assertEqual("Complete law summary", record["what"])
        self.assertTrue(record["llm_processing"]["every_extracted_character_sent"])
        self.assertEqual(len(text), record["llm_processing"]["processed_characters"])
        self.assertGreater(sum("FULL CHUNK TEXT:" in prompt for prompt in prompts), 1)

    def test_criminal_law_selection_never_includes_civil_category(self):
        processor = LlamaProcessor()
        catalog = [
            {"title": "Companies Act, 2013", "category": "Civil Laws"},
            {"title": "Indian Penal Code 1860", "category": "Criminal Law"},
            {"title": "Bharatiya Nyaya Sanhita 2023", "category": "New Criminal Laws"},
        ]
        selected = processor.select_relevant_catalog_acts("Criminal Law", catalog)
        self.assertEqual(
            {"Indian Penal Code 1860", "Bharatiya Nyaya Sanhita 2023"},
            {item["title"] for item in selected},
        )

    def test_section_index_outputs_section_number_and_what(self):
        processor = LlamaProcessor(max_text_chars=9000)

        def fake_request(prompt):
            if "INPUT SECTIONS:" in prompt:
                return {
                    "sections": [
                        {"section": "1", "what": "Section 1 gives the short title."},
                        {"section": "2", "what": "Section 2 defines company."},
                    ]
                }
            return {
                "law": "Example Act, 2026",
                "what": "The Act establishes the example framework.",
                "key_points": [],
                "section_references": ["1", "2"],
            }

        processor._request_json = fake_request
        sections = {
            "1": {"section": "1", "title": "Short title", "text": "This Act is the Example Act.", "chapter": "CHAPTER I"},
            "2": {"section": "2", "title": "Definitions", "text": "Company means an example company.", "chapter": "CHAPTER I"},
        }
        record = processor.generate_section_index_record(
            "Business Law", "Example Act, 2026", sections, {"extraction": {"page_count": 2}}
        )
        self.assertEqual(2, record["section_count"])
        self.assertEqual("Section 2 defines company.", record["sections"]["2"]["what"])
        self.assertTrue(record["llm_processing"]["every_parsed_section_indexed"])

    def test_invalid_json_response_is_retried(self):
        processor = LlamaProcessor(max_retries=2, retry_base_delay=0)
        processor.wait_until_ready = lambda **kwargs: True
        bad = Mock()
        bad.raise_for_status.return_value = None
        bad.json.return_value = {"response": "plain text, not JSON"}
        good = Mock()
        good.raise_for_status.return_value = None
        good.json.return_value = {"response": '{"what":"Recovered JSON"}'}
        with patch("backend.collector.llama_processor.requests.post", side_effect=[bad, good]) as mocked:
            result = processor._request_json("return JSON")
        self.assertEqual("Recovered JSON", result["what"])
        self.assertEqual(2, mocked.call_count)

    def test_large_act_uses_no_more_than_eight_sections_per_batch(self):
        processor = LlamaProcessor(
            max_text_chars=12000,
            large_act_threshold=160,
            large_act_batch_chars=7000,
            large_act_max_sections=8,
            large_act_batch_pause=0,
        )
        batch_sizes = []

        def fake_request(prompt):
            if "INPUT SECTIONS:" in prompt:
                payload = json.loads(prompt.split("INPUT SECTIONS:\n", 1)[1])
                batch_sizes.append(len(payload))
                return {
                    "sections": [
                        {"section": item["section"], "what": f"Meaning of {item['section']}"}
                        for item in payload
                    ]
                }
            return {
                "law": "Large Act, 2026",
                "what": "Overall meaning",
                "key_points": [],
                "section_references": [],
            }

        processor._request_json = fake_request
        sections = {
            str(number): {
                "section": str(number),
                "title": f"Section {number}",
                "text": "Short statutory text.",
                "chapter": "",
            }
            for number in range(1, 162)
        }
        record = processor.generate_section_index_record(
            "Criminal Law", "Large Act, 2026", sections, {"extraction": {}}
        )
        self.assertTrue(record["llm_processing"]["large_act_mode"])
        self.assertLessEqual(max(batch_sizes), 8)

    def test_completed_act_is_found_and_can_be_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            builder = IndianActDictionaryBuilder(Path(directory), "Criminal Law")
            builder.store(
                {
                    "record_type": "indian_bare_act_section_index",
                    "law": "Information Technology Act 2000",
                    "section_count": 1,
                    "sections": {"1": {"what": "Meaning"}},
                    "llm_processing": {"every_parsed_section_indexed": True},
                    "source": {"url": "https://vault.drishtijudiciary.com/it.pdf"},
                }
            )
            existing = builder.find_existing_complete("IT Act", "https://vault.drishtijudiciary.com/it.pdf")
            self.assertIsNotNone(existing)


if __name__ == "__main__":
    unittest.main()
