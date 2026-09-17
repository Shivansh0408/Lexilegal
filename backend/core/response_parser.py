"""Validation, merging, and frontend formatting for LLM judge output."""

from __future__ import annotations

import json
import logging
import re
from copy import deepcopy

LOGGER = logging.getLogger("lexbrief.response")

ARRAY_FIELDS = (
    "parties",
    "facts",
    "chronology",
    "evidence",
    "admitted_or_undisputed_facts",
    "disputed_facts",
    "credibility_and_reliability",
    "procedural_history",
    "questions_for_determination",
    "missing_information",
    "legal_context",
    "analysis_limitations",
)

LAWYER_ARRAY_FIELDS = (
    "strategy_facts",
    "objectives",
    "case_theories",
    "elements_and_issues",
    "strengths",
    "vulnerabilities",
    "contradictions",
    "evidence_assessment",
    "procedural_opportunities",
    "legal_arguments",
    "investigation_plan",
    "disclosure_requests",
    "witness_plan",
    "applications_and_motions",
    "negotiation_considerations",
    "hearing_trial_plan",
    "deadlines_and_preservation",
    "risk_register",
    "unresolved_questions",
    "missing_information",
    "next_actions",
    "legal_context",
    "ethics_and_safety",
    "analysis_limitations",
)

MATERIAL_TERMS = {
    "accused", "admitted", "alleged", "alleges", "assault", "bruise", "complainant",
    "complaint", "contusion", "denies", "disputed", "examined", "fir", "heard", "injury",
    "investigation", "medical", "message", "photograph", "police", "recorded", "reported",
    "seized", "sent", "slapped", "statement", "struck", "testimony", "threat", "threatened",
    "witness", "voice note", "pushed", "received", "observed", "married", "filed", "produced",
    "phone", "device", "cctv", "forensic", "preserved", "obtained", "investigating officer",
}
ACTION_TERMS = {
    "alleged", "said", "states", "reported", "recorded", "examined", "heard", "saw", "sent",
    "seized", "denied", "married", "threatened", "slapped", "struck", "pushed", "filed",
    "received", "observed", "produced", "collected", "visited", "identified", "noticed",
    "obtained", "preserved", "tested",
}
DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|"
    r"Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+\d{2,4}|(?:19|20)\d{2})\b",
    re.IGNORECASE,
)


def _json_object(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("The LLM response did not contain a JSON object.")
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("The LLM response must be a JSON object.")
    return value


def _deduplicate(values: list) -> list:
    seen: set[str] = set()
    result = []
    for value in values:
        key = json.dumps(value, sort_keys=True, ensure_ascii=False).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def parse_and_merge(raw_outputs: list[str]) -> dict:
    """Parse all chunk responses; arrays intentionally have no item cap."""

    LOGGER.info("Response step 1/3: validating %d LLM response(s)", len(raw_outputs))
    parsed = []
    for index, raw in enumerate(raw_outputs, start=1):
        try:
            parsed.append(_json_object(raw))
        except (ValueError, json.JSONDecodeError) as exc:
            LOGGER.warning("Ignoring invalid LLM JSON response %d: %s", index, exc)
    if not parsed:
        raise ValueError("None of the LLM responses contained valid JSON.")
    merged = {"case_title": None, "case_overview": ""}
    for field in ARRAY_FIELDS:
        merged[field] = []

    for item in parsed:
        if not merged["case_title"] and isinstance(item.get("case_title"), str):
            merged["case_title"] = item["case_title"].strip() or None
        overview = item.get("case_overview")
        if isinstance(overview, str) and overview.strip():
            if overview.strip() not in merged["case_overview"]:
                merged["case_overview"] = " ".join(
                    part for part in (merged["case_overview"], overview.strip()) if part
                )
        for field in ARRAY_FIELDS:
            values = item.get(field, [])
            if isinstance(values, list):
                merged[field].extend(value for value in values if value not in (None, "", {}))

    LOGGER.info("Response step 2/3: de-duplicating merged analysis")
    for field in ARRAY_FIELDS:
        merged[field] = _deduplicate(merged[field])
    return merged


def parse_and_merge_lawyer(raw_outputs: list[str]) -> dict:
    """Parse and de-duplicate every chunk of the lawyer strategy response."""

    LOGGER.info("Lawyer response step 1/3: validating %d response(s)", len(raw_outputs))
    parsed = []
    for index, raw in enumerate(raw_outputs, start=1):
        try:
            parsed.append(_json_object(raw))
        except (ValueError, json.JSONDecodeError) as exc:
            LOGGER.warning("Ignoring invalid lawyer JSON response %d: %s", index, exc)
    if not parsed:
        raise ValueError("None of the lawyer LLM responses contained valid JSON.")

    merged = {"case_title": None, "case_overview": "", "client_position": ""}
    for field in LAWYER_ARRAY_FIELDS:
        merged[field] = []
    for item in parsed:
        if not merged["case_title"] and isinstance(item.get("case_title"), str):
            merged["case_title"] = item["case_title"].strip() or None
        for scalar in ("case_overview", "client_position"):
            value = item.get(scalar)
            if isinstance(value, str) and value.strip() and value.strip() not in merged[scalar]:
                merged[scalar] = " ".join(part for part in (merged[scalar], value.strip()) if part)
        for field in LAWYER_ARRAY_FIELDS:
            values = item.get(field, [])
            if isinstance(values, list):
                merged[field].extend(value for value in values if value not in (None, "", {}))
    for field in LAWYER_ARRAY_FIELDS:
        merged[field] = _deduplicate(merged[field])
    return merged


def _document_pages(document_text: str) -> list[tuple[int | None, str]]:
    matches = list(re.finditer(r"\[Page\s+(\d+)\]\s*", document_text))
    if not matches:
        return [(None, document_text)]
    pages = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(document_text)
        pages.append((int(match.group(1)), document_text[match.end() : end]))
    return pages


def _status_for(text: str) -> str:
    lowered = text.lower()
    if any(term in lowered for term in (
        "denies", "disputes", "contested", "defence says", "defense says", "accused states", "accused says",
    )):
        return "disputed"
    if any(term in lowered for term in ("admitted fact", "undisputed", "both parties agree", "she accepts", "he accepts")):
        return "admitted"
    if any(term in lowered for term in ("medical", "records", "recorded", "seized", "metadata", "examined")):
        return "supported"
    if any(term in lowered for term in ("alleges", "alleged", "says", "states", "reported")) or re.search(
        r"\bi\s+(?:married|told|sent|felt|explained|understood|remained)", lowered
    ):
        return "alleged"
    return "unclear"


def _asserted_by(text: str) -> str | None:
    lowered = text.lower()
    if "pw-1" in lowered or "complainant" in lowered or "wife" in lowered:
        return "Complainant / PW-1"
    if "accused" in lowered or "defence" in lowered or "defense" in lowered:
        return "Accused / defence"
    if "pw-2" in lowered or "neighbour" in lowered or "neighbor" in lowered:
        return "Neighbour / PW-2"
    if "pw-3" in lowered or "sister" in lowered:
        return "Sister / PW-3"
    if "pw-4" in lowered or "doctor" in lowered or "medical officer" in lowered:
        return "Medical witness / PW-4"
    if "pw-5" in lowered or "investigating officer" in lowered or "police" in lowered:
        return "Investigating officer / PW-5"
    return None


def _support_for(text: str) -> str:
    lowered = text.lower()
    sources = []
    mappings = (
        (("medical", "mlc", "doctor", "hospital"), "medical record/testimony"),
        (("photograph", "photo"), "photographic material"),
        (("message", "chat", "voice note", "phone", "digital"), "electronic material"),
        (("witness", "pw-", "neighbour", "sister"), "witness account"),
        (("seiz", "hanger", "physical exhibit"), "physical/seizure record"),
        (("fir", "complaint", "police report"), "complaint/investigation record"),
    )
    for terms, label in mappings:
        if any(term in lowered for term in terms):
            sources.append(label)
    return ", ".join(sources) if sources else "statement extracted from the uploaded record"


def _significance_for(text: str) -> str:
    lowered = text.lower()
    if any(term in lowered for term in ("injury", "bruise", "contusion", "medical", "pain")):
        return "May bear on occurrence, bodily injury, timing, causation, and corroboration."
    if any(term in lowered for term in ("threat", "afraid", "alarm")):
        return "May bear on the alleged threat, intention, surrounding context, and resulting alarm."
    if any(term in lowered for term in ("phone", "message", "voice note", "photograph", "metadata", "digital")):
        return "May bear on contemporaneity, corroboration, authorship, completeness, and authenticity."
    if any(term in lowered for term in ("not seized", "not preserved", "no forensic", "did not obtain", "gap")):
        return "May affect evidentiary weight, completeness of investigation, and reasonable-doubt assessment."
    if any(term in lowered for term in ("denies", "disputes", "alternative", "separation")):
        return "Identifies a competing account or motive theory requiring comparison with the supporting record."
    return "May bear on the chronology, identity, conduct, or reliability of the competing accounts."


def _material_sentences(document_text: str) -> list[tuple[int | None, str]]:
    candidates: list[tuple[int | None, str]] = []
    seen: set[str] = set()
    excluded = (
        "simulated educational document", "not an authentic", "educational simulation",
        "not for filing", "citation caution", "research-source links", "prepared in english",
        "this compilation is designed", "this original hypothetical", "no answer to these questions",
        "no person named in it is real", "allegations are not facts merely", "guilt in a criminal trial",
        "not presented as a verbatim", "ordinary wear and tear", "arrest is not automatic",
        "prosecution submits", "defence submits", "defense submits", "requests that the admissible",
        "direct testimony does not require", "domestic assaults commonly occur", "record objections and rulings",
        "citation for understanding", "supreme court judgment",
    )
    for page_number, page_text in _document_pages(document_text):
        flattened = re.sub(r"\s+", " ", page_text).strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\s+[●•]\s+", flattened):
            sentence = re.sub(r"\s+", " ", sentence).strip(" -•\n")
            lowered = sentence.lower()
            if not 50 <= len(sentence) <= 900 or any(term in lowered for term in excluded):
                continue
            if re.match(r"^(?:\d+\s+)?whether\b", lowered):
                continue
            if any(term in lowered for term in ("suggested record controls", "trial preparation checklist")):
                continue
            if "statutory threshold" in lowered and not any(
                term in lowered for term in ("slapped", "struck", "pushed", "threatened", "injury")
            ):
                continue
            if any(term in lowered for term in ("i request lawful", "i am willing to provide")):
                continue
            material_hits = sum(1 for term in MATERIAL_TERMS if term in lowered)
            action_hit = any(term in lowered for term in ACTION_TERMS)
            dated = bool(DATE_PATTERN.search(sentence))
            if material_hits < 2 and not (dated and action_hit):
                continue
            signature = re.sub(r"[^a-z0-9]+", " ", lowered).strip()
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append((page_number, sentence))
    return candidates


def extractive_analysis(document_text: str, reason: str | None = None) -> dict:
    """Curate every material extractable fact when an LLM chunk is unavailable."""

    LOGGER.warning("Using multi-fact extractive recovery%s", f": {reason}" if reason else "")
    candidates = _material_sentences(document_text)
    if not candidates:
        body = re.sub(r"\[Page\s+\d+\]", " ", document_text)
        candidate = re.sub(r"\s+", " ", body).strip()[:900] or "Text was extracted from the uploaded court file."
        candidates = [(None, candidate)]

    facts = []
    chronology = []
    evidence = []
    disputed_facts = []
    admitted_facts = []
    missing_information = []
    evidence_seen: set[str] = set()
    missing_seen: set[str] = set()

    for page_number, sentence in candidates:
        status = _status_for(sentence)
        support = _support_for(sentence)
        confidence = "high" if status == "admitted" else "medium" if status == "supported" else "low" if status == "unclear" else "medium"
        fact = {
            "fact": sentence,
            "status": status,
            "asserted_by": _asserted_by(sentence),
            "support": support,
            "page_reference": f"Page {page_number}" if page_number else None,
            "legal_significance": _significance_for(sentence),
            "confidence": confidence,
        }
        facts.append(fact)

        date_match = DATE_PATTERN.search(sentence)
        if date_match:
            chronology.append({
                "date": date_match.group(0),
                "event": sentence,
                "source": f"Page {page_number}" if page_number else "Uploaded record",
            })
        if status == "admitted":
            admitted_facts.append(sentence)
        if status == "disputed":
            disputed_facts.append({
                "issue": sentence,
                "positions": "The uploaded record identifies competing positions; compare the original testimony and exhibits.",
                "supporting_material": support,
            })

        lowered = sentence.lower()
        evidence_type = None
        if any(term in lowered for term in ("medical", "mlc", "doctor", "hospital", "injury")):
            evidence_type = "medical"
        elif any(term in lowered for term in ("phone", "message", "voice note", "digital", "metadata")):
            evidence_type = "electronic"
        elif any(term in lowered for term in ("photograph", "document", "complaint", "fir", "statement")):
            evidence_type = "documentary"
        elif any(term in lowered for term in ("hanger", "physical exhibit", "seized")):
            evidence_type = "physical"
        elif any(term in lowered for term in ("witness", "pw-", "neighbour", "sister")):
            evidence_type = "testimonial"
        if evidence_type:
            evidence_key = f"{evidence_type}:{sentence.lower()}"
            if evidence_key not in evidence_seen:
                evidence_seen.add(evidence_key)
                evidence.append({
                    "item": sentence,
                    "type": evidence_type,
                    "offered_by": _asserted_by(sentence),
                    "proves_or_rebuts": _significance_for(sentence),
                    "reliability_note": "Verify this extracted description against the original exhibit and testimony.",
                    "source_reference": f"Page {page_number}" if page_number else None,
                })

        if any(term in lowered for term in (
            "not seized", "not preserved", "not obtained", "did not obtain", "no cctv", "no forensic",
            "cannot identify", "unavailable", "did not witness", "no eyewitness", "not forensically",
        )):
            key = sentence.lower()
            if key not in missing_seen:
                missing_seen.add(key)
                missing_information.append(sentence)

    overview_facts = [item["fact"] for item in facts[:3]]
    questions = []
    if disputed_facts:
        questions.append("Which competing factual account is supported by reliable and admissible evidence?")
    if any(item["type"] == "medical" for item in evidence):
        questions.append("What does the medical material reliably establish about injury, timing, and causation?")
    if any(item["type"] == "electronic" for item in evidence):
        questions.append("Are the electronic materials complete, authentic, attributable, and supported by the required foundation?")
    if missing_information:
        questions.append("What weight should be given to the identified evidentiary and investigation gaps?")

    return {
        "case_title": None,
        "case_overview": " ".join(overview_facts),
        "parties": [],
        "facts": facts,
        "chronology": chronology,
        "evidence": evidence,
        "admitted_or_undisputed_facts": admitted_facts,
        "disputed_facts": disputed_facts,
        "credibility_and_reliability": [
            "Extracted allegations and recorded observations must be checked against the original testimony and exhibits.",
            "Corroborative material may support timing or condition without independently proving identity or causation.",
        ],
        "procedural_history": [],
        "questions_for_determination": questions,
        "missing_information": missing_information,
        "legal_context": [],
        "analysis_limitations": ([
            "One or more LLM chunks could not be completed; those chunks were recovered with structured extractive analysis. "
            f"Technical detail: {reason}"
        ] if reason else []),
    }


def fallback_analysis(document_text: str, reason: str) -> dict:
    """Backward-compatible name for the full extractive recovery path."""

    return extractive_analysis(document_text, reason)


def extractive_lawyer_analysis(document_text: str, reason: str | None = None) -> dict:
    """Build a useful, conservative strategy brief when a lawyer LLM chunk fails."""

    base = extractive_analysis(document_text, None)
    strategy_facts = [
        {
            "fact": item.get("fact", ""),
            "status": item.get("status", "unclear"),
            "source": item.get("support"),
            "page_reference": item.get("page_reference"),
            "strategy_use": item.get("legal_significance", "Verify and assess this fact against the complete record."),
            "confidence": item.get("confidence", "low"),
        }
        for item in base["facts"]
    ]
    contradictions = [
        {
            "issue": item.get("issue", "Competing versions appear in the record."),
            "assertion_one": item.get("positions", "One account appears in the uploaded record."),
            "assertion_two": "The complete competing account must be confirmed from original statements and exhibits.",
            "sources": item.get("supporting_material", "Uploaded record"),
            "follow_up": "Create a source-by-source comparison without merging inconsistent versions.",
        }
        for item in base["disputed_facts"]
    ]
    evidence_assessment = [
        {
            "item": item.get("item", "Evidence item"),
            "helps": item.get("proves_or_rebuts", "May support a material factual proposition."),
            "hurts": "Its effect on the client position requires comparison with the complete record.",
            "authenticity_or_admissibility": item.get("reliability_note", "Verify authenticity and admissibility."),
            "next_step": "Inspect the original item, provenance, completeness, and any required certificate or foundation.",
            "source_reference": item.get("source_reference"),
        }
        for item in base["evidence"]
    ]
    return {
        "case_title": base.get("case_title"),
        "case_overview": base.get("case_overview", ""),
        "client_position": "The represented side is not reliably established by extractive recovery; counsel should confirm it.",
        "strategy_facts": strategy_facts,
        "objectives": ["Confirm the client, represented side, desired outcome, and immediate procedural posture."],
        "case_theories": [],
        "elements_and_issues": [],
        "strengths": [],
        "vulnerabilities": [],
        "contradictions": contradictions,
        "evidence_assessment": evidence_assessment,
        "procedural_opportunities": [],
        "legal_arguments": [],
        "investigation_plan": [
            {"priority": "high", "action": "Build a page-cited chronology from original records.", "purpose": "Preserve timing conflicts and identify gaps.", "source_basis": "Uploaded case file."},
            {"priority": "high", "action": "Obtain and inspect complete originals or certified copies of material exhibits.", "purpose": "Verify provenance, completeness, and accuracy.", "source_basis": "Uploaded case file and identified evidence."},
        ],
        "disclosure_requests": [
            {"material": item, "reason": "The uploaded record identifies this as absent, incomplete, or not preserved.", "source_basis": "Missing-information extraction."}
            for item in base["missing_information"]
        ],
        "witness_plan": [],
        "applications_and_motions": [],
        "negotiation_considerations": [],
        "hearing_trial_plan": [],
        "deadlines_and_preservation": [{"item": "Potentially relevant evidence", "date": "unknown", "action": "Issue lawful preservation steps promptly after counsel review.", "basis": "Avoid loss or alteration while the complete record is obtained."}],
        "risk_register": [],
        "unresolved_questions": base["questions_for_determination"],
        "missing_information": base["missing_information"],
        "next_actions": [
            {"priority": "urgent", "action": "Confirm all live deadlines, custody or hearing dates, and preservation needs.", "why": "The extractive fallback cannot reliably infer current deadlines."},
            {"priority": "high", "action": "Verify every extracted fact against its original page and exhibit.", "why": "Strategy should not rely on an automated paraphrase alone."},
        ],
        "legal_context": [],
        "ethics_and_safety": [
            "Preserve evidence and metadata without alteration.",
            "Use lawful witness-contact procedures and do not coach or influence testimony.",
            "Protect confidential and privileged material and verify current law from authoritative sources.",
        ],
        "analysis_limitations": ([
            "One or more LLM chunks could not be completed; those chunks received conservative extractive recovery. "
            f"Technical detail: {reason}"
        ] if reason else []),
    }


def format_for_frontend(analysis: dict, document: dict, matched_laws: list[str]) -> dict:
    """Final validation boundary before app.py sends data to the browser."""

    LOGGER.info("Response step 3/3: checking the frontend response contract")
    result = deepcopy(analysis)
    facts = result.get("facts")
    if not isinstance(facts, list) or not facts:
        raise ValueError("Analysis must contain at least one fact.")
    for field in ARRAY_FIELDS:
        if not isinstance(result.get(field), list):
            result[field] = []
    # Section identifiers are intentionally not exposed. The dictionary still
    # grounds retrieval and the LLM prompt, while the UI receives only Act names.
    result["matched_laws"] = sorted(set(matched_laws))
    result["document"] = document
    result["fact_count"] = len(result["facts"])
    result["disclaimer"] = (
        "Decision-support output only. Verify every fact against the original record and applicable current law."
    )
    return result


def format_lawyer_for_frontend(analysis: dict, document: dict, matched_laws: list[str]) -> dict:
    """Validate and finalize the lawyer response sent to the browser."""

    LOGGER.info("Lawyer response step 3/3: checking the frontend contract")
    result = deepcopy(analysis)
    facts = result.get("strategy_facts")
    if not isinstance(facts, list) or not facts:
        raise ValueError("Lawyer analysis must contain at least one strategy fact.")
    for field in LAWYER_ARRAY_FIELDS:
        if not isinstance(result.get(field), list):
            result[field] = []
    result["matched_laws"] = sorted(set(matched_laws))
    result["document"] = document
    result["fact_count"] = len(result["strategy_facts"])
    result["disclaimer"] = (
        "Lawyer work-product support only, not a legal opinion. Counsel must verify the record, current law, "
        "deadlines, professional duties, and strategic decisions."
    )
    return result
