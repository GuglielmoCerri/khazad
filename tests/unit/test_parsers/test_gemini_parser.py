"""Unit tests for the Gemini parser."""

from __future__ import annotations

import json

import httpx
import pytest

from khazad.adapters.parsers.gemini import GeminiParser

URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"


@pytest.fixture
def parser() -> GeminiParser:
    return GeminiParser()


def make_request(payload: dict, url: str = URL) -> httpx.Request:
    return httpx.Request("POST", url, content=json.dumps(payload).encode())


class TestCanHandle:
    """Verify URL matching for Gemini."""

    def test_matches_gemini_generate(self, parser):
        assert parser.can_handle(httpx.URL(URL)) is True

    def test_rejects_gemini_other_path(self, parser):
        url = httpx.URL(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:listModels"
        )
        assert parser.can_handle(url) is False

    def test_rejects_stream_generate(self, parser):
        """Gemini streaming uses a separate endpoint that passes through uncached."""
        url = httpx.URL(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:streamGenerateContent"
        )
        assert parser.can_handle(url) is False

    def test_rejects_different_host(self, parser):
        assert parser.can_handle(httpx.URL("https://api.openai.com/v1/chat/completions")) is False


class TestParseRequest:
    """Verify prompt extraction from Gemini request bodies."""

    def test_single_user_part(self, parser):
        req = make_request(
            {"contents": [{"role": "user", "parts": [{"text": "Explain quantum computing"}]}]}
        )
        parsed = parser.parse_request(req)
        assert parsed.prompt == "user: Explain quantum computing"
        assert parsed.model == "gemini-pro"
        assert parsed.stream is False

    def test_multiple_text_parts(self, parser):
        req = make_request(
            {"contents": [{"role": "user", "parts": [{"text": "Part one"}, {"text": "Part two"}]}]}
        )
        assert parser.parse_request(req).prompt == "user: Part one Part two"

    def test_full_conversation_in_prompt(self, parser):
        req = make_request(
            {
                "contents": [
                    {"role": "user", "parts": [{"text": "First"}]},
                    {"role": "model", "parts": [{"text": "Reply"}]},
                    {"role": "user", "parts": [{"text": "Second"}]},
                ]
            }
        )
        assert parser.parse_request(req).prompt == "user: First\nmodel: Reply\nuser: Second"

    def test_system_instruction_included(self, parser):
        req = make_request(
            {
                "systemInstruction": {"parts": [{"text": "Be brief."}]},
                "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
            }
        )
        assert parser.parse_request(req).prompt == "system: Be brief.\nuser: Hi"

    def test_raises_when_no_user_content(self, parser):
        req = make_request({"contents": [{"role": "model", "parts": [{"text": "I am a model."}]}]})
        with pytest.raises(ValueError, match="No user content"):
            parser.parse_request(req)

    def test_model_extracted_from_url(self, parser):
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        req = make_request({"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]}, url=url)
        assert parser.parse_request(req).model == "gemini-2.0-flash"
