"""Unit tests for the Anthropic parser."""

from __future__ import annotations

import json

import httpx
import pytest

from khazad.adapters.parsers.anthropic import AnthropicParser

URL = "https://api.anthropic.com/v1/messages"


@pytest.fixture
def parser() -> AnthropicParser:
    return AnthropicParser()


def make_request(payload: dict) -> httpx.Request:
    return httpx.Request("POST", URL, content=json.dumps(payload).encode())


class TestCanHandle:
    """Verify URL matching for Anthropic."""

    def test_matches_anthropic_messages(self, parser):
        assert parser.can_handle(httpx.URL(URL)) is True

    def test_rejects_anthropic_other_path(self, parser):
        assert parser.can_handle(httpx.URL("https://api.anthropic.com/v1/completions")) is False

    def test_rejects_openai(self, parser):
        assert parser.can_handle(httpx.URL("https://api.openai.com/v1/chat/completions")) is False


class TestParseRequest:
    """Verify prompt extraction from Anthropic request bodies."""

    def test_simple_string_content(self, parser):
        req = make_request(
            {
                "model": "claude-3-sonnet",
                "messages": [{"role": "user", "content": "Hello Claude"}],
                "max_tokens": 1024,
            }
        )
        parsed = parser.parse_request(req)
        assert parsed.prompt == "user: Hello Claude"
        assert parsed.model == "claude-3-sonnet"

    def test_structured_content_blocks(self, parser):
        req = make_request(
            {
                "model": "claude-3-sonnet",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "First part"},
                            {"type": "text", "text": "Second part"},
                        ],
                    }
                ],
                "max_tokens": 1024,
            }
        )
        assert parser.parse_request(req).prompt == "user: First part Second part"

    def test_system_field_included(self, parser):
        req = make_request(
            {
                "model": "claude-3-sonnet",
                "system": "Be terse.",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 1024,
            }
        )
        assert parser.parse_request(req).prompt == "system: Be terse.\nuser: Hi"

    def test_full_conversation_in_prompt(self, parser):
        req = make_request(
            {
                "model": "claude-3-sonnet",
                "messages": [
                    {"role": "user", "content": "First question"},
                    {"role": "assistant", "content": "Answer"},
                    {"role": "user", "content": "Follow-up"},
                ],
                "max_tokens": 1024,
            }
        )
        assert parser.parse_request(req).prompt == (
            "user: First question\nassistant: Answer\nuser: Follow-up"
        )

    def test_raises_when_no_user_message(self, parser):
        req = make_request(
            {
                "model": "claude-3-sonnet",
                "messages": [{"role": "assistant", "content": "I am here to help."}],
                "max_tokens": 1024,
            }
        )
        with pytest.raises(ValueError, match="No user message"):
            parser.parse_request(req)

    def test_stream_flag(self, parser):
        req = make_request(
            {
                "model": "claude-3-sonnet",
                "messages": [{"role": "user", "content": "x"}],
                "stream": True,
            }
        )
        assert parser.parse_request(req).stream is True


class TestStreamChunks:
    """Verify SSE stream simulation for Anthropic format."""

    def test_stream_yields_required_events(self, parser, anthropic_chat_response):
        chunks = [c.decode() for c in parser.stream_chunks(anthropic_chat_response)]
        event_types = [c.split("event: ")[1].split("\n")[0] for c in chunks if "event: " in c]
        assert "message_start" in event_types
        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types
        assert "content_block_stop" in event_types
        assert "message_delta" in event_types
        assert "message_stop" in event_types

    def test_stream_contains_text(self, parser, anthropic_chat_response):
        full_text = ""
        for chunk in parser.stream_chunks(anthropic_chat_response):
            decoded = chunk.decode()
            if "content_block_delta" in decoded:
                data = json.loads(decoded.split("data: ")[1].strip())
                full_text += data.get("delta", {}).get("text", "")
        assert full_text == "Paris is the capital of France."


class TestResponseFromStream:
    """Verify reconstruction of a full message from captured SSE bytes."""

    def test_round_trip(self, parser, anthropic_chat_response):
        sse = b"".join(parser.stream_chunks(anthropic_chat_response))
        rebuilt = parser.response_from_stream(sse)
        assert rebuilt is not None
        data = json.loads(rebuilt)
        original = json.loads(anthropic_chat_response)
        assert data["content"][0]["text"] == original["content"][0]["text"]
        assert data["model"] == original["model"]
        assert data["stop_reason"] == original["stop_reason"]
        assert data["role"] == "assistant"

    def test_returns_none_without_message_start(self, parser):
        assert parser.response_from_stream(b"event: ping\ndata: {}\n\n") is None
