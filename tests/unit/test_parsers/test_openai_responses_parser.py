"""Unit tests for the OpenAI Responses API parser."""

from __future__ import annotations

import json

import httpx
import pytest

from khazad.adapters.parsers.openai_responses import OpenAIResponsesParser

URL = "https://api.openai.com/v1/responses"


@pytest.fixture
def parser() -> OpenAIResponsesParser:
    return OpenAIResponsesParser()


def make_request(payload: dict) -> httpx.Request:
    return httpx.Request("POST", URL, content=json.dumps(payload).encode())


class TestCanHandle:
    def test_matches_responses_endpoint(self, parser):
        assert parser.can_handle(httpx.URL(URL)) is True

    def test_matches_compatible_proxy(self, parser):
        assert parser.can_handle(httpx.URL("https://my-proxy.example.com/v1/responses")) is True

    def test_matches_azure_responses(self, parser):
        url = httpx.URL("https://my-resource.openai.azure.com/v1/responses")
        assert parser.can_handle(url) is True

    def test_rejects_chat_completions(self, parser):
        assert parser.can_handle(httpx.URL("https://api.openai.com/v1/chat/completions")) is False

    def test_rejects_other_path(self, parser):
        assert parser.can_handle(httpx.URL("https://api.openai.com/v1/embeddings")) is False


class TestParseRequest:
    def test_string_input(self, parser):
        req = make_request({"model": "gpt-4o", "input": "What is the capital of France?"})
        parsed = parser.parse_request(req)
        assert parsed.prompt == "user: What is the capital of France?"
        assert parsed.model == "gpt-4o"
        assert parsed.stream is False

    def test_instructions_included(self, parser):
        req = make_request({"model": "gpt-4o", "instructions": "Be brief.", "input": "Hello"})
        assert parser.parse_request(req).prompt == "instructions: Be brief.\nuser: Hello"

    def test_list_input_full_conversation(self, parser):
        req = make_request(
            {
                "model": "gpt-4o",
                "input": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "First question"},
                    {"role": "assistant", "content": "First answer"},
                    {"role": "user", "content": "Follow-up question"},
                ],
            }
        )
        assert parser.parse_request(req).prompt == (
            "system: Be concise.\n"
            "user: First question\n"
            "assistant: First answer\n"
            "user: Follow-up question"
        )

    def test_list_input_with_content_parts(self, parser):
        req = make_request(
            {
                "model": "gpt-4o",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Hello"},
                            {"type": "input_text", "text": "world"},
                        ],
                    }
                ],
            }
        )
        assert parser.parse_request(req).prompt == "user: Hello world"

    def test_raises_when_no_input_field(self, parser):
        with pytest.raises(ValueError, match="No 'input' field"):
            parser.parse_request(make_request({"model": "gpt-4o"}))

    def test_raises_when_no_user_message_in_list(self, parser):
        req = make_request(
            {"model": "gpt-4o", "input": [{"role": "system", "content": "You are helpful."}]}
        )
        with pytest.raises(ValueError, match="No user message"):
            parser.parse_request(req)

    def test_stream_true(self, parser, openai_responses_streaming_body):
        req = httpx.Request("POST", URL, content=openai_responses_streaming_body)
        assert parser.parse_request(req).stream is True


class TestBuildResponse:
    def test_returns_200_json(self, parser, openai_responses_response):
        resp = parser.build_response(openai_responses_response)
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/json"
        assert resp.content == openai_responses_response

    def test_response_body_is_valid_json(self, parser, openai_responses_response):
        data = json.loads(parser.build_response(openai_responses_response).content)
        assert data["object"] == "response"
        assert data["output"][0]["content"][0]["text"] == "Paris is the capital of France."


class TestStreamChunks:
    def test_yields_bytes(self, parser, openai_responses_response):
        chunks = list(parser.stream_chunks(openai_responses_response))
        assert len(chunks) > 0
        assert all(isinstance(c, bytes) for c in chunks)

    def test_event_sequence(self, parser, openai_responses_response):
        raw = b"".join(parser.stream_chunks(openai_responses_response))
        assert b"response.created" in raw
        assert b"response.output_text.delta" in raw
        assert b"response.completed" in raw

    def test_delta_chunks_carry_text(self, parser, openai_responses_response):
        """All delta payloads together must reconstruct the original assistant text."""
        full_text = ""
        for chunk in parser.stream_chunks(openai_responses_response):
            for line in chunk.decode().splitlines():
                if line.startswith("data:"):
                    payload = json.loads(line[len("data:") :].strip())
                    if payload.get("type") == "response.output_text.delta":
                        full_text += payload.get("delta", "")
        assert full_text == "Paris is the capital of France."

    def test_each_chunk_is_valid_sse_frame(self, parser, openai_responses_response):
        for chunk in parser.stream_chunks(openai_responses_response):
            assert chunk.startswith(b"event:"), f"Unexpected chunk: {chunk!r}"


class TestResponseFromStream:
    def test_round_trip(self, parser, openai_responses_response):
        sse = b"".join(parser.stream_chunks(openai_responses_response))
        rebuilt = parser.response_from_stream(sse)
        assert rebuilt is not None
        data = json.loads(rebuilt)
        original = json.loads(openai_responses_response)
        assert data["status"] == "completed"
        assert data["model"] == original["model"]
        assert (
            data["output"][0]["content"][0]["text"]
            == (original["output"][0]["content"][0]["text"])
        )

    def test_returns_none_without_completed_event(self, parser):
        sse = b'event: response.created\ndata: {"type": "response.created"}\n\n'
        assert parser.response_from_stream(sse) is None
