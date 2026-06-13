"""Unit tests for the OpenAI parser."""

from __future__ import annotations

import json

import httpx
import pytest

from khazad.adapters.parsers.openai import OpenAIParser

URL = "https://api.openai.com/v1/chat/completions"


@pytest.fixture
def parser() -> OpenAIParser:
    return OpenAIParser()


def make_request(payload: dict) -> httpx.Request:
    return httpx.Request("POST", URL, content=json.dumps(payload).encode())


class TestCanHandle:
    """Verify URL matching."""

    def test_matches_openai_chat(self, parser):
        assert parser.can_handle(httpx.URL(URL)) is True

    def test_matches_azure_chat(self, parser):
        url = httpx.URL(
            "https://my-resource.openai.azure.com/openai/deployments/gpt-4/chat/completions"
        )
        assert parser.can_handle(url) is True

    def test_matches_compatible_proxy(self, parser):
        url = httpx.URL("https://my-litellm-proxy.example.com/v1/chat/completions")
        assert parser.can_handle(url) is True

    def test_rejects_openai_other_path(self, parser):
        assert parser.can_handle(httpx.URL("https://api.openai.com/v1/embeddings")) is False

    def test_rejects_anthropic(self, parser):
        assert parser.can_handle(httpx.URL("https://api.anthropic.com/v1/messages")) is False


class TestParseRequest:
    """Verify prompt/model/stream extraction from various request shapes."""

    def test_single_user_message(self, parser):
        req = make_request({"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]})
        parsed = parser.parse_request(req)
        assert parsed.prompt == "user: Hello"
        assert parsed.model == "gpt-4"
        assert parsed.stream is False

    def test_full_conversation_in_prompt(self, parser):
        req = make_request(
            {
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "First question"},
                    {"role": "assistant", "content": "First answer"},
                    {"role": "user", "content": "Follow-up question"},
                ],
            }
        )
        parsed = parser.parse_request(req)
        assert parsed.prompt == (
            "system: Be concise.\n"
            "user: First question\n"
            "assistant: First answer\n"
            "user: Follow-up question"
        )

    def test_multimodal_content_parts_flattened(self, parser):
        req = make_request(
            {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe"},
                            {"type": "image_url", "image_url": {"url": "https://x/img.png"}},
                            {"type": "text", "text": "this image"},
                        ],
                    }
                ],
            }
        )
        assert parser.parse_request(req).prompt == "user: Describe this image"

    def test_raises_when_no_user_message(self, parser):
        req = make_request(
            {"model": "gpt-4", "messages": [{"role": "system", "content": "You are helpful."}]}
        )
        with pytest.raises(ValueError, match="No user message"):
            parser.parse_request(req)

    def test_stream_true(self, parser):
        req = make_request(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "x"}], "stream": True}
        )
        assert parser.parse_request(req).stream is True


class TestBuildResponse:
    """Verify response reconstruction."""

    def test_returns_200_json(self, parser, openai_chat_response):
        resp = parser.build_response(openai_chat_response)
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/json"
        assert resp.content == openai_chat_response

    def test_response_body_is_valid_json(self, parser, openai_chat_response):
        data = json.loads(parser.build_response(openai_chat_response).content)
        assert data["choices"][0]["message"]["content"] == "Paris is the capital of France."


class TestStreamChunks:
    """Verify SSE stream simulation."""

    def test_stream_yields_chunks(self, parser, openai_chat_response):
        chunks = list(parser.stream_chunks(openai_chat_response))
        assert len(chunks) >= 3  # role + at least 1 content + finish + DONE

    def test_stream_ends_with_done(self, parser, openai_chat_response):
        chunks = list(parser.stream_chunks(openai_chat_response))
        assert chunks[-1] == b"data: [DONE]\n\n"

    def test_chunks_are_valid_sse(self, parser, openai_chat_response):
        for chunk in parser.stream_chunks(openai_chat_response):
            decoded = chunk.decode()
            assert decoded.startswith("data: ")
            assert decoded.endswith("\n\n")

    def test_stream_contains_full_content(self, parser, openai_chat_response):
        full_text = ""
        for chunk in parser.stream_chunks(openai_chat_response):
            decoded = chunk.decode()
            if decoded.startswith("data: ") and decoded.strip() != "data: [DONE]":
                data = json.loads(decoded[6:])
                full_text += data.get("choices", [{}])[0].get("delta", {}).get("content", "")
        assert full_text == "Paris is the capital of France."

    def test_stream_has_finish_reason_stop(self, parser, openai_chat_response):
        last_data = None
        for chunk in parser.stream_chunks(openai_chat_response):
            decoded = chunk.decode().strip()
            if decoded != "data: [DONE]" and decoded.startswith("data: "):
                last_data = json.loads(decoded[6:])
        assert last_data is not None
        assert last_data["choices"][0]["finish_reason"] == "stop"


class TestResponseFromStream:
    """Verify reconstruction of a full response from captured SSE bytes."""

    def test_round_trip(self, parser, openai_chat_response):
        """stream_chunks → response_from_stream must preserve the content."""
        sse = b"".join(parser.stream_chunks(openai_chat_response))
        rebuilt = parser.response_from_stream(sse)
        assert rebuilt is not None
        data = json.loads(rebuilt)
        original = json.loads(openai_chat_response)
        assert (
            data["choices"][0]["message"]["content"]
            == (original["choices"][0]["message"]["content"])
        )
        assert data["model"] == original["model"]
        assert data["object"] == "chat.completion"

    def test_returns_none_for_empty_stream(self, parser):
        assert parser.response_from_stream(b"data: [DONE]\n\n") is None

    def test_collects_usage_when_present(self, parser):
        chunks = [
            b'data: {"object": "chat.completion.chunk", "id": "c1", "model": "gpt-4",'
            b' "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": null}]}\n\n',
            b'data: {"object": "chat.completion.chunk", "id": "c1", "model": "gpt-4",'
            b' "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],'
            b' "usage": {"total_tokens": 5}}\n\n',
            b"data: [DONE]\n\n",
        ]
        rebuilt = parser.response_from_stream(b"".join(chunks))
        data = json.loads(rebuilt)
        assert data["usage"] == {"total_tokens": 5}
        assert data["choices"][0]["message"]["content"] == "Hi"
