"""HTTP transport interceptor — patches httpx to route LLM traffic through the cache."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from khazad.khazad import Khazad, PreparedRequest

logger = logging.getLogger("khazad")

# ---------------------------------------------------------------------------
# Patch install / uninstall
# ---------------------------------------------------------------------------

_original_async_init = None
_original_sync_init = None


def install(cache: Khazad) -> None:
    """Monkey-patch httpx.Client and httpx.AsyncClient to use Khazad transports.

    Safe to call repeatedly: only the *first* install captures the pristine
    ``__init__`` references. Subsequent calls swap in a new cache without
    losing the original — so ``uninstall()`` always restores real httpx.
    """
    global _original_async_init, _original_sync_init

    if _original_async_init is None:
        _original_async_init = httpx.AsyncClient.__init__
    if _original_sync_init is None:
        _original_sync_init = httpx.Client.__init__

    original_async = _original_async_init
    original_sync = _original_sync_init

    def patched_async_init(self: httpx.AsyncClient, *args, **kwargs) -> None:
        original_async(self, *args, **kwargs)
        self._transport = CachedAsyncTransport(cache, self._transport)

    def patched_sync_init(self: httpx.Client, *args, **kwargs) -> None:
        original_sync(self, *args, **kwargs)
        self._transport = CachedSyncTransport(cache, self._transport)

    httpx.AsyncClient.__init__ = patched_async_init  # type: ignore[method-assign]
    httpx.Client.__init__ = patched_sync_init  # type: ignore[method-assign]

    logger.info("[Khazad] HTTP transport patches installed")


def uninstall() -> None:
    """Restore the original httpx transports."""
    global _original_async_init, _original_sync_init

    if _original_async_init is not None:
        httpx.AsyncClient.__init__ = _original_async_init  # type: ignore[method-assign]
        _original_async_init = None

    if _original_sync_init is not None:
        httpx.Client.__init__ = _original_sync_init  # type: ignore[method-assign]
        _original_sync_init = None

    logger.info("[Khazad] HTTP transport patches removed")


# ---------------------------------------------------------------------------
# Transport wrappers
# ---------------------------------------------------------------------------


class CachedSyncTransport(httpx.BaseTransport):
    """Sync httpx transport that intercepts LLM requests for caching."""

    def __init__(self, cache: Khazad, wrapped: httpx.BaseTransport) -> None:
        self._cache = cache
        self._wrapped = wrapped

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        cache = self._cache
        prepared = cache.prepare(request) if cache.is_active() else None
        if prepared is None:
            return self._wrapped.handle_request(request)

        hit = cache.lookup(prepared)
        if hit is not None:
            return _replay(prepared, hit)

        response = self._wrapped.handle_request(request)
        if response.status_code != 200:
            return response

        if _is_sse(response):
            if not _can_capture(response):
                return response
            stream = _SyncTeeStream(
                response.stream, lambda raw: _store_stream(cache, prepared, raw)
            )
            return _swap_stream(response, stream)

        try:
            response.read()
            cache.store(prepared, response.content)
        except Exception:
            logger.warning("[Khazad] Failed to store response in cache", exc_info=True)
        return response

    def close(self) -> None:
        self._wrapped.close()


class CachedAsyncTransport(httpx.AsyncBaseTransport):
    """Async httpx transport that intercepts LLM requests for caching."""

    def __init__(self, cache: Khazad, wrapped: httpx.AsyncBaseTransport) -> None:
        self._cache = cache
        self._wrapped = wrapped

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        cache = self._cache
        prepared = cache.prepare(request) if cache.is_active() else None
        if prepared is None:
            return await self._wrapped.handle_async_request(request)

        # Embedding and Redis search are blocking — keep the event loop free.
        loop = asyncio.get_running_loop()
        hit = await loop.run_in_executor(None, cache.lookup, prepared)
        if hit is not None:
            return _replay(prepared, hit)

        response = await self._wrapped.handle_async_request(request)
        if response.status_code != 200:
            return response

        if _is_sse(response):
            if not _can_capture(response):
                return response
            stream = _AsyncTeeStream(
                response.stream, lambda raw: _store_stream(cache, prepared, raw)
            )
            return _swap_stream(response, stream)

        try:
            await response.aread()
            await loop.run_in_executor(None, cache.store, prepared, response.content)
        except Exception:
            logger.warning("[Khazad] Failed to store response in cache", exc_info=True)
        return response

    async def aclose(self) -> None:
        await self._wrapped.aclose()


# ---------------------------------------------------------------------------
# Cache hit / miss plumbing
# ---------------------------------------------------------------------------


def _replay(prepared: PreparedRequest, hit) -> httpx.Response:
    """Build the response for a cache hit — JSON body or simulated SSE stream."""
    if prepared.stream:
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            stream=_ReplayStream(prepared.parser.stream_chunks(hit.response_data)),
        )
    return prepared.parser.build_response(hit.response_data)


def _store_stream(cache: Khazad, prepared: PreparedRequest, raw: bytes) -> None:
    """Reconstruct a canonical JSON response from raw SSE bytes and cache it."""
    body = prepared.parser.response_from_stream(raw)
    if body:
        cache.store(prepared, body)


def _is_sse(response: httpx.Response) -> bool:
    return "text/event-stream" in response.headers.get("content-type", "")


def _can_capture(response: httpx.Response) -> bool:
    """Raw stream bytes are only parseable when the body is not compressed."""
    return response.headers.get("content-encoding", "identity").lower() in ("", "identity")


def _swap_stream(
    response: httpx.Response, stream: httpx.SyncByteStream | httpx.AsyncByteStream
) -> httpx.Response:
    """Clone a response, replacing its stream with a tee'd one."""
    return httpx.Response(
        status_code=response.status_code,
        headers=response.headers,
        stream=stream,
        extensions=response.extensions,
    )


# ---------------------------------------------------------------------------
# Stream helpers
# ---------------------------------------------------------------------------


class _ReplayStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    """Serve synthesized SSE chunks to either a sync or an async client."""

    def __init__(self, chunks: Iterator[bytes]) -> None:
        self._chunks = chunks

    def __iter__(self):
        yield from self._chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk
            await asyncio.sleep(0)  # let other tasks run between frames


class _SyncTeeStream(httpx.SyncByteStream):
    """Pass chunks through untouched; cache the body once fully consumed."""

    def __init__(self, inner: httpx.SyncByteStream, on_complete: Callable[[bytes], None]) -> None:
        self._inner = inner
        self._on_complete = on_complete

    def __iter__(self):
        parts: list[bytes] = []
        for chunk in self._inner:
            parts.append(chunk)
            yield chunk
        # Natural exhaustion only — an aborted stream is never cached.
        try:
            self._on_complete(b"".join(parts))
        except Exception:
            logger.warning("[Khazad] Failed to store streamed response", exc_info=True)

    def close(self) -> None:
        self._inner.close()


class _AsyncTeeStream(httpx.AsyncByteStream):
    """Async variant of :class:`_SyncTeeStream`; stores off the event loop."""

    def __init__(self, inner: httpx.AsyncByteStream, on_complete: Callable[[bytes], None]) -> None:
        self._inner = inner
        self._on_complete = on_complete

    async def __aiter__(self):
        parts: list[bytes] = []
        async for chunk in self._inner:
            parts.append(chunk)
            yield chunk
        body = b"".join(parts)
        asyncio.get_running_loop().run_in_executor(None, self._safe_complete, body)

    def _safe_complete(self, body: bytes) -> None:
        try:
            self._on_complete(body)
        except Exception:
            logger.warning("[Khazad] Failed to store streamed response", exc_info=True)

    async def aclose(self) -> None:
        await self._inner.aclose()
