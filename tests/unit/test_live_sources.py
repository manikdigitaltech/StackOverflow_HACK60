"""Behavior contracts for the resilience guarantee of live_sources clients.

Targets Phase 5. These are the most important tests in the whole subsystem
per the design constraint: a live-source failure must never propagate as an
exception, only as an empty list plus a logged warning.
"""
import requests

from core.config.settings import settings
from core.rag.live_sources.arxiv_client import search_arxiv
from core.rag.live_sources.semantic_scholar_client import search_semantic_scholar


def test_search_semantic_scholar_returns_empty_list_on_timeout(monkeypatch):
    """A requests.Timeout during the API call must be swallowed, returning []."""
    def boom(*args, **kwargs):
        raise requests.Timeout("simulated timeout")
    monkeypatch.setattr(requests, "get", boom)
    assert search_semantic_scholar("any query") == []


def test_search_semantic_scholar_returns_empty_list_on_http_error(monkeypatch):
    """A non-2xx response must be swallowed, returning [], not raised."""
    class FakeResponse:
        def raise_for_status(self):
            raise requests.HTTPError("429 Too Many Requests")
    monkeypatch.setattr(requests, "get", lambda *a, **kw: FakeResponse())
    assert search_semantic_scholar("any query") == []


def test_search_arxiv_returns_empty_list_on_network_error(monkeypatch):
    """Any network-layer exception from the arxiv client must be swallowed, returning []."""
    def boom(*args, **kwargs):
        raise requests.ConnectionError("simulated network failure")
    monkeypatch.setattr(requests, "get", boom)
    assert search_arxiv("any query") == []


_ATOM_3_ENTRIES = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><id>http://arxiv.org/abs/1</id><title>Paper One</title><summary>s1</summary><published>2024-01-01</published></entry>
  <entry><id>http://arxiv.org/abs/2</id><title>Paper Two</title><summary>s2</summary><published>2024-01-02</published></entry>
  <entry><id>http://arxiv.org/abs/3</id><title>Paper Three</title><summary>s3</summary><published>2024-01-03</published></entry>
</feed>"""


def test_search_semantic_scholar_sends_api_key_header_when_configured(monkeypatch):
    """If LIVE_SOURCES__SEMANTIC_SCHOLAR_API_KEY is set, it must be sent as
    x-api-key -- omitted entirely (not sent as empty string) when unset."""
    monkeypatch.setattr(settings.live_sources, "semantic_scholar_api_key", "test-key-123")
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass
        def json(self):
            return {"data": []}

    def fake_get(*args, **kwargs):
        captured["headers"] = kwargs.get("headers")
        return FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)
    search_semantic_scholar("any query")
    assert captured["headers"] == {"x-api-key": "test-key-123"}


def test_search_semantic_scholar_omits_header_when_no_key(monkeypatch):
    monkeypatch.setattr(settings.live_sources, "semantic_scholar_api_key", None)
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass
        def json(self):
            return {"data": []}

    def fake_get(*args, **kwargs):
        captured["headers"] = kwargs.get("headers")
        return FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)
    search_semantic_scholar("any query")
    assert captured["headers"] == {}


def test_search_arxiv_respects_k(monkeypatch):
    """search_arxiv never returns more than k results."""
    class FakeResponse:
        text = _ATOM_3_ENTRIES
        def raise_for_status(self):
            pass
    monkeypatch.setattr(requests, "get", lambda *a, **kw: FakeResponse())
    hits = search_arxiv("any query", k=2)
    assert len(hits) == 2
    assert all(h.source == "arxiv" for h in hits)
