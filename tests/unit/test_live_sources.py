"""Behavior contracts for the resilience guarantee of live_sources clients.

Targets Phase 5. These are the most important tests in the whole subsystem
per the design constraint: a live-source failure must never propagate as an
exception, only as an empty list plus a logged warning.
"""


def test_search_semantic_scholar_returns_empty_list_on_timeout(monkeypatch):
    """A requests.Timeout during the API call must be swallowed, returning []."""
    raise NotImplementedError


def test_search_semantic_scholar_returns_empty_list_on_http_error(monkeypatch):
    """A non-2xx response must be swallowed, returning [], not raised."""
    raise NotImplementedError


def test_search_arxiv_returns_empty_list_on_network_error(monkeypatch):
    """Any network-layer exception from the arxiv client must be swallowed, returning []."""
    raise NotImplementedError


def test_search_arxiv_respects_k():
    """search_arxiv never returns more than k results."""
    raise NotImplementedError
