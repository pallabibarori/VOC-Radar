"""Small HTTP helper: one session, polite user agent, retries, timeouts."""
from __future__ import annotations

import time

import requests

USER_AGENT = "voc-tool/1.0 (customer-feedback research; +https://github.com/)"
_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json, text/html;q=0.9"})


def get(url: str, *, params: dict | None = None, headers: dict | None = None,
        timeout: float = 15, retries: int = 2) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = _session.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 429 and attempt < retries:   # rate limited: back off
                time.sleep(2 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(1 + attempt)
    raise last_exc  # type: ignore[misc]


def get_json(url: str, **kw):
    return get(url, **kw).json()


def post(url: str, **kw) -> requests.Response:
    kw.setdefault("timeout", 120)
    resp = _session.post(url, **kw)
    resp.raise_for_status()
    return resp
