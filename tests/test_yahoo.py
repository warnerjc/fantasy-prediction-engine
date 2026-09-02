"""Yahoo client tests (no network): token persistence, refresh, and get()."""

import json
import time

import pytest

from data import yahoo


@pytest.fixture(autouse=True)
def yahoo_env(tmp_path, monkeypatch):
    """Point the client at a throwaway token file and dummy credentials."""
    monkeypatch.setenv("YAHOO_CLIENT_ID", "cid")
    monkeypatch.setenv("YAHOO_CLIENT_SECRET", "secret")
    monkeypatch.setenv("YAHOO_REDIRECT_URI", "https://localhost:8000/callback")
    monkeypatch.setenv("YAHOO_TOKEN_PATH", str(tmp_path / "yahoo_token.json"))
    return tmp_path


class _Resp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self.ok = status < 400
        self._payload = payload or {}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload


def test_token_round_trip():
    tok = {"access_token": "a", "refresh_token": "r", "expires_at": 123.0}
    yahoo._save_token(tok)
    assert yahoo._load_token() == tok


def test_access_token_refreshes_when_expired(monkeypatch):
    yahoo._save_token({"access_token": "old", "refresh_token": "r0",
                       "expires_at": time.time() - 10})

    calls = []

    def fake_post_token(data):
        calls.append(data)
        return {"access_token": "new", "expires_in": 3600}  # no refresh_token echoed

    monkeypatch.setattr(yahoo, "_post_token", fake_post_token)

    assert yahoo._access_token() == "new"
    assert calls[0]["grant_type"] == "refresh_token"
    # a refresh response without its own refresh_token keeps the existing one
    assert yahoo._load_token()["refresh_token"] == "r0"


def test_get_appends_format_and_bearer(monkeypatch):
    yahoo._save_token({"access_token": "tok123", "refresh_token": "r",
                       "expires_at": time.time() + 3600})
    seen = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        seen["url"], seen["params"], seen["headers"] = url, params, headers
        return _Resp(200, {"ok": True})

    monkeypatch.setattr(yahoo.requests, "get", fake_get)

    assert yahoo.get("league/nfl.l.236625/settings") == {"ok": True}
    assert seen["params"]["format"] == "json"
    assert seen["headers"]["Authorization"] == "Bearer tok123"


def test_get_retries_once_on_401(monkeypatch):
    yahoo._save_token({"access_token": "stale", "refresh_token": "r",
                       "expires_at": time.time() + 3600})
    responses = [_Resp(401, {"error": "expired"}), _Resp(200, {"ok": 1})]
    monkeypatch.setattr(yahoo.requests, "get",
                        lambda *a, **k: responses.pop(0))
    monkeypatch.setattr(yahoo, "_post_token",
                        lambda data: {"access_token": "fresh", "expires_in": 3600})

    assert yahoo.get("users;use_login=1/games") == {"ok": 1}
    assert responses == []  # both consumed


def test_find_all_pulls_nested_dicts():
    payload = {"a": [{"league_key": "1", "name": "x"},
                     {"z": {"league_key": "2", "name": "y"}}]}
    keys = sorted(d["league_key"] for d in yahoo._find_all(payload, "league_key"))
    assert keys == ["1", "2"]
