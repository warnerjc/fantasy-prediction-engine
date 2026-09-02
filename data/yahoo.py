"""Thin Yahoo Fantasy Sports API client (read-only) + one-time OAuth2 login.

Credentials come from `.env` (see `.env.example` / `config.py`):
``YAHOO_CLIENT_ID``, ``YAHOO_CLIENT_SECRET``, ``YAHOO_REDIRECT_URI``.

Usage:

    bin/yahoo-auth login              # authorize once; caches tokens under data/cache/
    bin/yahoo-auth whoami             # prove the token works
    bin/yahoo-auth leagues            # list your NFL leagues + their league_keys
    bin/yahoo-auth settings <key>     # dump a league's raw settings JSON
    bin/yahoo-auth raw <path> [k=v]   # GET any fantasy/v2 path, print the JSON

Yahoo access tokens last ~1h; ``get()`` refreshes transparently via the stored
refresh token, so ``login`` is a one-time step until the refresh token is revoked.

The settings JSON this dumps is not yet wired into ``scoring.normalize_yahoo`` —
its shape differs from the hand-captured ``league-configs/yahoo-236625-scoring.json``.
Mapping one to the other is the next step; for now this just proves the pipe.
"""

from __future__ import annotations

import argparse
import json
import secrets
import ssl
import time
import urllib.parse
import webbrowser

import requests

import config

AUTH_URL = "https://api.login.yahoo.com/oauth2/request_auth"
TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"
BASE = "https://fantasysports.yahooapis.com/fantasy/v2"
_TIMEOUT = 30


def _cfg() -> dict:
    return config.yahoo_config()


def _creds() -> tuple[str, str]:
    c = _cfg()
    cid = c["client_id"] or config.get("YAHOO_CLIENT_ID", required=True)
    secret = c["client_secret"] or config.get("YAHOO_CLIENT_SECRET", required=True)
    return cid, secret


# --- token storage -----------------------------------------------------------

def _load_token() -> dict | None:
    p = _cfg()["token_path"]
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _save_token(tok: dict) -> None:
    p = _cfg()["token_path"]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(tok, indent=2))
    try:
        p.chmod(0o600)
    except OSError:
        pass


def _token_from_response(d: dict, *, fallback_refresh: str | None = None) -> dict:
    return {
        "access_token": d["access_token"],
        "refresh_token": d.get("refresh_token") or fallback_refresh,
        "token_type": d.get("token_type", "bearer"),
        "guid": d.get("xoauth_yahoo_guid"),
        "expires_at": time.time() + int(d.get("expires_in", 3600)),
    }


# --- OAuth2 ------------------------------------------------------------------

def _post_token(data: dict) -> dict:
    cid, secret = _creds()
    r = requests.post(TOKEN_URL, data={"redirect_uri": _cfg()["redirect_uri"], **data},
                      auth=(cid, secret), headers={"Accept": "application/json"},
                      timeout=_TIMEOUT)
    if not r.ok:
        raise RuntimeError(f"Yahoo token endpoint {r.status_code}: {r.text}")
    return r.json()


def _exchange_code(code: str) -> dict:
    tok = _token_from_response(_post_token({"grant_type": "authorization_code", "code": code}))
    _save_token(tok)
    return tok


def _refresh(tok: dict) -> dict:
    if not tok.get("refresh_token"):
        raise RuntimeError("stored token has no refresh_token; run `bin/yahoo-auth login`")
    new = _token_from_response(
        _post_token({"grant_type": "refresh_token", "refresh_token": tok["refresh_token"]}),
        fallback_refresh=tok["refresh_token"],
    )
    _save_token(new)
    return new


def _access_token(*, force_refresh: bool = False) -> str:
    tok = _load_token()
    if tok is None:
        raise RuntimeError("no Yahoo token cached; run `bin/yahoo-auth login` first")
    if force_refresh or tok.get("expires_at", 0) - time.time() < 60:
        tok = _refresh(tok)
    return tok["access_token"]


def _auth_url(state: str) -> str:
    c = _cfg()
    cid, _ = _creds()
    params = {
        "client_id": cid,
        "redirect_uri": c["redirect_uri"],
        "response_type": "code",
        "state": state,
        "language": "en-us",
    }
    # Yahoo Fantasy access is granted by the app's API Permissions, not a `scope`
    # string — sending scope=fspt-r here returns `invalid_scope`. Only pass one if
    # explicitly set (e.g. `openid` for the OIDC flow).
    if c.get("scope"):
        params["scope"] = c["scope"]
    return f"{AUTH_URL}?{urllib.parse.urlencode(params)}"


def _self_signed_cert():
    """(certfile, keyfile) temp paths, or None if `cryptography` isn't installed."""
    try:
        import datetime
        import tempfile

        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        return None

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
        .sign(key, hashes.SHA256())
    )
    cf = tempfile.NamedTemporaryFile(delete=False, suffix="-cert.pem")
    kf = tempfile.NamedTemporaryFile(delete=False, suffix="-key.pem")
    cf.write(cert.public_bytes(serialization.Encoding.PEM))
    kf.write(key.private_bytes(serialization.Encoding.PEM,
                               serialization.PrivateFormat.TraditionalOpenSSL,
                               serialization.NoEncryption()))
    cf.close()
    kf.close()
    return cf.name, kf.name


def _capture_code_via_server(redirect_uri: str, expected_state: str) -> str | None:
    """Serve the redirect URI once and return the ``code`` query param. Returns
    None if a local listener can't be stood up (caller falls back to paste)."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    parsed = urllib.parse.urlparse(redirect_uri)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    deadline = time.time() + 300
    got: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            if q.get("code") or q.get("error"):
                got["code"] = (q.get("code") or [None])[0]
                got["state"] = (q.get("state") or [None])[0]
                got["error"] = (q.get("error") or [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            msg = "Authorized. Close this tab and return to the terminal." if got.get("code") \
                else "Waiting for the authorization redirect..."
            self.wfile.write(f"<h3>{msg}</h3>".encode())

        def log_message(self, *_):  # silence
            pass

    # Bind all interfaces (":: " / 0.0.0.0) so Windows->WSL2 localhost forwarding reaches it.
    try:
        httpd = HTTPServer(("", port), Handler)
    except OSError as e:
        print(f"  (local listener can't bind port {port}: {e})")
        return None
    httpd.timeout = 2

    if parsed.scheme == "https":
        pair = _self_signed_cert()
        if pair is None:
            httpd.server_close()
            print("  (https redirect needs `pip install cryptography` for the listener)")
            return None
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(pair[0], pair[1])
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)

    print(f"  waiting for the Yahoo redirect on {redirect_uri}")
    print("  (self-signed cert — the browser will warn; choose Advanced -> proceed)")
    try:
        while not got and time.time() < deadline:
            try:
                httpd.handle_request()  # returns after httpd.timeout if nothing arrived
            except (ssl.SSLError, OSError, ValueError):
                continue  # cert-warning handshake aborts land here; keep listening
    finally:
        httpd.server_close()

    if not got:
        print("  (no redirect received within 5 min)")
        return None
    if got.get("error"):
        raise RuntimeError(f"Yahoo denied authorization: {got['error']}")
    if expected_state and got.get("state") and got["state"] != expected_state:
        raise RuntimeError("OAuth state mismatch — aborting (possible CSRF)")
    return got.get("code")


def _code_from_pasted(text: str) -> str:
    text = text.strip()
    if "error=" in text or "code=" in text:
        q = urllib.parse.parse_qs(urllib.parse.urlparse(text).query)
        if q.get("error"):
            raise RuntimeError(
                f"Yahoo redirected with an error: {q['error'][0]} "
                f"({q.get('error_description', ['?'])[0]}). Nothing to exchange."
            )
        return q["code"][0]
    return text


def authorize(*, manual: bool = False, open_browser: bool = True) -> dict:
    """Run the 3-legged OAuth2 flow and cache the tokens. Returns the token dict."""
    redirect_uri = _cfg()["redirect_uri"]
    state = secrets.token_urlsafe(16)
    url = _auth_url(state)
    code: str | None = None

    if not manual:
        print(f"Opening browser to authorize:\n  {url}\n")
        if open_browser:
            webbrowser.open(url)
        try:
            code = _capture_code_via_server(redirect_uri, state)
        except RuntimeError:
            raise
        except Exception as e:  # noqa: BLE001 - any listener failure -> paste path
            print(f"  local listener failed ({e}); use the paste flow below")

    if not code:
        print(
            "Open this URL, approve access, then copy the `code` value from the "
            f"URL you're redirected to (it will start with {redirect_uri}):\n\n  {url}\n"
        )
        code = _code_from_pasted(input("Paste code (or the full redirected URL): "))

    tok = _exchange_code(code)
    print(f"Authorized. Token cached at {_cfg()['token_path']}")
    return tok


# --- API -------------------------------------------------------------------

def get(path: str, **params) -> dict:
    """GET ``{BASE}/{path}`` as JSON, with transparent token refresh on 401."""
    params.setdefault("format", "json")
    url = f"{BASE}/{path.lstrip('/')}"
    for attempt in (1, 2):
        token = _access_token(force_refresh=(attempt == 2))
        r = requests.get(url, params=params,
                         headers={"Authorization": f"Bearer {token}"}, timeout=_TIMEOUT)
        if r.status_code == 401 and "additional_authorization_required" in r.text:
            raise RuntimeError(
                "Yahoo: additional_authorization_required — this account/app is not "
                "granted Fantasy Sports API access. Access is approval-gated now: apply "
                "at https://sports.yahoo.com/developer/access/ and wait for Yahoo's "
                "approval email. Once approved, re-authorize: `rm data/cache/"
                "yahoo_token.json && bin/yahoo-auth login --manual`. If an approved "
                "token still 401s, try YAHOO_SCOPE=fspt-r in .env and re-auth."
            )
        if r.status_code == 401 and attempt == 1:
            continue
        if not r.ok:
            raise RuntimeError(f"Yahoo {r.status_code} for {path}: {r.text[:400]}")
        return r.json()
    raise RuntimeError(f"Yahoo kept returning 401 for {path}")


def _find_all(obj, key: str) -> list[dict]:
    """Every dict anywhere in Yahoo's deeply-nested JSON that contains ``key``."""
    out: list[dict] = []
    if isinstance(obj, dict):
        if key in obj:
            out.append(obj)
        for v in obj.values():
            out += _find_all(v, key)
    elif isinstance(obj, list):
        for v in obj:
            out += _find_all(v, key)
    return out


def league_key(league_id: str | None = None, code: str = "nfl") -> str:
    lid = league_id or _cfg()["league_id"]
    if ".l." in str(lid):
        return str(lid)
    return f"{code}.l.{lid}"


def my_leagues(code: str = "nfl") -> list[dict]:
    payload = get(f"users;use_login=1/games;game_keys={code}/leagues")
    seen, leagues = set(), []
    for d in _find_all(payload, "league_key"):
        if d["league_key"] not in seen:
            seen.add(d["league_key"])
            leagues.append(d)
    return leagues


def league_settings(key: str) -> dict:
    return get(f"league/{key}/settings")


# --- CLI -------------------------------------------------------------------

def _cmd_whoami(_args) -> None:
    tok = _load_token()
    if not tok:
        raise SystemExit("no token cached; run `bin/yahoo-auth login` first")
    payload = get("users;use_login=1/games;game_keys=nfl")
    games = _find_all(payload, "game_key")
    print(f"guid: {tok.get('guid')}")
    print(f"token expires in {int(tok.get('expires_at', 0) - time.time())}s "
          f"(auto-refreshes)")
    print(f"NFL games visible: {[g.get('game_key') for g in games]}")


def _cmd_leagues(_args) -> None:
    rows = my_leagues()
    if not rows:
        print("no NFL leagues found for this account")
        return
    for lg in rows:
        print(f"{lg.get('league_key'):<18} {lg.get('name', '?'):<28} "
              f"teams={lg.get('num_teams', '?'):<3} scoring={lg.get('scoring_type', '?')}")


def _cmd_settings(args) -> None:
    print(json.dumps(league_settings(league_key(args.league_key)), indent=2))


def _cmd_raw(args) -> None:
    params = dict(kv.split("=", 1) for kv in args.params)
    print(json.dumps(get(args.path, **params), indent=2))


def _cmd_login(args) -> None:
    authorize(manual=args.manual, open_browser=not args.no_browser)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("login", help="one-time OAuth2 authorization")
    p.add_argument("--manual", action="store_true",
                   help="skip the local listener; paste the code by hand")
    p.add_argument("--no-browser", action="store_true", help="don't auto-open the browser")
    p.set_defaults(func=_cmd_login)

    sub.add_parser("whoami", help="print guid + prove the token works").set_defaults(
        func=_cmd_whoami)
    sub.add_parser("leagues", help="list this account's NFL leagues").set_defaults(
        func=_cmd_leagues)

    p = sub.add_parser("settings", help="dump a league's settings JSON")
    p.add_argument("league_key", nargs="?", default=None,
                   help="full league_key or bare id (default: YAHOO_LEAGUE_ID)")
    p.set_defaults(func=_cmd_settings)

    p = sub.add_parser("raw", help="GET any fantasy/v2 path")
    p.add_argument("path")
    p.add_argument("params", nargs="*", help="extra query params as k=v")
    p.set_defaults(func=_cmd_raw)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
