import base64
import importlib.util
import socketserver
import threading
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("serve_site", ROOT / "scripts" / "serve_site.py")
serve_site = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve_site)


def serve(state: Path, password: str):
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), serve_site.make_handler(state, password))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def fetch(port, path, auth=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}")
    if auth is not None:
        req.add_header("Authorization", "Basic " + base64.b64encode(auth.encode()).decode())
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""


def test_no_password_means_open(tmp_path):
    (tmp_path / "live.json").write_text('{"markets": {}}')
    srv, port = serve(tmp_path, "")
    try:
        assert fetch(port, "/live.json") == (200, b'{"markets": {}}')
    finally:
        srv.shutdown()


def test_password_is_required_and_any_user_name_works(tmp_path):
    (tmp_path / "live.json").write_text('{"markets": {}}')
    srv, port = serve(tmp_path, "s3cret")
    try:
        assert fetch(port, "/live.json")[0] == 401
        assert fetch(port, "/", auth="me:wrong")[0] == 401
        assert fetch(port, "/live.json", auth="anyone:s3cret")[0] == 200
        assert fetch(port, "/", auth=":s3cret")[0] == 200
        assert fetch(port, "/healthz")[0] == 200            # the platform health check needs no password
    finally:
        srv.shutdown()
