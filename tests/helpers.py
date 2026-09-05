"""Shared fixtures: a fake upstream socket, a relay on a free port, and a
reset of the forwarder's module-level state between tests.

The forwarder keeps its state in module globals (_upstream, _stats, the port
settings), so tests must run one at a time and reset that state in setUp.
unittest runs sequentially, which is all this relies on."""
from __future__ import annotations

import socket
import threading
import time
import unittest
from unittest import mock

from omp_forwarder import forwarder as fwd

HOST = "127.0.0.1"


def free_port() -> int:
    """A port nothing listens on. Used wherever a test needs a dead upstream."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_for(pred, timeout: float = 5.0, what: str = "condition") -> None:
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out after {timeout}s waiting for {what}")


def http_response(status: int = 200, body: bytes = b"",
                  ctype: str = "text/plain",
                  extra: tuple[str, ...] = ()) -> bytes:
    reason = {200: "OK", 404: "Not Found", 500: "Internal Server Error",
              503: "Service Unavailable"}.get(status, "Whatever")
    lines = [f"HTTP/1.1 {status} {reason}", f"Content-Type: {ctype}",
             f"Content-Length: {len(body)}", "Connection: close", *extra,
             "", ""]
    return "\r\n".join(lines).encode() + body


def read_http_message(sock: socket.socket) -> bytes:
    """Read one HTTP request: the head, then Content-Length bytes of body.
    Returns whatever arrived if the peer closes or goes quiet first."""
    buf = b""
    try:
        while b"\r\n\r\n" not in buf:
            chunk = sock.recv(65536)
            if not chunk:
                return buf
            buf += chunk
        head, _, body = buf.partition(b"\r\n\r\n")
        length = 0
        for line in head.split(b"\r\n")[1:]:
            k, _, v = line.partition(b":")
            if k.strip().lower() == b"content-length":
                length = int(v.strip())
        while len(body) < length:
            chunk = sock.recv(65536)
            if not chunk:
                break
            body += chunk
        return head + b"\r\n\r\n" + body
    except OSError:
        return buf


def reset_state() -> None:
    fwd._upstream = None
    fwd.FORCED_UPSTREAM = None
    fwd.EXCLUDE_PORTS = set()
    # No test should sit in the reload wait; the ones that care set it.
    fwd.WAIT_FOR_MODEL = 0
    fwd.STUDIO_FALLBACK = False
    fwd.UPSTREAM_EXE = None
    fwd._upstream_exe = None
    fwd.CANDIDATE_PORTS = []
    fwd._container_port = None
    fwd._upstream_kind = None
    fwd.PREFER = "llama-server"
    fwd._port_owner = {}
    # A dead port, so a test that falls back cannot reach a real Studio.
    fwd.STUDIO_PORT = free_port()
    # clear+update rather than rebind: the module holds the dict by identity.
    fwd._stats.clear()
    fwd._stats.update({"conns": 0, "requests": 0, "2xx": 0, "4xx": 0,
                       "5xx": 0, "fallbacks": 0, "unavailable": 0,
                       "latency": [], "model": "",
                       "tok_prompt": 0, "tok_cached": 0, "tok_gen": 0,
                       "started": time.time()})
    fwd._tok_last = None
    fwd._tok_dirty.clear()
    fwd.TOKENS_FILE = None          # no test writes a day file by accident
    fwd._day_totals = {}
    fwd._days_dirty = False
    fwd._last_day = None
    # Container-upstream mode state
    fwd.WSL_DISTRO = None
    fwd.CONTAINER_NAME = None
    fwd._container_status = None
    fwd._container_keepalive = None
    fwd._container_last_start = 0.0
    # Upstream health + deployment facts, refreshed by the sampler thread.
    fwd._upstream_healthy = False
    fwd._upstream_facts = {}
    # The /__control auth token, generated in main().
    fwd._control_token = ""
    # Lane identity: --name and --peer flags.
    fwd.FWD_NAME = None
    fwd.PEERS = []
    fwd.FWD_GPU = None
    # Per-peer and GPU sampler state.
    fwd._peer_state = {}
    fwd._gpu_state = []
    # Unload control: the start command and the operator's stop latch.
    fwd.UPSTREAM_CMD = None
    fwd._operator_stopped = False
    fwd._upstream_child = None


class FakeUpstream:
    """A TCP server that speaks just enough HTTP to stand in for llama-server.

    It records the raw bytes of every request it receives, answers through a
    responder callable, then closes the connection. The responder returns a
    list of items: bytes are sent, numbers are slept, so a test can script a
    response that trickles out like an SSE stream.

    The default responder answers /health with 200 and everything else with
    404, which is enough for discover() to accept it."""

    def __init__(self, responder=None, healthy: bool = True):
        self.responder = responder or self._default
        self.healthy = healthy
        self.received: list[bytes] = []
        self.paths: list[str] = []
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((HOST, 0))
        self._srv.listen(16)
        self.port: int = self._srv.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()

    def _default(self, method: str, path: str, raw: bytes) -> list:
        if path == "/health":
            if self.healthy:
                return [http_response(200, b'{"status":"ok"}',
                                      "application/json")]
            return [http_response(503, b'{"status":"loading"}',
                                  "application/json")]
        # Deployment-fact endpoints. SGLang answers /get_server_info;
        # llama-server answers /props. Both 404 by default so that
        # upstream_facts() reads "unknown" for the engine.
        if path == "/get_server_info":
            return [http_response(200, b'{"default_chat_template_kwargs":{"enable_thinking":false},"speculative_algorithm":"","tp_size":1,"pp_size":1,"model_path":"/models/test"}',
                                  "application/json")]
        if path == "/props":
            return [http_response(404)]
        return [http_response(404)]

    def _accept(self) -> None:
        while True:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,),
                             daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        conn.settimeout(5)
        raw = read_http_message(conn)
        self.received.append(raw)
        try:
            method, path = (p.decode("latin-1")
                            for p in raw.split(b" ", 2)[:2])
        except ValueError:
            method, path = "", ""
        self.paths.append(path)
        try:
            for item in self.responder(method, path, raw):
                if isinstance(item, (int, float)):
                    time.sleep(item)
                else:
                    conn.sendall(item)
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def close(self) -> None:
        self._srv.close()


def raw_request(port: int, parts: list, timeout: float = 10.0) -> bytes:
    """Send `parts` to the relay (bytes are sent, numbers are slept between
    sends), then read until EOF. A reset from the peer counts as EOF."""
    with socket.create_connection((HOST, port), timeout=timeout) as c:
        for part in parts:
            if isinstance(part, (int, float)):
                time.sleep(part)
            else:
                c.sendall(part)
        out = b""
        try:
            while True:
                data = c.recv(65536)
                if not data:
                    break
                out += data
        except OSError:
            pass
    return out


class ForwarderCase(unittest.TestCase):
    """Base for tests that touch forwarder state: resets it, silences log()."""

    def setUp(self):
        self._log = mock.patch.object(fwd, "log", lambda msg: None)
        self._log.start()
        self.addCleanup(self._log.stop)
        reset_state()
        self.addCleanup(reset_state)

    def fake(self, *args, **kwargs) -> FakeUpstream:
        f = FakeUpstream(*args, **kwargs)
        self.addCleanup(f.close)
        return f


class RelayCase(ForwarderCase):
    """Base for tests that drive the relay: starts it on a free port."""

    def setUp(self):
        super().setUp()
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((HOST, 0))
        self.srv.listen(16)
        self.port = self.srv.getsockname()[1]
        fwd.LISTEN_PORT = self.port
        fwd.EXCLUDE_PORTS = {self.port, fwd.STUDIO_PORT}
        self.thread = threading.Thread(target=fwd._serve_forever,
                                       args=(self.srv,), daemon=True)
        self.thread.start()

    def tearDown(self):
        # shutdown() first: on Linux close() alone does not wake a blocked
        # accept(), so _serve_forever would never see the closed socket.
        # Windows raises ENOTCONN on shutdown of a listener, hence the guard.
        try:
            self.srv.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.srv.close()
        self.thread.join(2)
        self.assertFalse(self.thread.is_alive(),
                         "_serve_forever did not stop on a closed listener")
        super().tearDown()

    def wait_done(self, n: int) -> None:
        """Block until `n` relayed requests have fully completed."""
        wait_for(lambda: len(fwd._stats["latency"]) >= n,
                 what=f"{n} completed relays")
