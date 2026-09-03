"""A fixed local port that forwards to whatever llama-server Unsloth Studio
is currently running.

THE PROBLEM. Unsloth Studio serves an OpenAI-compatible API on :8888, but that
endpoint is a Python proxy in front of the real `llama-server` process. The
proxy costs measurable time on every request, and it re-streams every token.
Measured on a dual-GPU Windows box, same model and prompt:

    short request, median of 10   0.302 s via :8888    0.136 s direct
    400-token generation          109 tok/s via :8888  127 tok/s direct

An agent loop makes many small calls, so both the per-call latency and the
per-token cost compound.

WHY YOU CANNOT JUST POINT AT llama-server. Studio starts it on a random free
port and re-rolls that port on every model reload -- 54966, then 60008, then
55084 in a single morning. No client can hold a direct endpoint.

WHAT THIS DOES. It listens on one fixed port, finds Studio's current
llama-server, and relays raw TCP. Raw TCP rather than HTTP, so SSE streaming,
keep-alive and chunked bodies pass through untouched. Re-discovery is lazy: it
happens when a connect fails, so a model reload costs one failed connection
instead of a config edit.

If no llama-server is running it falls back to Studio's own proxy, which makes
that one request pay the overhead but triggers Studio's load-on-demand. Every
later request then goes direct.

It also serves its own dashboard, because a client that bypasses :8888 becomes
invisible to Studio's API panel. See stats.py.

It binds loopback only, and there is deliberately no option to change that. A
forwarder that removes an API key requirement should not be reachable off the
machine.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request

HOST = "127.0.0.1"

#: Set from the command line by main(). Module-level because the tray and the
#: stats page read live state from this module.
LISTEN_PORT = 8890
STUDIO_PORT = 8888
FORCED_UPSTREAM: int | None = None
EXCLUDE_PORTS: set[int] = set()

_lock = threading.Lock()
_upstream: int | None = None

#: Counters only the forwarder can produce. llama-server's /metrics has no
#: error counter, so without these a 500 from the chat template is completely
#: silent -- which is exactly how a real template bug went unnoticed.
_stats: dict = {"conns": 0, "requests": 0, "2xx": 0, "4xx": 0, "5xx": 0,
                "fallbacks": 0, "latency": [], "model": "",
                "started": time.time()}


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def _server_pids() -> set[str]:
    """PIDs of running llama-server processes.

    Windows only, and intentionally cheap: this can run on a failed-connect
    path, so it uses tasklist rather than anything heavier. On other platforms,
    pin the port with --upstream-port instead."""
    if os.name != "nt":
        return set()
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq llama-server.exe",
             "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=20).stdout
    except Exception as exc:
        log(f"tasklist failed: {exc!r}")
        return set()
    return set(re.findall(r'"llama-server\.exe","(\d+)"', out))


def _listening_ports(pids: set[str]) -> list[int]:
    if not pids:
        return []
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "TCP"],
                             capture_output=True, text=True,
                             timeout=30).stdout
    except Exception as exc:
        log(f"netstat failed: {exc!r}")
        return []
    ports = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[-2] != "LISTENING":
            continue
        if parts[-1] not in pids:
            continue
        m = re.search(r":(\d+)$", parts[1])
        if m:
            p = int(m.group(1))
            if p not in EXCLUDE_PORTS:
                ports.append(p)
    # Highest port first: Studio's random pick is always high, so a stray
    # low-numbered listener is more likely to be something else.
    return sorted(set(ports), reverse=True)


def _healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(
                f"http://{HOST}:{port}/health", timeout=4) as r:
            return r.status == 200
    except Exception:
        return False


def discover(force: bool = False) -> int | None:
    """The current llama-server port, cached. force=True re-scans."""
    global _upstream
    with _lock:
        if FORCED_UPSTREAM:
            _upstream = FORCED_UPSTREAM
            return _upstream
        if _upstream and not force and _healthy(_upstream):
            return _upstream
        for port in _listening_ports(_server_pids()):
            if _healthy(port):
                if port != _upstream:
                    log(f"upstream -> {HOST}:{port}")
                    _stats["model"] = ""      # re-read it for the new server
                _upstream = port
                return port
        if _upstream is not None:
            log("no healthy llama-server found")
        _upstream = None
        return None


def _note_status(chunk: bytes) -> None:
    """Count an HTTP status code seen at the head of an upstream read.

    Deliberately naive: it looks only at reads that BEGIN with a status line,
    rather than tracking HTTP framing. A relay that parsed framing could break
    streaming, and on loopback a status line practically never splits across
    reads. So this can undercount; it will not miscount."""
    if not chunk.startswith(b"HTTP/1."):
        return
    try:
        code = int(chunk.split(b" ", 2)[1])
    except (IndexError, ValueError):
        return
    if code < 400:
        bucket = "2xx"
    elif code < 500:
        bucket = "4xx"
    else:
        bucket = "5xx"
    _stats[bucket] = _stats.get(bucket, 0) + 1


def _pump(src: socket.socket, dst: socket.socket,
          watch_status: bool = False) -> None:
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            if watch_status:
                _note_status(data)
            dst.sendall(data)
    except OSError:
        pass
    finally:
        # Half-close so the peer sees EOF instead of hanging on a streamed body.
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def _read_request_head(client: socket.socket) -> bytes:
    """Bytes up to the end of the request line. Anything read past it is kept
    and forwarded verbatim, so the body is never disturbed."""
    head = b""
    client.settimeout(20)
    try:
        while b"\r\n" not in head and len(head) < 16384:
            chunk = client.recv(4096)
            if not chunk:
                break
            head += chunk
    except OSError:
        return b""
    finally:
        try:
            client.settimeout(None)
        except OSError:
            pass
    return head


def _serve_local(client: socket.socket, path: str) -> None:
    """Serve the dashboard (/__stats) and its data (/__stats.json)."""
    from . import stats as stats_mod
    # sys.modules[__name__] rather than importing this module by name: under
    # some entry points that would build a SECOND module object with its own
    # _upstream, which then reads as None while the real one is serving.
    me = sys.modules[__name__]
    if path.startswith("/__stats.json"):
        if not _stats.get("model"):
            _stats["model"] = stats_mod.upstream_model(_upstream)
        body = json.dumps(stats_mod.snapshot(me, _stats)).encode()
        ctype = "application/json"
    else:
        body = stats_mod.PAGE.encode("utf-8")
        ctype = "text/html; charset=utf-8"
    lines = [
        "HTTP/1.1 200 OK",
        f"Content-Type: {ctype}",
        f"Content-Length: {len(body)}",
        "Cache-Control: no-store",
        "Connection: close",
        "", "",
    ]
    try:
        client.sendall("\r\n".join(lines).encode() + body)
    except OSError:
        pass


def handle(client: socket.socket) -> None:
    _stats["conns"] += 1
    head = _read_request_head(client)
    if not head:
        client.close()
        return
    try:
        target = head.split(b" ", 2)[1].decode("latin-1")
    except IndexError:
        target = ""
    if target.startswith("/__stats"):
        _serve_local(client, target)
        client.close()
        return

    _stats["requests"] += 1
    started = time.time()
    port = discover()
    if port is None:
        port = discover(force=True)
    if port is None:
        log(f"no llama-server; falling back to Studio :{STUDIO_PORT}")
        port = STUDIO_PORT
    try:
        up = socket.create_connection((HOST, port), timeout=15)
    except OSError:
        # Studio almost certainly re-rolled the port. Re-scan once and retry.
        port = discover(force=True)
        if port is None:
            client.close()
            return
        try:
            up = socket.create_connection((HOST, port), timeout=15)
        except OSError as exc:
            log(f"upstream connect failed: {exc!r}")
            client.close()
            return
    if port == STUDIO_PORT:
        _stats["fallbacks"] += 1
    for sk in (client, up):
        sk.settimeout(None)
        sk.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        up.sendall(head)                  # the bytes we peeked to route
    except OSError:
        client.close()
        up.close()
        return
    t = threading.Thread(target=_pump, args=(up, client, True), daemon=True)
    t.start()
    _pump(client, up)
    t.join()
    lat = _stats["latency"]
    lat.append(time.time() - started)
    del lat[:-200]
    for sk in (client, up):
        try:
            sk.close()
        except OSError:
            pass


def _serve_forever(srv: socket.socket) -> None:
    while True:
        try:
            client, _ = srv.accept()
        except KeyboardInterrupt:
            log("stopping")
            return
        except OSError as exc:
            log(f"accept failed: {exc!r}")
            continue
        threading.Thread(target=handle, args=(client,), daemon=True).start()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="omp-forwarder",
        description="A fixed local port that forwards to Unsloth Studio's "
                    "current llama-server, bypassing its proxy.")
    p.add_argument("--port", type=int, default=8890,
                   help="port to listen on (default: 8890)")
    p.add_argument("--studio-port", type=int, default=8888,
                   help="Unsloth Studio's API port, used as a fallback when "
                        "no model is loaded (default: 8888)")
    p.add_argument("--upstream-port", type=int, default=None,
                   help="skip auto-discovery and always use this llama-server "
                        "port. Required on non-Windows, where discovery is "
                        "not implemented.")
    p.add_argument("--exclude-port", type=int, action="append", default=[],
                   metavar="N",
                   help="never treat this port as the upstream. Repeatable. "
                        "Use it for other llama-server instances you run.")
    p.add_argument("--tray", action="store_true",
                   help="show a Windows tray icon (needs pywin32)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    global LISTEN_PORT, STUDIO_PORT, FORCED_UPSTREAM, EXCLUDE_PORTS
    args = _parse_args(argv)
    LISTEN_PORT = args.port
    STUDIO_PORT = args.studio_port
    FORCED_UPSTREAM = args.upstream_port
    # Never forward to ourselves or to the proxy we exist to bypass.
    EXCLUDE_PORTS = {LISTEN_PORT, STUDIO_PORT} | set(args.exclude_port)
    _stats["started"] = time.time()

    tray = None
    if args.tray:
        # Lazy and guarded: a box without pywin32 still runs headless.
        try:
            from .tray import Tray
            tray = Tray(sys.modules[__name__],
                        os.environ.get("OMP_FORWARDER_LOG"))
        except Exception as exc:
            log(f"tray unavailable, running headless: {exc!r}")
            tray = None

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind((HOST, LISTEN_PORT))
    except OSError as exc:
        log(f"cannot bind {HOST}:{LISTEN_PORT}: {exc!r}")
        return 1
    srv.listen(128)
    log(f"listening on {HOST}:{LISTEN_PORT}")
    log(f"stats at http://{HOST}:{LISTEN_PORT}/__stats")
    if os.name != "nt" and not FORCED_UPSTREAM:
        log("WARNING: auto-discovery is Windows-only; pass --upstream-port")
    log(f"initial upstream: {discover(force=True)}")

    if tray is not None:
        # Win32 needs its message pump on the thread that created the window,
        # and the accept loop must not block it, so accept moves to a thread.
        threading.Thread(target=_serve_forever, args=(srv,),
                         daemon=True).start()
        tray.run()
        return 0
    _serve_forever(srv)
    return 0
