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
                "tok_prompt": 0, "tok_gen": 0,
                "started": time.time()}

#: The /metrics sample the token tally was last folded from, as
#: (port, prompt_tokens_total, tokens_predicted_total). See _tally_tokens.
_tok_last: tuple[int, float, float] | None = None
_tok_lock = threading.Lock()
#: Set when a relayed request completes; the sampler thread waits on it.
_tok_dirty = threading.Event()

#: Per-day token totals, so the figure survives restarts and can be compared
#: with a paid API's daily usage. {"YYYY-MM-DD": {"prompt": n, "gen": n}},
#: persisted to TOKENS_FILE (beside the log). Guarded by _tok_lock.
TOKENS_FILE: str | None = None
_day_totals: dict = {}
_days_dirty = False
_last_day: str | None = None


def _today() -> str:
    """Local date. A function so tests can move the clock."""
    return time.strftime("%Y-%m-%d")


def _default_tokens_file() -> str:
    log_path = os.environ.get("OMP_FORWARDER_LOG")
    if log_path:
        base = os.path.dirname(log_path)
    elif os.name == "nt":
        base = os.path.join(os.environ.get("LOCALAPPDATA")
                            or os.path.expanduser("~"), "omp-forwarder")
    else:
        base = os.path.join(os.path.expanduser("~"), ".omp-forwarder")
    return os.path.join(base, "tokens.json")


def _load_days() -> None:
    global _day_totals
    if not TOKENS_FILE:
        return
    try:
        with open(TOKENS_FILE, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return                      # first run, or unreadable: start empty
    if isinstance(raw, dict):
        _day_totals = {d: {"prompt": int(v.get("prompt", 0)),
                           "gen": int(v.get("gen", 0))}
                       for d, v in raw.items() if isinstance(v, dict)}


def _save_days() -> None:
    global _days_dirty
    if not TOKENS_FILE:
        return
    with _tok_lock:
        data = json.dumps(_day_totals, indent=1, sort_keys=True)
        _days_dirty = False
    try:
        os.makedirs(os.path.dirname(TOKENS_FILE), exist_ok=True)
        tmp = TOKENS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(data)
        os.replace(tmp, TOKENS_FILE)       # never leave a half-written file
    except OSError as exc:
        log(f"cannot write {TOKENS_FILE}: {exc!r}")


def _today_tokens() -> tuple[int, int]:
    with _tok_lock:
        d = _day_totals.get(_today(), {})
        return d.get("prompt", 0), d.get("gen", 0)


def _log_day(day: str, suffix: str = "") -> None:
    d = _day_totals.get(day, {})
    log(f"tokens {day}{suffix}: {d.get('prompt', 0):,} prompt, "
        f"{d.get('gen', 0):,} generated")


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


def _tally_tokens(port: int | None, metrics: dict,
                  baseline: bool = False) -> None:
    """Fold one /metrics sample into the forwarder's own token tally.

    Why the forwarder keeps a tally at all: every llama-server counter is
    cumulative since THAT PROCESS started, and Studio starts a new process on
    every model reload. So a total read straight from /metrics covers only the
    current process, and a dashboard baseline taken before a reload reads
    garbage afterwards. The forwarder outlives reloads. It differences
    successive samples and adds the deltas up.

    A port it has not seen, or a counter that went backwards on the same port,
    means a new process. Everything that process has counted happened on our
    watch, so it goes in whole. baseline=True records the sample without
    adding, for the server that was already running when we started: its
    earlier work was not ours.

    What it misses: tokens the OLD process produced after the last sample and
    before it died. The sampler runs a couple of seconds after requests
    complete, so that window is small. What it does not distinguish: this is
    everything llama-server counted while we were running, so requests from
    Studio's own UI or another client land in it too. Only body parsing could
    separate them, and the relay must not parse bodies."""
    global _tok_last, _last_day, _days_dirty
    if not port or not metrics:
        return
    prompt = metrics.get("llamacpp:prompt_tokens_total", 0.0)
    gen = metrics.get("llamacpp:tokens_predicted_total", 0.0)
    with _tok_lock:
        last = _tok_last
        _tok_last = (port, prompt, gen)
        if baseline:
            return
        if (last is None or last[0] != port
                or prompt < last[1] or gen < last[2]):
            last = (port, 0.0, 0.0)
        dp, dg = int(prompt - last[1]), int(gen - last[2])
        _stats["tok_prompt"] += dp
        _stats["tok_gen"] += dg
        # Per-day book-keeping. The finished day is logged at the first
        # sample of the next one, which is the earliest moment we know it
        # is over.
        day = _today()
        if _last_day and day != _last_day:
            _log_day(_last_day)
        _last_day = day
        if dp or dg:
            t = _day_totals.setdefault(day, {"prompt": 0, "gen": 0})
            t["prompt"] += dp
            t["gen"] += dg
            _days_dirty = True


def _sample_tokens(baseline: bool = False) -> None:
    from . import stats as stats_mod
    port = _upstream
    _tally_tokens(port, stats_mod.upstream_metrics(port), baseline)


def _token_sampler() -> None:
    """Sample shortly after requests complete, so the tally stays current
    with the dashboard closed and a reload loses at most a few seconds.

    Also persists the per-day totals, at most every 30 s and within 30 s of
    the last change: the wait has a timeout so a burst that ends is still
    written even if no further request ever arrives."""
    last_save = 0.0
    while True:
        if _tok_dirty.wait(timeout=30):
            time.sleep(2)
            _tok_dirty.clear()
            _sample_tokens()
        if _days_dirty and time.time() - last_save >= 30:
            _save_days()
            last_save = time.time()


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
    _tok_dirty.set()
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
            # A closed listener raises on every accept, so without this the
            # loop would spin and log forever. The tests close it to stop us;
            # in production nothing closes it. On Linux a blocked accept() is
            # woken by shutdown(), not by close(), so the tests call shutdown
            # first and there is a moment where the socket is shut but not yet
            # closed: the pause below covers it, and also keeps a persistent
            # accept failure (EMFILE, say) from becoming a hot loop.
            if srv.fileno() == -1:
                return
            log(f"accept failed: {exc!r}")
            time.sleep(0.1)
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
    p.add_argument("--tokens-file", default=None, metavar="PATH",
                   help="where per-day token totals are kept (default: "
                        "tokens.json beside the log, or in "
                        "%%LOCALAPPDATA%%\\omp-forwarder)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    global LISTEN_PORT, STUDIO_PORT, FORCED_UPSTREAM, EXCLUDE_PORTS
    global TOKENS_FILE, _last_day
    args = _parse_args(argv)
    LISTEN_PORT = args.port
    STUDIO_PORT = args.studio_port
    FORCED_UPSTREAM = args.upstream_port
    # Never forward to ourselves or to the proxy we exist to bypass.
    EXCLUDE_PORTS = {LISTEN_PORT, STUDIO_PORT} | set(args.exclude_port)
    _stats["started"] = time.time()
    TOKENS_FILE = args.tokens_file or _default_tokens_file()
    _load_days()
    _last_day = _today()

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
    # Whatever that server counted before now was not our traffic.
    _sample_tokens(baseline=True)
    threading.Thread(target=_token_sampler, daemon=True).start()
    log(f"token totals in {TOKENS_FILE}")
    if any(_today_tokens()):
        _log_day(_today(), " so far, from earlier runs")

    if tray is not None:
        # Win32 needs its message pump on the thread that created the window,
        # and the accept loop must not block it, so accept moves to a thread.
        threading.Thread(target=_serve_forever, args=(srv,),
                         daemon=True).start()
        tray.run()
    else:
        _serve_forever(srv)
    # Clean exit (tray Exit, or Ctrl-C): flush the day file and say where
    # today stands, so the log alone answers "how much did I use today".
    _sample_tokens()
    _save_days()
    _log_day(_today(), " at exit")
    return 0
