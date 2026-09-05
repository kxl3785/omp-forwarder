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
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from typing import Callable
HOST = "127.0.0.1"

#: Set from the command line by main(). Module-level because the tray and the
#: stats page read live state from this module.
LISTEN_PORT = 8890
STUDIO_PORT = 8888
FORCED_UPSTREAM: int | None = None
EXCLUDE_PORTS: set[int] = set()
#: Seconds to wait for a llama-server to appear before giving up on a request.
#: A model reload leaves a gap of tens of seconds; see _await_upstream.
WAIT_FOR_MODEL = 30
#: Relay to Studio when no llama-server is running. Off by default: Studio
#: requires its own API key, so for a client that does not hold one the
#: fallback returns 401 rather than an answer. See _serve_unavailable.
STUDIO_FALLBACK = False
#: Only consider a llama-server whose executable path contains this substring.
#: Off by default, which keeps the old "any llama-server" behaviour. See
UPSTREAM_EXE: str | None = None
#: Human label for this forwarder (from --name), shown on the dashboard and
#: in the page title. None when unset.
FWD_NAME: str | None = None
#: Ports of other forwarders on this machine (from --peer, repeatable).
#: Rendered as links in the dashboard header; never probed.
PEERS: list[int] = []

#: Container-upstream mode, enabled when both --wsl-distro and --container are
#: given. WSL_DISTRO hosts a Docker container that runs the model server;
#: CONTAINER_NAME is that container.
WSL_DISTRO: str | None = None
CONTAINER_NAME: str | None = None
#: Last observed `docker inspect -f {{.State.Status}}` result ("running",
#: "exited", ...), or None when container mode is off. Read by the dashboard.
_container_status: str | None = None
#: Keepalive child so WSL2 does not tear the distro down between our
#: commands; see _container_keepalive_start.
_container_keepalive: subprocess.Popen | None = None
#: Timestamp of the last auto-start attempt; the monitor retries at most
#: once a minute so a container that keeps crashing is not hammered.
#: Ports given on the command line (--candidate-port, repeatable): model
#: servers that discovery cannot find by executable, e.g. a container
#: published port.
CANDIDATE_PORTS: list[int] = []
#: The container's published port as last read by `docker port` on the
#: monitor thread; None when container mode is off or the call failed.
#: Refreshed on the 10-second timer, never on the request path.
_container_port: int | None = None
#: What the current _upstream was found to be: "llama-server" (executable
#: matched in netstat), "candidate" (--candidate-port / a derived
#: container port), "explicit" (--upstream-port), or None when no upstream.
#: Read by the dashboard; set in discover().
_upstream_kind: str | None = None

_lock = threading.Lock()
_upstream: int | None = None

#: Last /health result for the current upstream, refreshed by the health
#: sampler thread; never probed on the request path. Drives the dashboard's
#: honest "ready" light -- /health (not /metrics) is the answer SGLang gives,
#: so a healthy upstream must not read as "unreachable" just because it has
#: no /metrics endpoint.
#: The configured choice between the two upstream kinds when both are
#: healthy, from --prefer (default llama-server).
PREFER: str = "llama-server"
#: Set when a relayed request completes; the sampler thread waits on it.
#: Deployment facts the monitor thread last read from the upstream's
#: /get_server_info or /props, refreshed every 10 s so the dashboard poll
#: (the request path) only copies them. Shape: {engine, thinking,
#: speculative, parallel, model_path}.
_upstream_facts: dict = {}
#: One random 32-hex string, generated at startup. The /__control endpoint
#: requires it in the query string before it will mutate the container. The
#: dashboard is loopback-only, but any web page the user visits can make the
#: browser send requests to 127.0.0.1; a foreign page cannot read
#: /__stats.json (no CORS headers are sent, and none must be added), so it
#: cannot learn the token, and a GET can never mutate. See _serve_local.
_control_token: str = ""
#: Counters only the forwarder can produce. llama-server's /metrics has no
#: error counter, so without these a 500 from the chat template is completely
#: silent -- which is exactly how a real template bug went unnoticed.
_stats: dict = {"conns": 0, "requests": 0, "2xx": 0, "4xx": 0, "5xx": 0,
                "fallbacks": 0, "unavailable": 0, "latency": [], "model": "",
                "tok_prompt": 0, "tok_cached": 0, "tok_gen": 0,
                "started": time.time()}

#: The /metrics sample the token tally was last folded from, as
#: (port, prompt_tokens_total, prompt_tokens_cached_total,
#: tokens_predicted_total). See _tally_tokens.
_tok_last: tuple[int, float, float, float] | None = None
_tok_lock = threading.Lock()
#: Set when a relayed request completes; the sampler thread waits on it.
_tok_dirty = threading.Event()

#: Per-day token totals, so the figure survives restarts and can be compared
#: with a paid API's daily usage.
#: {"YYYY-MM-DD": {"prompt": n, "cached": n, "gen": n}}, persisted to
#: TOKENS_FILE (beside the log). Guarded by _tok_lock.
#:
#: "prompt" and "cached" are DISJOINT, which is what makes the comparison
#: work: llama-server counts a prompt token in exactly one of them, so they
#: line up with a paid API's "input" and "cache read" lines.
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
        # "cached" is absent in files written before it was tracked; those
        # days keep a 0 rather than being dropped.
        _day_totals = {d: {"prompt": int(v.get("prompt", 0)),
                           "cached": int(v.get("cached", 0)),
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


def _today_tokens() -> tuple[int, int, int]:
    """(prompt, cached, generated) for today, across restarts."""
    with _tok_lock:
        d = _day_totals.get(_today(), {})
        return d.get("prompt", 0), d.get("cached", 0), d.get("gen", 0)


def recent_days(limit: int = 30) -> list[dict]:
    """Newest first, for the usage page. A copy: the caller must not hold
    the lock while rendering."""
    with _tok_lock:
        days = sorted(_day_totals.items(), reverse=True)[:limit]
    return [{"day": d, "prompt": v.get("prompt", 0),
             "cached": v.get("cached", 0), "gen": v.get("gen", 0)}
            for d, v in days]


def _log_day(day: str, suffix: str = "") -> None:
    d = _day_totals.get(day, {})
    log(f"tokens {day}{suffix}: {d.get('prompt', 0):,} prompt, "
        f"{d.get('cached', 0):,} cached, {d.get('gen', 0):,} generated")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def _run_wsl(args: list[str], timeout: float = 10.0):
    """All container-mode subprocess calls go through this one function so
    tests can replace it with a fake that returns recorded output."""
    try:
        return subprocess.run(args, capture_output=True, text=True,
                              timeout=timeout)
    except Exception:
        return None

def _spawn_wsl(args: list[str]) -> subprocess.Popen:
    """Long-lived child (the keepalive). Separate from _run_wsl because
    Popen must not be waited on; tests replace this too."""
    return subprocess.Popen(args, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


def _poll_container_status() -> None:
    """Read docker inspect once and remember the result in _container_status.
    Runs on the monitor thread's timer; never on the request path."""
    global _container_status, _container_last_start
    if not WSL_DISTRO or not CONTAINER_NAME:
        return
    res = _run_wsl(["wsl.exe", "-d", WSL_DISTRO, "-u", "root", "--",
                    "docker", "inspect", "-f", "{{.State.Status}}",
                    CONTAINER_NAME],
                   timeout=10)
    if res is None:
        _container_status = "error"
        return
    _container_status = res.stdout.strip() or "error"
    # Auto-start once per minute: docker start on an exited container, but
    # only when the relay has no healthy upstream, so a container that keeps
    # crashing is not restarted in a tight loop.
    if _container_status == "exited" and _upstream is None:
        now = time.time()
        if now - _container_last_start >= 60:
            _container_last_start = now
            _run_wsl(["wsl.exe", "-d", WSL_DISTRO, "-u", "root", "--",
                      "docker", "start", CONTAINER_NAME],
                     timeout=30)
            log(f"container {CONTAINER_NAME} exited; ran docker start")


def _container_keepalive_start() -> None:
    global _container_keepalive
    # WSL2 shuts a distro down when its last client exits, and Docker inside
    # it then SIGTERMs every container: an upstream launched from a shell
    # dies a few seconds after that shell closes. A resident `sleep infinity`
    # keeps the distro alive for as long as the forwarder runs.
    proc = _spawn_wsl(
        ["wsl.exe", "-d", WSL_DISTRO, "-u", "root", "--", "sleep", "infinity"])
    log(f"keepalive: wsl -d {WSL_DISTRO} pid {proc.pid}")
    _container_keepalive = proc


def _container_keepalive_stop() -> None:
    global _container_keepalive
    if _container_keepalive is not None:
        _container_keepalive.terminate()
        try:
            _container_keepalive.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _container_keepalive.kill()
        _container_keepalive = None
        log("keepalive stopped")


def _container_port_from_docker() -> int | None:
    """The container's published port, or None when `docker port` fails.
    A container inside WSL2 has no Windows process behind its listener, so
    netstat matching by executable path cannot see it: it is invisible to
    executable discovery and can only become a candidate port. Runs on the
    monitor thread's 10-second timer, never on the request path; a failure
    simply means no derived candidate this pass, and the last good value
    is kept."""
    global _container_port
    if not (WSL_DISTRO and CONTAINER_NAME):
        _container_port = None
        return None
    res = _run_wsl(["wsl.exe", "-d", WSL_DISTRO, "-u", "root", "--",
                   "docker", "port", CONTAINER_NAME],
                  timeout=10)
    if res is None or res.returncode != 0:
        return None
    port = _parse_container_port(res.stdout)
    if port is not None:
        _container_port = port
        log(f"container {CONTAINER_NAME} publishes port {port}")
    return port


def _parse_container_port(out: str) -> int | None:
    """The host port a `docker port` output publishes, or None.

    `docker port` prints one line per mapping, IPv4 and IPv6:
        30000/tcp -> 0.0.0.0:30000
        30000/tcp -> [::]:30000
    The host port is what comes after the LAST colon of each line, and the
    IPv6 line repeats the same one. Deduping leaves one port when the
    container publishes one. A malformed or empty output yields None,
    which the caller treats as "no derived candidate"."""
    ports = []
    for line in out.splitlines():
        m = re.search(r":(\d+)\s*$", line.rstrip())
        if m:
            p = int(m.group(1))
            if p not in ports:
                ports.append(p)
    return ports[0] if len(ports) == 1 else None


def _container_monitor() -> None:
    """Background thread: polls container status every 10 s so the dashboard
    shows a live state and auto-start can trigger. Also re-reads the
    container's published port, the only candidate source that can change
    underneath us (a `docker port` re-run after the container comes up).
    Never on the request path."""
    while True:
        time.sleep(10)
        _poll_container_status()
        _container_port_from_docker()


def _sample_health() -> None:
    """Remember whether the current upstream answers /health with 200.
    Runs on the sampler thread; never on the request path.

    This is what makes the dashboard's status light honest: SGLang has no
    /metrics (it 404s) but DOES answer /health and /v1/models, so the old
    "live = /metrics sample succeeded" light read a healthy serving upstream
    as red "unreachable". /health is the endpoint every engine behind this
    forwarder actually implements."""
    global _upstream_healthy
    port = _upstream
    if port is None:
        _upstream_healthy = False
        return
    _upstream_healthy = _healthy(port)


def _sample_upstream_facts() -> None:
    """Read the deployment facts (engine, thinking, speculative, parallel,
    model_path) from the upstream and remember them in _upstream_facts.
    Runs on the sampler thread; the dashboard poll only copies them.
    All HTTP goes through stats.upstream_facts, one replaceable function, so
    the tests swap it for recorded JSON."""
    global _upstream_facts
    from . import stats as stats_mod
    _upstream_facts = stats_mod.upstream_facts(_upstream)



def _upstream_sampler() -> None:
    """Background thread: probes /health and reads the deployment facts every
    10 s so the dashboard's status light and Deployment panel are truthful
    without paying for them on the request path. Started unconditionally,
    alongside the token sampler, so non-container deployments get the fields
    too."""
    while True:
        time.sleep(10)
        _sample_health()
        _sample_upstream_facts()


def _exe_path(pid: str | int) -> str | None:
    """The executable a PID is running, or None.

    WHY THIS EXISTS. Discovery takes the highest healthy port and cannot
    otherwise tell two llama-server processes apart. A second one -- a
    different model, run for something else entirely -- gets picked up and
    answers your requests, with nothing in the reply to say so. That happened
    here: excluding Studio's port made discovery select a 4B model belonging
    to another project.

    The executable path is the only stable discriminator. Ports move on every
    reload and model names change whenever you load a different model; an
    install directory does not.

    WHY ctypes. Reading the path costs 0.03 ms for every llama-server on the
    box, against 252 ms to shell out to PowerShell, and it adds no dependency.
    (`wmic` is not an option: Windows 11 has removed it.) Discovery can run on
    a failed-connect path, so it has to stay cheap."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        import ctypes.wintypes as wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # PROCESS_QUERY_LIMITED_INFORMATION: enough for the image name, and
        # unlike PROCESS_QUERY_INFORMATION it works without extra rights.
        handle = k32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return None
        try:
            buf = ctypes.create_unicode_buffer(32768)
            size = wintypes.DWORD(len(buf))
            if k32.QueryFullProcessImageNameW(handle, 0, buf,
                                              ctypes.byref(size)):
                return buf.value
        finally:
            k32.CloseHandle(handle)
    except Exception:
        return None                 # never let discovery die over this
    return None


#: port -> owning PID, filled in by _listening_ports. Kept so discover() can
#: report which executable the chosen upstream belongs to without a second
#: netstat on every dashboard poll.
_port_owner: dict[int, str] = {}
#: Executable path of the current upstream, for the dashboard.
_upstream_exe: str | None = None


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
    pids = set(re.findall(r'"llama-server\.exe","(\d+)"', out))
    if not UPSTREAM_EXE:
        return pids
    keep, want = set(), UPSTREAM_EXE.lower()
    for pid in pids:
        path = _exe_path(pid)
        if path and want in path.lower():
            keep.add(pid)
        else:
            log(f"ignoring llama-server pid {pid} ({path or 'path unknown'}): "
                f"does not match --upstream-exe {UPSTREAM_EXE!r}")
    return keep


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
                _port_owner[p] = parts[-1]
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


def _candidate_ports() -> list[int]:
    """Ports to probe besides the executable-matched llama-server scan:
    --candidate-port entries in the order given, then the container's
    published port (derived on the monitor thread). Deduped, excluding the
    ports already excluded (our listen port and Studio's). The candidate
    list is never empty-checked: with no candidate at all the scan still
    finds llama-server exactly as before."""
    seen = set()
    out = []
    for port in list(CANDIDATE_PORTS) + ([_container_port] if _container_port else []):
        if port in EXCLUDE_PORTS or port in seen:
            continue
        seen.add(port)
        out.append(port)
    return out


def discover(force: bool = False) -> int | None:
    """The current upstream port, cached. force=True re-scans.

    Two kinds of candidate: "llama-server" (executable-matched, as before)
    and "candidate" (--candidate-port / a derived container port, which the
    executable scan cannot see). Both are probed with /health. Among the
    healthy ones the --prefer setting picks the kind (default llama-server);
    within llama-server the highest healthy port wins, within candidate the
    first healthy one. An explicit --upstream-port still wins over all of
    this and is unchanged."""
    global _upstream, _upstream_exe, _upstream_kind
    with _lock:
        if FORCED_UPSTREAM:
            _upstream = FORCED_UPSTREAM
            _upstream_kind = "explicit"
            return _upstream
        if _upstream and not force and _healthy(_upstream):
            return _upstream

        servers = _listening_ports(_server_pids())
        candidates = _candidate_ports()

        # Which kind we take first. The preference decides which LIST we
        # walk first; within a kind the order is fixed: servers are already
        # highest-port-first, candidates in the order given.
        order = (("llama-server", servers), ("candidate", candidates)) \
            if PREFER == "llama-server" else \
            (("candidate", candidates), ("llama-server", servers))

        for kind, ports in order:
            for port in ports:
                if not _healthy(port):
                    continue
                if kind == "llama-server":
                    owner = _port_owner.get(port)
                    _upstream_exe = _exe_path(owner) if owner else None
                else:
                    _upstream_exe = None
                if port != _upstream:
                    log(f"upstream -> {HOST}:{port} ({kind})"
                        + (f" ({_upstream_exe})" if _upstream_exe else ""))
                    _stats["model"] = ""      # re-read it for the new server
                _upstream = port
                _upstream_kind = kind
                return port
        if _upstream is not None:
            log("no healthy upstream found")
        _upstream = None
        _upstream_exe = None
        _upstream_kind = None
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
    cached = metrics.get("llamacpp:prompt_tokens_cached_total", 0.0)
    gen = metrics.get("llamacpp:tokens_predicted_total", 0.0)
    with _tok_lock:
        last = _tok_last
        _tok_last = (port, prompt, cached, gen)
        if baseline:
            return
        if (last is None or last[0] != port or prompt < last[1]
                or cached < last[2] or gen < last[3]):
            last = (port, 0.0, 0.0, 0.0)
        dp = int(prompt - last[1])
        dc = int(cached - last[2])
        dg = int(gen - last[3])
        _stats["tok_prompt"] += dp
        _stats["tok_cached"] += dc
        _stats["tok_gen"] += dg
        # Per-day book-keeping. The finished day is logged at the first
        # sample of the next one, which is the earliest moment we know it
        # is over.
        day = _today()
        if _last_day and day != _last_day:
            _log_day(_last_day)
        _last_day = day
        if dp or dc or dg:
            t = _day_totals.setdefault(day, {"prompt": 0, "cached": 0,
                                             "gen": 0})
            t["prompt"] += dp
            t["cached"] = t.get("cached", 0) + dc
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


#: Paths the forwarder answers itself instead of relaying.
#:
#: /favicon.ico is here for a reason that is not cosmetic. The relay routes
#: only the FIRST request on a connection; anything pipelined after it follows
#: wherever that first one went. A browser opening the dashboard asks for
#: /favicon.ico first, that got relayed upstream, and the page's own
#: /__stats.json fetch then reused the same keep-alive connection and reached
#: llama-server instead -- a 404, and a page of zeros on first load. Answering
#: it here, with Connection: close, keeps the JSON on a fresh connection.
LOCAL_PREFIXES = ("/__stats", "/__usage", "/__control", "/favicon.ico")




def _control_action(action: str, timeout: float = 30.0) -> str:
    """Run `docker <action> CONTAINER_NAME` in the WSL distro via the
    _run_wsl seam, then re-read the container status once and return it.
    Only start/stop/restart are accepted; anything else is "" so the caller
    can reject it. Never called on the relay path: the handler runs it
    synchronously because /__control is a local path, not the relay."""
    if not WSL_DISTRO or not CONTAINER_NAME:
        return "error"
    if action not in ("start", "stop", "restart"):
        return "error"
    _run_wsl(["wsl.exe", "-d", WSL_DISTRO, "-u", "root", "--",
              "docker", action, CONTAINER_NAME],
             timeout=timeout)
    _poll_container_status()
    return _container_status or "error"


def _serve_control(client: socket.socket, path: str, method: str) -> None:
    """POST /__control?token=<T>&action=<start|stop|restart>.
    Routes on the first line's method and query string only; the body is
    never read. GET (any query) -> 405. Wrong or missing token -> 403.
    Unknown action -> 400. Container mode off -> 404. Success -> 200 with a
    tiny JSON body. The token is checked before anything runs, so the only
    way to mutate the container is to hold the token, which a foreign page
    cannot read because /__stats.json sends no CORS headers.
    Answers with Connection: close for the same reason /__stats does: it is
    a local path and the browser must not reuse a keep-alive connection."""
    # Split off the query string; the path part is /__control.
    _, _, query = path.partition("?")
    params = urllib.parse.parse_qs(query, keep_blank_values=True)
    token = (params.get("token") or [""])[0]
    action = (params.get("action") or [""])[0]

    def reply(status: str, body: dict) -> None:
        raw = json.dumps(body).encode()
        lines = [
            f"HTTP/1.1 {status}",
            "Content-Type: application/json",
            f"Content-Length: {len(raw)}",
            "Cache-Control: no-store",
            "Connection: close",
            "", "",
        ]
        try:
            client.sendall("\r\n".join(lines).encode() + raw)
        except OSError:
            pass

    # Container mode off: nothing to control.
    if not (WSL_DISTRO and CONTAINER_NAME):
        reply("404 Not Found", {"ok": False, "error": "container mode off"})
        return
    # Only POST mutates; a GET can never do anything.
    if method != "POST":
        reply("405 Method Not Allowed",
              {"ok": False, "error": "POST only"})
        return
    if token != _control_token:
        reply("403 Forbidden", {"ok": False, "error": "bad token"})
        return
    if action not in ("start", "stop", "restart"):
        reply("400 Bad Request", {"ok": False, "error": "unknown action"})
        return
    status = _control_action(action)
    reply("200 OK", {"ok": True, "action": action, "status": status})


def _serve_local(client: socket.socket, path: str, method: str = "GET") -> None:
    """Serve the live dashboard (/__stats), the usage page (/__usage), the
    JSON both of them poll (/__stats.json), the container controls
    (/__control), and /favicon.ico. method is the first line's verb; only
    /__control branches on it. The body is never read or parsed."""
    from . import stats as stats_mod
    # sys.modules[__name__] rather than importing this module by name: under
    # some entry points that would build a SECOND module object with its own
    # _upstream, which then reads as None while the real one is serving.
    me = sys.modules[__name__]
    if path.startswith("/__control"):
        _serve_control(client, path, method)
        return
    if path.startswith("/favicon.ico"):
        try:
            client.sendall(b"HTTP/1.1 204 No Content\r\n"
                           b"Cache-Control: max-age=86400\r\n"
                           b"Connection: close\r\n\r\n")
        except OSError:
            pass
        return
    if path.startswith(("/__stats.json", "/__usage.json")):
        if not _stats.get("model"):
            _stats["model"] = stats_mod.upstream_model(_upstream)
        body = json.dumps(stats_mod.snapshot(me, _stats)).encode()
        ctype = "application/json"
    elif path.startswith("/__usage"):
        from . import usage as usage_mod
        body = usage_mod.PAGE.encode("utf-8")
        ctype = "text/html; charset=utf-8"
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


def _await_upstream() -> int | None:
    """The current llama-server port, waiting up to WAIT_FOR_MODEL seconds for
    one to appear.

    A model reload leaves a window with no llama-server at all. Measured on a
    27B model: a request arrived 7 s into the gap and the server appeared 34 s
    later. Failing instantly there turns "the model is still loading" into an
    error, so this polls instead. The cost is a slow first request after a
    reload, which is the right trade: the client gets its answer."""
    port = discover()
    if port is not None:
        return port
    port = discover(force=True)
    if port is not None or WAIT_FOR_MODEL <= 0:
        return port
    deadline = time.time() + WAIT_FOR_MODEL
    log(f"no llama-server; waiting up to {WAIT_FOR_MODEL}s for one")
    while time.time() < deadline:
        time.sleep(1.0)
        port = discover(force=True)
        if port is not None:
            log(f"llama-server appeared after "
                f"{WAIT_FOR_MODEL - (deadline - time.time()):.0f}s")
            return port
    return None


def _serve_unavailable(client: socket.socket) -> None:
    """503 with Retry-After, in the OpenAI error shape.

    WHY NOT FALL BACK TO STUDIO. Studio's API requires its own key. A client
    pointed at this forwarder sends whatever key it likes, because
    llama-server ignores keys entirely -- so relaying to Studio hands that
    client a 401. That is the worst signal available: 401 means "your
    credentials are wrong, stop trying", when the truth is "the model is
    loading, try again shortly". It cost a real outage. A 503 with
    Retry-After says the right thing, and every OpenAI client understands it.

    --studio-fallback restores the old behaviour for a client that does hold
    Studio's key, where the fallback really does trigger load-on-demand."""
    body = json.dumps({"error": {
        "message": ("No llama-server is running. Unsloth Studio is probably "
                    "loading a model; retry shortly."),
        "type": "service_unavailable",
        "code": "model_not_loaded",
    }}).encode()
    head = "\r\n".join([
        "HTTP/1.1 503 Service Unavailable",
        "Content-Type: application/json",
        f"Content-Length: {len(body)}",
        "Retry-After: 5",
        "Cache-Control: no-store",
        "Connection: close",
        "", "",
    ]).encode()
    try:
        client.sendall(head + body)
    except OSError:
        pass


def handle(client: socket.socket) -> None:
    _stats["conns"] += 1
    head = _read_request_head(client)
    if not head:
        client.close()
        return
    # parts after METHOD, so TARGET keeps its whole query string and we never
    # look past the first line into the body.
    parts = head.split(b" ", 2)
    method = parts[0].decode("latin-1") if parts else ""
    target = parts[1].decode("latin-1") if len(parts) > 1 else ""
    if target.startswith(LOCAL_PREFIXES):
        _serve_local(client, target, method)
        client.close()
        return

    _stats["requests"] += 1
    started = time.time()
    port = _await_upstream()
    if port is None:
        # Nothing to relay to. Answer 503 ourselves rather than handing the
        # client Studio's 401 -- see _serve_unavailable.
        if not STUDIO_FALLBACK:
            _stats["unavailable"] += 1
            log(f"no llama-server after {WAIT_FOR_MODEL}s; returning 503")
            _serve_unavailable(client)
            client.close()
            return
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
    p.add_argument("--wait-for-model", type=int, default=30, metavar="SECONDS",
                   help="when no llama-server is running, wait this long for "
                        "one to appear before failing the request. A model "
                        "reload leaves a gap of tens of seconds. 0 disables "
                        "the wait (default: 30)")
    p.add_argument("--upstream-exe", default=None, metavar="SUBSTRING",
                   help="only use a llama-server whose executable path "
                        "contains this. Use it when you run more than one: "
                        "discovery cannot otherwise tell them apart, and the "
                        "wrong model will answer silently. Try '.unsloth'")
    p.add_argument("--candidate-port", type=int, action="append", default=[],
                   metavar="PORT",
                   help="a model server discovery cannot find by executable "
                        "path, e.g. a container's published port. Probed "
                        "with /health like everything else. Repeatable; "
                        "order is the tie-break among healthy candidates.")
    p.add_argument("--prefer", choices=("llama-server", "candidate"),
                   default="llama-server",
                   help="which kind of upstream to take when both are "
                        "healthy: the executable-matched llama-server "
                        "(default) or a --candidate-port / derived container "
                        "port. The preferred kind wins; the other is the "
                        "fallback when nothing of the preferred kind is up.")
    p.add_argument("--studio-fallback", action="store_true",
                   help="relay to Studio when no llama-server is running, "
                        "instead of answering 503. Only useful if your client "
                        "sends Studio's API key; without it Studio returns "
                        "401, which clients read as a fatal config error")
    p.add_argument("--tray", action="store_true",
                   help="show a Windows tray icon (needs pywin32)")
    p.add_argument("--tokens-file", default=None, metavar="PATH",
                   help="where per-day token totals are kept (default: "
                        "tokens.json beside the log, or in "
                        "%%LOCALAPPDATA%%\\omp-forwarder)")
    p.add_argument("--wsl-distro", default=None, metavar="NAME",
                   help="WSL distro that hosts the upstream Docker container "
                        "(e.g. Ubuntu-24.04). Use together with --container.")
    p.add_argument("--container", default=None, metavar="NAME",
                   help="Docker container name inside the distro (e.g. sgl). "
                        "Use together with --wsl-distro.")
    p.add_argument("--name", default=None, metavar="TEXT",
                   help="a human label for this forwarder, shown on the "
                        "dashboard and in the page title, e.g. 'GPU1 · "
                        "thinking on'. Useful when two forwarders (one per "
                        "GPU) run side by side.")
    p.add_argument("--peer", type=int, action="append", default=[],
                   metavar="PORT",
                   help="another forwarder on this machine; rendered as a "
                        "link in the dashboard header. Repeatable. Links only; "
                        "peers are never probed.")
    return p.parse_args(argv)


def _serve_and_cleanup(serve: Callable[[], None]) -> None:
    """Run the serve step, then flush day stats and stop the keepalive.
    finally: must run on every exit path (including exceptions) so the
    wsl.exe sleep-infinity child is never orphaned and WSL2 can tear the
    distro down.
    """
    try:
        serve()
        # Clean exit (tray Exit, or Ctrl-C): flush the day file and say where
        # today stands, so the log alone answers "how much did I use today".
        _sample_tokens()
        _save_days()
        _log_day(_today(), " at exit")
    finally:
        _container_keepalive_stop()



def main(argv: list[str] | None = None) -> int:
    global LISTEN_PORT, STUDIO_PORT, FORCED_UPSTREAM, EXCLUDE_PORTS
    global TOKENS_FILE, _last_day, WAIT_FOR_MODEL, STUDIO_FALLBACK
    global UPSTREAM_EXE, WSL_DISTRO, CONTAINER_NAME
    global CANDIDATE_PORTS, PREFER
    global _container_status, _container_last_start
    global FWD_NAME, PEERS, _control_token
    args = _parse_args(argv)

    # Validate the two container flags: both or neither.
    if (args.wsl_distro and not args.container) or \
       (args.container and not args.wsl_distro):
        log("--wsl-distro and --container must be given together")
        return 1

    UPSTREAM_EXE = args.upstream_exe
    CANDIDATE_PORTS = list(args.candidate_port)
    PREFER = args.prefer
    WSL_DISTRO = args.wsl_distro
    CONTAINER_NAME = args.container
    FWD_NAME = args.name
    PEERS = list(args.peer)
    # One token for the lifetime of this process: the /__control endpoint
    # checks it, and the dashboard learns it from /__stats.json.
    _control_token = secrets.token_hex(16)
    LISTEN_PORT = args.port
    STUDIO_PORT = args.studio_port
    FORCED_UPSTREAM = args.upstream_port
    WAIT_FOR_MODEL = args.wait_for_model
    STUDIO_FALLBACK = args.studio_fallback
    # Never forward to ourselves or to the proxy we exist to bypass.
    EXCLUDE_PORTS = {LISTEN_PORT, STUDIO_PORT} | set(args.exclude_port)
    _stats["started"] = time.time()
    TOKENS_FILE = args.tokens_file or _default_tokens_file()
    _load_days()
    _last_day = _today()

    if WSL_DISTRO and CONTAINER_NAME:
        _container_keepalive_start()
        _poll_container_status()
        _container_port_from_docker()
        threading.Thread(target=_container_monitor, daemon=True).start()

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
    log(f"usage at http://{HOST}:{LISTEN_PORT}/__usage")
    if os.name != "nt" and not FORCED_UPSTREAM:
        log("WARNING: auto-discovery is Windows-only; pass --upstream-port")
    log(f"initial upstream: {discover(force=True)}")
    # Whatever that server counted before now was not our traffic.
    _sample_tokens(baseline=True)
    threading.Thread(target=_token_sampler, daemon=True).start()
    threading.Thread(target=_upstream_sampler, daemon=True).start()
    log(f"token totals in {TOKENS_FILE}")
    if any(_today_tokens()):
        _log_day(_today(), " so far, from earlier runs")

    if tray is not None:
        # Win32 needs its message pump on the thread that created the window,
        # and the accept loop must not block it, so accept moves to a thread.
        threading.Thread(target=_serve_forever, args=(srv,),
                         daemon=True).start()
        _serve_and_cleanup(tray.run)
    else:
        _serve_and_cleanup(lambda: _serve_forever(srv))

    return 0
