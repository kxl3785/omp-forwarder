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
import shlex
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
#: GPU index this lane's model server runs on, from --gpu. The operator
#: declares it: no server exposes it, so the dashboard's GPU panel cannot
#: infer it either. None when unset.
FWD_GPU: int | None = None
#: Command that starts this lane's model server when the upstream is a plain
#: process (from --upstream-cmd). Without it a process lane can be stopped
#: from the dashboard but not started: the forwarder knows the PID from the
#: netstat scan, but nothing tells it how the server was launched.
UPSTREAM_CMD: str | None = None
#: Set when the operator pressed stop, cleared by start/restart. While set,
#: the container monitor must not auto-start an exited container -- an
#: "unload the GPU" button that quietly reloads the GPU a minute later is
#: worse than no button.
_operator_stopped: bool = False
#: The process started by --upstream-cmd, if any, so it is not orphaned.
_upstream_child: subprocess.Popen | None = None
#: Model presets: named launch recipes an operator can assign to this lane's
#: GPU from the dashboard (from --presets, default presets.json beside
#: tokens.json). A preset is {"kind": "process"|"container", "port": int or
#: a "{gpu}" template, and for process "cmd", for container "distro",
#: "container" (name template) and "run" (a docker run line)}. Templates
#: take {gpu}, {port} and {name}. Reserved for launch recipes that have been
#: measured; the seed file holds the two that won on 2026-09-05.
PRESETS_FILE: str | None = None
_presets: dict = {}
#: The preset this lane currently fronts, or None. Persisted with the latch.
_preset: str | None = None
#: The container monitor thread runs at most once per process. A lane that
#: starts without --container and later assigns a container preset needs it
#: started on demand, or the dashboard reads "unknown" for a running container.
_container_monitor_started: bool = False
#: Each --peer forwarder's /__control token, read from its /__stats.json by
#: the sampler. Kept out of _peer_state on purpose: the snapshot copies that
#: dict to the page, and a page must never hold another lane's token. The
#: relay in _serve_control is the only reader.
_peer_tokens: dict = {}
#: True once _sample_health has run. stats.snapshot skips the /metrics and
#: /slots readers when the sampler has a verdict and it is "unhealthy".
_health_sampled: bool = False


def _ensure_container_monitor() -> None:
    global _container_monitor_started
    if _container_monitor_started:
        return
    _container_monitor_started = True
    threading.Thread(target=_container_monitor, daemon=True).start()


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
#: When auto-start last ran docker start; it retries at most once a minute.
#: This line went missing in a later round and nothing noticed: the tests
#: create the attribute in reset_state(), so only a real container-mode
#: forwarder crashed, on its first poll. tests/test_unload.py now checks
#: that every `global` name is defined at module level.
_container_last_start: float = 0.0
#: Result of the last /health probe of the upstream, set by _sample_health on
#: the sampler thread. Defined here so a read before the first sample is
#: False rather than a NameError; the same static test guards this one.
_upstream_healthy: bool = False
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
#: Per-peer state, refreshed by the monitor thread: {port: {port, name,
#: healthy, engine, thinking, gpu, reachable}}. Exposed in snapshot() as the
#: `peers` list; snapshot filters out this forwarder's own port.
_peer_state: dict[int, dict] = {}
#: GPU rows from nvidia-smi, refreshed by the monitor thread. Empty when
#: nvidia-smi is missing or fails; the panel then says "no GPU data".
_gpu_state: list[dict] = []

#: The /metrics sample the token tally was last folded from, as
#: (port, prompt_total, cached_total, gen_total). See _tally_tokens.
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


def _run_host(args: list[str], timeout: float = 10.0):
    """Host-side subprocess seam: runs a bare command on the host (nvidia-smi
    and friends) rather than inside a WSL distro. Tests replace this with
    recorded output, exactly as they replace _run_wsl. The timeout matters:
    the monitor thread must never stall on a hung query."""
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


def _spawn_host(args: list[str]) -> subprocess.Popen:
    """Long-lived host child: the model server launched by --upstream-cmd.
    Its own seam, so tests can assert what would be started without
    starting anything."""
    return subprocess.Popen(args, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


def _poll_container_status(auto_start: bool = True) -> None:
    """Read docker inspect once and remember the result in _container_status.
    Runs on the monitor thread's timer; never on the request path.

    auto_start=False for the call main() makes at startup. At that moment
    discover() has not yet assigned _upstream, so an exited container looked
    like "no upstream, container down" and was started -- a fresh forwarder
    reloaded a GPU the operator had just unloaded. Only the monitor thread,
    which runs after discovery, may auto-start."""
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
    # crashing is not restarted in a tight loop -- and never while the
    # operator's own stop is in force, or the stop button would be a lie.
    if (auto_start and _container_status == "exited" and _upstream is None
            and not _operator_stopped):
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
    is kept. The log fires only when the derived port actually changes, not
    every 10-second tick: the value is stable for minutes and a line per
    tick is pure noise."""
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
        changed = port != _container_port
        _container_port = port
        if changed:
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
        _reevaluate_upstream()


def _sample_health() -> None:
    """Remember whether the current upstream answers /health with 200.
    Runs on the sampler thread; never on the request path.

    This is what makes the dashboard's status light honest: SGLang has no
    /metrics (it 404s) but DOES answer /health and /v1/models, so the old
    "live = /metrics sample succeeded" light read a healthy serving upstream
    as red "unreachable". /health is the endpoint every engine behind this
    forwarder actually implements."""
    global _upstream_healthy, _health_sampled
    _health_sampled = True
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



def _sample_peers() -> None:
    """Read each --peer's /__stats.json and remember it in _peer_state.
    Runs on the sampler thread; the dashboard poll only copies the result.

    A browser cannot read a peer at all -- no CORS, and none may be added --
    so the forwarder does the read and publishes it: the pill on this page
    is the only place a peer's name or health shows. GETs go through the
    same replaceable HTTP function the facts panel uses, so tests record
    the answers instead of opening connections. 2 s: the monitor owns the
    whole 10 s cadence, so one slow peer must not stretch it."""
    global _peer_state
    from . import stats as stats_mod
    seen: set[int] = set()
    for port in PEERS:
        if port == LISTEN_PORT:
            continue  # a forwarder never lists itself
        # 5 s, not 2: a peer's first snapshot after ITS start is slow until
        # its own sampler has run once (see stats.snapshot on dead ports),
        # and a 2 s read missed it, so the Lanes panel said "unreachable"
        # and the relay answered "peer not read yet" for the first minute.
        d = stats_mod._http_get_json(port, "/__stats.json", timeout=5)
        if isinstance(d, dict):
            facts = d.get("facts") or {}
            _peer_state[port] = {
                "port": port,
                "name": d.get("name"),
                "healthy": bool(d.get("healthy")),
                "engine": facts.get("engine", "unknown"),
                "thinking": facts.get("thinking", "unknown"),
                "gpu": d.get("gpu"),
                "reachable": True,
                # Preset state for the Lanes panel, so one page can show and
                # drive every card. No token here: see _peer_tokens.
                "preset": d.get("preset"),
                "loading": bool(d.get("loading")),
                "operator_stopped": bool(d.get("operator_stopped")),
                "presets": list(d.get("presets") or []),
                "model": d.get("model") or "",
            }
            if d.get("control_token"):
                _peer_tokens[port] = d["control_token"]
            seen.add(port)
        else:
            # Dead port: keep the last shape but say the read failed. The
            # name/engine/thinking/gpu fields then render grey, which is
            # the honest reading of "we do not know".
            prev = _peer_state.get(port)
            _peer_state[port] = {
                "port": port,
                "name": prev["name"] if prev else None,
                "healthy": False,
                "engine": prev["engine"] if prev else "unknown",
                "thinking": prev["thinking"] if prev else "unknown",
                "gpu": prev["gpu"] if prev else None,
                "reachable": False,
            }
    # Ports no longer in --peer must not keep serving stale state.
    for p in list(_peer_state):
        if p not in seen and p not in PEERS:
            del _peer_state[p]


def _sample_gpus() -> None:
    """Run nvidia-smi once and parse it into _gpu_state. Runs on the
    sampler thread; the dashboard poll only copies the list.

    The query asks for MiB and integer percentages so the panel can show
    GiB to one decimal without floating-point drift in the source. A
    missing binary or a failed run leaves the list empty and the panel
    says "no GPU data" -- there is no second source to fall back on,
    and a half-parsed card would read as a number rather than as
    "nothing"."""
    global _gpu_state
    res = _run_host(["nvidia-smi",
                    "--query-gpu=index,name,memory.used,memory.total,"
                    "utilization.gpu",
                    "--format=csv,noheader,nounits"],
                   timeout=5)
    if res is None or res.returncode != 0 or not res.stdout:
        _gpu_state = []
        return
    out: list[dict] = []
    for line in res.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        # Five fields, all numeric except the name. A malformed line is
        # skipped rather than guessed: a half-parsed card would read as a
        # number.
        if len(parts) != 5:
            continue
        try:
            out.append({
                "index": int(parts[0]),
                "name": parts[1],
                "mem_used_mib": int(parts[2]),
                "mem_total_mib": int(parts[3]),
                "util_pct": int(parts[4]),
            })
        except ValueError:
            continue
    _gpu_state = out


def _upstream_sampler() -> None:
    """Background thread: probes /health and reads the deployment facts every
    10 s so the dashboard's status light and Deployment panel are truthful
    without paying for them on the request path. Started unconditionally,
    alongside the token sampler, so non-container deployments get the fields
    too. Peers and GPUs ride the same cadence: they are all "background
    truth" the request path must not pay for."""
    while True:
        time.sleep(10)
        _sample_health()
        _sample_upstream_facts()
        _sample_peers()
        _sample_gpus()
        # The owner PID of a process upstream, so the Upstream-process bit
        # can say "pid N" and stop has something to terminate. The exe scan
        # never records it for an explicit or candidate port.
        _refresh_upstream_pid()


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

        choice = _choose_upstream()
        if choice is None:
            if _upstream is not None:
                log("no healthy upstream found")
            _upstream = None
            _upstream_exe = None
            _upstream_kind = None
            return None

        port, kind = choice
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


def _choose_upstream() -> tuple[int, str] | None:
    """The preferred healthy upstream, with no side effects on _upstream.

    Re-scans the llama-server processes and the candidate ports, probes
    each with /health in the --prefer order, and returns (port, kind) for
    the first healthy one. None when nothing is healthy. This is the
    scanning core of discover(), factored out so the monitor thread can
    ask "what would discovery pick now?" without committing the result."""
    servers = _listening_ports(_server_pids())
    candidates = _candidate_ports()

    order = (("llama-server", servers), ("candidate", candidates)) \
        if PREFER == "llama-server" else \
        (("candidate", candidates), ("llama-server", servers))

    for kind, ports in order:
        for port in ports:
            if _healthy(port):
                return (port, kind)
    return None


def _reevaluate_upstream() -> None:
    """Periodically re-run discovery so a new healthy upstream can take
    over without waiting for the current one to fail.

    Runs on the monitor thread's 10-second timer. Preference is meaningless
    if it is only evaluated once: the operator wants Studio's server to take
    over whenever it comes back, and the container to take over when it
    goes away. With no explicit --upstream-port, we ask what discover()
    would choose now; if it differs from the current upstream and the new
    choice is healthy, we switch _upstream so new connections go to it.
    Existing relayed connections are untouched. When the current upstream
    is still the preferred healthy choice, nothing is logged. When
    --upstream-port is set, this is a no-op."""
    global _upstream, _upstream_kind, _upstream_exe
    if FORCED_UPSTREAM:
        return
    choice = _choose_upstream()
    if choice is None:
        return
    port, kind = choice
    old = _upstream
    if port == old:
        return
    with _lock:
        if kind == "llama-server":
            owner = _port_owner.get(port)
            _upstream_exe = _exe_path(owner) if owner else None
        else:
            _upstream_exe = None
        _stats["model"] = ""
        _upstream = port
        _upstream_kind = kind
    log(f"upstream -> {HOST}:{port} ({kind}), preferred over {old}")


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
    # llama-server's counters when the engine is llama-server; SGLang's when
    # it is. The two sets never appear together, so picking the right one by
    # presence is safe: a sample with neither set contributes nothing.
    if "sglang:prompt_tokens_total" in metrics or \
       "sglang:generation_tokens_total" in metrics:
        prompt = metrics.get("sglang:prompt_tokens_total", 0.0)
        cached = 0.0  # SGLang does not split prompt vs cached; it is one counter
        gen = metrics.get("sglang:generation_tokens_total", 0.0)
    else:
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




def _upstream_pid() -> str | None:
    """PID of the process behind the current upstream, as recorded in
    _port_owner, or None when the upstream is a container or unknown. A
    dictionary lookup only: this runs on the /__stats.json request path."""
    if WSL_DISTRO and CONTAINER_NAME:
        return None
    if _upstream is None:
        return None
    return _port_owner.get(_upstream)


def _latch_path() -> str | None:
    """Where this lane's operator state is remembered: beside tokens.json,
    one file per listen port so two lanes on one machine do not share it."""
    if not TOKENS_FILE:
        return None
    return os.path.join(os.path.dirname(TOKENS_FILE),
                        f"control-{LISTEN_PORT}.json")


#: What _load_latch found on disk, for main() to adopt: a preset assigned in
#: an earlier run of this lane, with the port and container it launched.
_saved_state: dict = {}


def _load_latch() -> None:
    """Restore the operator's stop and the lane's assignment from disk at
    startup. The latch used to live only in memory, so restarting the
    forwarder forgot the operator's stop and the container monitor reloaded
    the GPU within a minute. An unload -- and an assignment -- must outlive
    the process that performed it."""
    global _operator_stopped, _preset, _saved_state
    path = _latch_path()
    if not path:
        return
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        d = {}
    _operator_stopped = bool(d.get("operator_stopped"))
    _preset = d.get("preset") or None
    _saved_state = dict(d)


def _save_latch() -> None:
    path = _latch_path()
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"operator_stopped": _operator_stopped,
                       "preset": _preset,
                       "upstream_port": FORCED_UPSTREAM,
                       "distro": WSL_DISTRO,
                       "container": CONTAINER_NAME}, fh)
        os.replace(tmp, path)
    except OSError as exc:
        log(f"cannot write {path}: {exc!r}")


def _presets_path() -> str | None:
    if PRESETS_FILE:
        return PRESETS_FILE
    if not TOKENS_FILE:
        return None
    return os.path.join(os.path.dirname(TOKENS_FILE), "presets.json")


def _load_presets() -> None:
    """Read the presets file into _presets. Missing or malformed means no
    presets: the Model row then shows nothing and assign answers 404."""
    global _presets
    path = _presets_path()
    if not path:
        _presets = {}
        return
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        _presets = {k: v for k, v in d.items()
                    if isinstance(v, dict) and v.get("kind") in ("process", "container")}
    except (OSError, ValueError) as exc:
        log(f"presets unreadable at {path}: {exc!r}")
        _presets = {}


def _render(template, gpu: int, port: int | None = None, name: str = "") -> str:
    """Fill {gpu}, {port} and {name} in a preset template. str.replace, not
    str.format: a docker run line carries JSON braces of its own."""
    s = str(template)
    s = s.replace("{gpu}", str(gpu))
    if port is not None:
        s = s.replace("{port}", str(port))
    return s.replace("{name}", name)


def _preset_port(p: dict, gpu: int) -> int:
    return int(_render(p.get("port", 0), gpu))


def _unload_current(timeout: float = 30.0) -> None:
    """Stop whatever this lane fronts, container or process, and drop the
    container-mode globals so the next assignment starts clean."""
    global WSL_DISTRO, CONTAINER_NAME, _container_status
    if WSL_DISTRO and CONTAINER_NAME:
        _run_wsl(["wsl.exe", "-d", WSL_DISTRO, "-u", "root", "--",
                  "docker", "stop", CONTAINER_NAME], timeout=timeout)
        _container_keepalive_stop()
        WSL_DISTRO, CONTAINER_NAME, _container_status = None, None, None
        return
    pid = _upstream_pid()
    if pid is None:
        _refresh_upstream_pid()
        pid = _upstream_pid()
    if pid is not None:
        if os.name == "nt":
            _run_host(["taskkill", "/PID", str(pid), "/F"], timeout=timeout)
        else:
            _run_host(["kill", str(pid)], timeout=timeout)


def _assign_preset(name: str, timeout: float = 60.0) -> tuple[str, int | None]:
    """Put preset `name` on this lane's GPU: unload what is there, launch the
    recipe with {gpu}/{port}/{name} filled, point the lane's upstream at the
    new port and let the existing wait-then-503 path cover the load. Returns
    (status, port). Statuses: "loading", "no-gpu", "unknown-preset".

    The lane's GPU comes from --gpu; a preset never chooses a card. Two
    lanes therefore give every arrangement: tune on one card and SGLang on
    the other, the reverse, or the same recipe on both."""
    global FORCED_UPSTREAM, _upstream, _upstream_kind, _upstream_exe, _upstream_healthy
    global WSL_DISTRO, CONTAINER_NAME, _operator_stopped, _preset, _upstream_child
    if FWD_GPU is None:
        return "no-gpu", None
    p = _presets.get(name)
    if not p:
        return "unknown-preset", None
    gpu = int(FWD_GPU)
    port = _preset_port(p, gpu)
    _unload_current()
    if p["kind"] == "process":
        _upstream_child = _spawn_host(
            shlex.split(_render(p["cmd"], gpu, port), posix=(os.name != "nt")))
    else:
        cname = _render(p.get("container", "lane{gpu}"), gpu, port)
        distro = p["distro"]
        run = _render(p["run"], gpu, port, cname)
        # A stale container of the same name would make docker run fail.
        _run_wsl(["wsl.exe", "-d", distro, "-u", "root", "--",
                  "docker", "rm", "-f", cname], timeout=timeout)
        _run_wsl(["wsl.exe", "-d", distro, "-u", "root", "--", "bash", "-c", run],
                 timeout=timeout)
        WSL_DISTRO, CONTAINER_NAME = distro, cname
        _container_keepalive_start()
        # Show a status at once and keep it live: a lane that started as a
        # process lane has no container monitor yet.
        _poll_container_status(auto_start=False)
        _container_port_from_docker()
        _ensure_container_monitor()
    with _lock:
        # Point the lane at the new port now, as a static --upstream-port
        # lane would be. Clearing _upstream and leaving discover() to re-adopt
        # the forced port looked cleaner and was wrong: discover() runs on a
        # request, so until a client arrived the health sampler probed
        # nothing, the page read "loading" through a served completion, and
        # stop found no PID to kill because the lookup keys on _upstream.
        FORCED_UPSTREAM = port
        _upstream, _upstream_kind, _upstream_exe = port, "explicit", None
        _upstream_healthy = False
        _stats["model"] = ""
    _operator_stopped = False
    _preset = name
    _save_latch()
    log(f"assigned preset {name} to gpu {gpu} on port {port}")
    return "loading", port


def _port_pid(port: int | None) -> str | None:
    """Owning PID of one listening TCP port, from a single netstat pass, any
    executable. The executable scan records owners only for llama-server
    PIDs it matched, so an explicit --upstream-port or a candidate port
    never gets one -- and the first live test of the stop button answered
    "no upstream process to stop" with a 27B model plainly running. Goes
    through _run_host so tests replace it. Windows only, like the scan."""
    if os.name != "nt" or port is None:
        return None
    res = _run_host(["netstat", "-ano", "-p", "TCP"], timeout=30)
    if res is None:
        return None
    suffix = f":{port}"
    for line in res.stdout.splitlines():
        parts = line.split()
        if (len(parts) >= 5 and parts[-2] == "LISTENING"
                and parts[1].endswith(suffix)):
            return parts[-1]
    return None


def _refresh_upstream_pid() -> None:
    """Record the owner of a process upstream so the dashboard and the stop
    control know it. Called from the monitor thread, and once on demand by
    the control handler; never from the request path."""
    if WSL_DISTRO and CONTAINER_NAME:
        return
    port = _upstream
    if port is None:
        return
    pid = _port_pid(port)
    if pid:
        _port_owner[port] = pid
    else:
        _port_owner.pop(port, None)


def _control_action(action: str, timeout: float = 30.0) -> str:
    """Apply start/stop/restart to whatever this lane fronts; return a status.

    Container lane: `docker <action> CONTAINER_NAME` through _run_wsl, then
    one status re-read. Process lane: stop terminates the PID the netstat
    scan attributed to the upstream port; start runs UPSTREAM_CMD through
    _spawn_host, or answers "no-command" when none was given; restart is the
    two in sequence. "stop" is the unload: for a container it frees the GPU
    the container held, for a process it ends the server holding it.

    stop sets _operator_stopped and start/restart clear it, so the container
    monitor's auto-start cannot undo a deliberate unload. Never called on the
    relay path: /__control is a local path, so the handler runs it inline."""
    global _operator_stopped, _upstream_child
    if action not in ("start", "stop", "restart"):
        return "error"
    if WSL_DISTRO and CONTAINER_NAME:
        _run_wsl(["wsl.exe", "-d", WSL_DISTRO, "-u", "root", "--",
                  "docker", action, CONTAINER_NAME],
                 timeout=timeout)
        _operator_stopped = (action == "stop")
        _save_latch()
        _poll_container_status()
        return _container_status or "error"
    status = "error"
    if action in ("stop", "restart"):
        pid = _upstream_pid()
        if pid is None:
            # Right after an assign the sampler has not yet mapped the new
            # port to its owner (it runs every 10 s). One netstat pass here
            # keeps stop honest instead of answering "no-process" over a
            # loaded card.
            _refresh_upstream_pid()
            pid = _upstream_pid()
        if pid is None:
            return "no-process"
        if os.name == "nt":
            _run_host(["taskkill", "/PID", str(pid), "/F"], timeout=timeout)
        else:
            _run_host(["kill", str(pid)], timeout=timeout)
        _operator_stopped = True
        _save_latch()
        status = "stopped"
    if action in ("start", "restart"):
        if not UPSTREAM_CMD:
            return "no-command"
        # posix=False on Windows keeps backslashes in paths intact.
        _upstream_child = _spawn_host(
            shlex.split(UPSTREAM_CMD, posix=(os.name != "nt")))
        _operator_stopped = False
        _save_latch()
        status = "starting"
    return status


_RELAY_STATUS = {200: "200 OK", 400: "400 Bad Request", 403: "403 Forbidden",
                 404: "404 Not Found", 405: "405 Method Not Allowed",
                 409: "409 Conflict", 503: "503 Service Unavailable"}


def _peer_control(port: int, token: str, action: str, preset: str,
                  timeout: float = 90.0) -> tuple[int, dict]:
    """POST one control action to a peer forwarder and return (status, body).
    The lane relay's one network call; tests replace it. 90 s because a
    container stop or an assign can take that long on the far side."""
    import http.client
    q = urllib.parse.urlencode({"token": token, "action": action,
                                **({"preset": preset} if preset else {})})
    conn = http.client.HTTPConnection(HOST, port, timeout=timeout)
    try:
        conn.request("POST", "/__control?" + q)
        resp = conn.getresponse()
        raw = resp.read()
        try:
            body = json.loads(raw.decode("utf-8", "replace"))
        except ValueError:
            body = {"ok": False, "error": raw[:200].decode("utf-8", "replace")}
        return resp.status, body
    except OSError as exc:
        return 502, {"ok": False, "error": f"peer unreachable: {exc!r}"}
    finally:
        conn.close()


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

    # Learn the PID before judging what is controllable: the monitor fills it
    # every 10 s, but a click right after startup arrives sooner, and an
    # unknown PID must not read as "nothing to control". One netstat pass on
    # a local control request; the relay never waits on it.
    if not (WSL_DISTRO and CONTAINER_NAME) and _upstream_pid() is None:
        _refresh_upstream_pid()
    # Nothing to control: no container, no process behind the upstream port,
    # no command that could start one, and no presets to assign.
    if not ((WSL_DISTRO and CONTAINER_NAME) or _upstream_pid()
            or UPSTREAM_CMD or _presets):
        reply("404 Not Found", {"ok": False, "error": "nothing to control"})
        return
    # Only POST mutates; a GET can never do anything.
    if method != "POST":
        reply("405 Method Not Allowed",
              {"ok": False, "error": "POST only"})
        return
    if token != _control_token:
        reply("403 Forbidden", {"ok": False, "error": "bad token"})
        return
    lane = (params.get("lane") or [""])[0]
    if lane and lane != str(LISTEN_PORT):
        # Drive another lane from this page. Gated by THIS lane's token,
        # limited to ports named in --peer (never an open proxy), and the
        # peer's own token is supplied here, server-side: the page never
        # holds it, which is what keeps the no-CORS design intact.
        try:
            lane_port = int(lane)
        except ValueError:
            lane_port = -1
        if lane_port not in PEERS:
            reply("400 Bad Request", {"ok": False, "error": "unknown lane",
                                      "lanes": sorted(PEERS)})
            return
        peer_token = _peer_tokens.get(lane_port)
        if not peer_token:
            reply("503 Service Unavailable",
                  {"ok": False, "lane": lane_port,
                   "error": "peer not read yet; try again in 10 s"})
            return
        code, body = _peer_control(lane_port, peer_token, action,
                                   (params.get("preset") or [""])[0])
        body = dict(body) if isinstance(body, dict) else {"ok": False}
        body["lane"] = lane_port
        reply(_RELAY_STATUS.get(code, "502 Bad Gateway"), body)
        return
    if action == "assign":
        preset = (params.get("preset") or [""])[0]
        status, port = _assign_preset(preset)
        if status == "no-gpu":
            reply("409 Conflict", {"ok": False, "action": action,
                                   "error": "this lane has no --gpu; a preset "
                                            "needs a card to land on"})
        elif status == "unknown-preset":
            reply("400 Bad Request", {"ok": False, "action": action,
                                      "error": "unknown preset",
                                      "presets": sorted(_presets)})
        else:
            reply("200 OK", {"ok": True, "action": action, "status": status,
                             "preset": preset, "port": port})
        return
    if action not in ("start", "stop", "restart"):
        reply("400 Bad Request", {"ok": False, "error": "unknown action"})
        return
    status = _control_action(action)
    if status == "no-command":
        reply("409 Conflict", {"ok": False, "action": action,
                               "error": "no --upstream-cmd: this lane can "
                                        "stop its process but not start one"})
        return
    if status == "no-process":
        reply("409 Conflict", {"ok": False, "action": action,
                               "error": "no upstream process to stop"})
        return
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
    p.add_argument("--presets", default=None, metavar="FILE",
                   help="JSON file of model presets the dashboard can assign "
                        "to this lane's GPU (default: presets.json beside the "
                        "tokens file). See README.")
    p.add_argument("--upstream-cmd", default=None, metavar="COMMAND",
                   help="command that starts this lane's model server when "
                        "it is a plain process (not a container). Enables "
                        "start/restart on the dashboard; stop works without "
                        "it. Run on the host exactly as given.")
    p.add_argument("--gpu", type=int, default=None, metavar="N",
                   help="index of the GPU this lane's model server runs on. "
                        "A label: no server exposes it, so the dashboard's "
                        "GPU panel marks the matching card as this lane.")
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
    global UPSTREAM_EXE, WSL_DISTRO, CONTAINER_NAME, UPSTREAM_CMD, PRESETS_FILE
    global CANDIDATE_PORTS, PREFER
    global _container_status, _container_last_start
    global FWD_NAME, PEERS, FWD_GPU, _control_token
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
    FWD_GPU = args.gpu
    UPSTREAM_CMD = args.upstream_cmd
    PRESETS_FILE = args.presets
    PEERS = list(args.peer)
    # A new process: any peer/GPU state is stale and must not survive.
    _peer_state.clear()
    _gpu_state.clear()
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
    # The operator's stop and assignment from a previous run of this lane.
    # Loaded before the container poll so an unloaded GPU stays unloaded,
    # and so a lane that assigned a preset last time fronts it again without
    # relaunching anything. Flags given on this command line still win.
    _load_latch()
    _load_presets()
    if _preset and _saved_state.get("upstream_port") and not FORCED_UPSTREAM:
        FORCED_UPSTREAM = int(_saved_state["upstream_port"])
        if not (WSL_DISTRO and CONTAINER_NAME) and _saved_state.get("container"):
            WSL_DISTRO = _saved_state.get("distro")
            CONTAINER_NAME = _saved_state.get("container")
        log(f"resuming preset {_preset} on port {FORCED_UPSTREAM}"
            + (f" (container {CONTAINER_NAME})" if CONTAINER_NAME else ""))

    if WSL_DISTRO and CONTAINER_NAME:
        _container_keepalive_start()
        # Status only: discover() has not run yet, so an auto-start here
        # would act on "no upstream" that is merely "not looked yet".
        _poll_container_status(auto_start=False)
        _container_port_from_docker()
        _ensure_container_monitor()

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
