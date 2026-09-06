"""The dashboard served at http://127.0.0.1:<port>/__stats

WHY IT EXISTS: Unsloth Studio's API panel only counts traffic through its own
proxy. A client pointed at this forwarder bypasses that proxy, so Studio goes
blind to exactly the workload you care about. This replaces it, and adds three
numbers Studio never had -- prefill rate, speculative-decode draft acceptance,
and prompt-cache hit rate.

Those three are the ones that find real faults. Draft acceptance collapsing
toward 35% means speculation quietly fell back to n-gram drafting and
throughput roughly halved. A falling cache-hit rate means prompts stopped
being stable, which costs far more than any decode tuning.

Division of labour: llama-server's own /metrics is authoritative for anything
about the model, because it is the thing doing the work. The forwarder only
contributes what it alone can see -- request counts, HTTP status codes and
round-trip latency -- because /metrics has no error counter, so a 500 from the
chat template is otherwise invisible.

Every field was checked against a live /metrics dump from llama.cpp build
b6xxx (2026-09). Notably ``kv_cache_usage_ratio`` does NOT exist there, so
there is deliberately no KV card -- it would read 0 forever.

SESSION RESET: every counter llama-server exposes is cumulative since the
server started, which is the wrong window when you want to know what just
happened. Each section carries a reset control that stores a baseline in the
browser and subtracts it, so the figures become "since you clicked". The
baseline is per-browser and never leaves it: the server keeps no session
state, so a reset in one window cannot disturb another.
"""
from __future__ import annotations

import json
import time
import urllib.request

#: Fallback only. This module is imported lazily, on the first /__stats hit,
#: so its own import time is not the forwarder's start time -- the forwarder
#: passes the real one in stats["started"].
import collections
import threading

_STARTED = time.time()

#: Finished streams, newest last. A slot is busy only while it generates,
#: and the page polls every 3 s, so every request shorter than that was
#: invisible: the panel read "4 slots idle" through a working agent loop.
#: A watcher thread samples /slots every second while anything is busy and
#: keeps what finished, so the panel can show traffic between bursts.
_stream_hist: dict = {}
_recent_streams: collections.deque = collections.deque(maxlen=12)
_watcher_started = False
#: The first sample only seeds: every slot already carries the id_task of
#: whatever it last ran, and those are history, not new traffic.
_seeded = False


def note_slots(slots: list, now: float) -> None:
    """Fold one /slots sample into the recent-stream record.

    Watching only BUSY slots missed most real traffic: an agent turn can
    finish in 0.6 s and no sampler at a sane cadence sees it running. But
    llama-server keeps a released slot's id_task and its final n_decoded
    until that slot takes new work, so a finished stream stays readable
    after the fact. Every slot is tracked, busy or idle, and a stream is
    recorded the first sample it is seen idle -- not when its slot is next
    used, which would lose every stream that landed on a slot nothing else
    touched again.

    n_decoded restarts with each task, so the task id is the only safe
    identity. A duration is real only if the stream was seen busy at two
    samples; otherwise the token count stands alone and the rate is None."""
    global _seeded
    for sl in slots or []:
        sid, task = sl.get("id"), sl.get("task")
        dec = sl.get("decoded", 0) or 0
        busy = bool(sl.get("busy"))
        h = _stream_hist.get(sid)
        if h is None or h["task"] != task:
            # A different task: whatever this slot held has finished. It is
            # recorded here only if it was never recorded on going idle.
            if h is not None and not h.get("seed") and not h.get("done"):
                _finish(sid, h)
            h = {"task": task, "decoded": dec, "prompt": sl.get("prompt", 0),
                 "cached": sl.get("cached", 0), "t0": now, "t": now,
                 # The decode clock starts at the first token, not when the
                 # slot took the request: see _finish.
                 "t_dec": now if (busy and dec > 0) else None,
                 "dec0": dec if (busy and dec > 0) else 0,
                 "seen": 1 if busy else 0,
                 # The task a slot happens to carry when this watcher starts
                 # finished before anyone was looking. Seeds never report.
                 "seed": not _seeded, "done": False}
            _stream_hist[sid] = h
        else:
            if dec >= h["decoded"]:
                h["decoded"] = dec
                h["prompt"] = sl.get("prompt", h["prompt"])
                h["cached"] = sl.get("cached", h["cached"])
            if busy:
                h["t"] = now
                h["seen"] += 1
        if not busy and not h["done"] and not h["seed"]:
            _finish(sid, h)
            h["done"] = True
    _seeded = True


def _finish(sid, h) -> None:
    if h["decoded"] <= 0:
        return
    secs = h["t"] - h["t0"]
    # One sample means the token count is real and the duration is not.
    rate = round(h["decoded"] / secs, 1) if (h["seen"] >= 2 and secs >= 1.0) else None
    _recent_streams.append({
        "slot": sid, "tokens": h["decoded"],
        "seconds": round(max(secs, 0.0), 1) if rate is not None else None,
        "rate": rate, "prompt": h["prompt"], "cached": h["cached"],
        "ended": h["t"],
    })


def recent_streams(lane) -> list:
    """The finished streams, newest first, each tagged with its lane."""
    return [dict(r, lane=lane) for r in reversed(_recent_streams)]


def _stream_watcher(fwd) -> None:
    """Sample /slots once a second while anything is busy, three seconds
    otherwise. Never on the request path; a dead or unhealthy upstream is
    not asked at all (a closed loopback port costs a 2 s timeout here)."""
    while True:
        busy = False
        try:
            if getattr(fwd, "_upstream_healthy", False):
                sl = upstream_slots(getattr(fwd, "_upstream", None))
                busy = any(x.get("busy") for x in sl)
                note_slots(sl, time.time())
        except Exception:
            pass
        # A constant second, busy or not. The first version slept 3 s while
        # idle and so missed every request shorter than that -- which is the
        # exact case this list exists to show. One /slots read per second on
        # loopback is cheap; an unhealthy upstream is not read at all.
        time.sleep(1.0)


def ensure_stream_watcher(fwd) -> None:
    global _watcher_started
    if _watcher_started:
        return
    _watcher_started = True
    threading.Thread(target=_stream_watcher, args=(fwd,), daemon=True).start()



def parse_metrics(text: str) -> dict:
    """Parse a Prometheus body. llama-server's lines have no labels; SGLang's
    carry a {label="value",...} block between the name and the value. The
    name is what we key on, so labels are stripped first.

    SGLang emits the SAME name once per label set (one line per engine
    instance / rank), so a key may appear on several lines. The spec says
    to sum those duplicates, not keep the last: the last one would
    under-count every metric the engine splits across ranks."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # name{labels} value  ->  name value. The value is whatever follows
        # the closing brace, when one is present; the label block itself is
        # dropped. Without the brace, the value is the last field.
        head, _, tail = s.partition("}")
        # head = "name{labels" or just "name".
        name = head.split("{", 1)[0].split(" ", 1)[0]
        if not name:
            continue
        val = tail.strip() if tail else head.rsplit(" ", 1)[-1]
        try:
            out[name] = out.get(name, 0.0) + float(val)
        except ValueError:
            pass
    return out


def upstream_metrics(port: int | None) -> dict:
    if not port:
        return {}
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/metrics", timeout=4) as r:
            return parse_metrics(r.read().decode("utf-8", "replace"))
    except Exception:
        return {}


def upstream_slots(port: int | None) -> list[dict]:
    """Per-slot state from llama-server's /slots.

    This is the only source of PER-STREAM throughput. /metrics aggregates
    everything, so with several requests in flight it cannot tell you that one
    stream is running at 66 tok/s on a 107k-token context while another does
    59 on 45k. The counter is next_token[0].n_decoded, which the page
    differences.

    id_task matters: n_decoded restarts when a slot picks up a new request, so
    a rate is only meaningful while id_task is unchanged."""
    if not port:
        return []
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/slots", timeout=4) as r:
            raw = json.loads(r.read().decode("utf-8", "replace"))
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    out = []
    for s in raw:
        nt = s.get("next_token") or [{}]
        nt = nt[0] if isinstance(nt, list) and nt else {}
        out.append({
            "id": s.get("id"),
            "task": s.get("id_task"),
            "busy": bool(s.get("is_processing")),
            "decoded": nt.get("n_decoded", 0),
            "remain": nt.get("n_remain", 0),
            "prompt": s.get("n_prompt_tokens", 0),
            "cached": s.get("n_prompt_tokens_cache", 0),
            "spec": bool(s.get("speculative")),
            # With --kv-unified this is the whole pool, shared by every slot.
            "n_ctx": s.get("n_ctx", 0),
        })
    return out


def upstream_model(port: int | None) -> str:
    if not port:
        return "-"
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/v1/models", timeout=4) as r:
            d = json.loads(r.read().decode("utf-8", "replace"))
    except Exception:
        return "-"
    items = d.get("models") or d.get("data") or []
    if not items:
        return "-"
    name = items[0].get("name") or items[0].get("id") or "-"
    return name.split("/")[-1]



def _http_get_json(port: int | None, path: str,
                   timeout: int = 4) -> dict | None:
    """GET a JSON endpoint on the upstream, or None on any failure. This is
    the ONE place the deployment-facts panel talks to the upstream: tests
    replace this with recorded JSON so they never open a real connection.
    (The metrics/slots/model readers above keep their own urlopen calls;
    they are exercised directly by the existing suite, so they stay as they
    are.)"""
    if not port:
        return None
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}{path}", timeout=timeout) as r:
            if r.status != 200:
                return None
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception:
        return None


def upstream_facts(port: int | None) -> dict:
    """The deployment facts the dashboard's Deployment panel shows. Refreshed
    by the forwarder's sampler thread, not on the request path.

    Everything the upstream exposes about itself, read from whichever endpoint
    it has: SGLang answers /get_server_info, llama-server answers /props. The
    engine is whichever of those returned 200; "unknown" when neither does.
    Values the engine does not publish read "unknown" rather than a guess --
    an em dash on the page beats a wrong answer."""
    if not port:
        return {"engine": "unknown", "thinking": "unknown",
                "speculative": "none", "parallel": "", "model_path": ""}
    info = _http_get_json(port, "/get_server_info")
    props = _http_get_json(port, "/props")

    engine = "unknown"
    if isinstance(info, dict):
        engine = "sglang"
    elif isinstance(props, dict):
        engine = "llama-server"

    facts = {"engine": engine}
    if engine == "sglang":
        kwargs = info.get("default_chat_template_kwargs") or {}
        on = kwargs.get("enable_thinking")
        facts["thinking"] = "on" if on is True else ("off" if on is False
                                                    else "unknown")
        spec = info.get("speculative_algorithm") or "none"
        facts["speculative"] = spec if spec else "none"
        tp = info.get("tp_size", 1)
        pp = info.get("pp_size", 1)
        facts["parallel"] = f"tp={tp} pp={pp}"
        facts["model_path"] = info.get("model_path", "") or ""
    elif engine == "llama-server":
        # llama-server does not publish speculative or parallel; only the
        # thinking flag, and only when it is present in the props.
        kw = props.get("chat_template_kwargs") or {}
        on = kw.get("enable_thinking")
        facts["thinking"] = "on" if on is True else ("off" if on is False
                                                     else "unknown")
        facts["speculative"] = "unknown"
        facts["parallel"] = ""
        facts["model_path"] = props.get("model", "") or ""
    else:
        facts["thinking"] = "unknown"
        facts["speculative"] = "none"
        facts["parallel"] = ""
        facts["model_path"] = ""
    return facts

#: Cumulative counters and gauges that add across lanes. Summed BEFORE the
#: page differences them, so the page's existing rate logic yields fleet
#: rates without knowing there is more than one lane.
_SUM_KEYS = (
    "conns", "requests", "status_2xx", "status_4xx", "status_5xx",
    "fallbacks", "unavailable", "tok_prompt", "tok_cached", "tok_gen",
    "tok_today_prompt", "tok_today_cached", "tok_today_gen",
    "gen_tokens", "gen_seconds", "prompt_tokens", "prompt_seconds",
    "prompt_cached", "decode_steps", "draft_total", "draft_accepted",
    "drafts", "processing", "deferred", "gen_throughput", "sglang_running",
    "sglang_queued", "sglang_prompt_tokens", "sglang_gen_tokens",
    "sglang_requests_total",
)
#: High-water marks and ratios: the largest lane speaks for the fleet.
_MAX_KEYS = ("ctx", "tokens_max", "sglang_context_len", "sglang_token_usage",
             "sglang_cache_hit_rate", "sglang_spec_accept_length",
             "latency_p50_ms")


def merge_snapshots(own: dict, peers: list) -> dict:
    """One snapshot for every lane: this lane's, with each reachable peer's
    folded in. Counters add, high-water marks take the max, per-stream rows
    concatenate with a lane tag and a lane-unique id, the model names join,
    healthy means any lane is. Everything about THIS lane -- its port,
    token, presets, assignment, peers, GPUs -- stays its own.

    The page differences cumulative counters between polls, so summing them
    first makes its throughput, prefill and request rates fleet totals with
    no change to that logic. One lane reloading (counters restart at zero)
    costs one odd tick, as it always did."""
    lanes = [own] + [p for p in peers if isinstance(p, dict)]
    out = dict(own)
    out["fleet"] = {"lanes": len(lanes),
                    "serving": sum(1 for l in lanes if l.get("healthy")),
                    "ports": [l.get("listen") for l in lanes]}
    if len(lanes) == 1:
        return out
    for k in _SUM_KEYS:
        out[k] = sum((l.get(k) or 0) for l in lanes)
    for k in _MAX_KEYS:
        vals = [l.get(k) for l in lanes if isinstance(l.get(k), (int, float))]
        out[k] = max(vals) if vals else own.get(k)
    slots = []
    for l in lanes:
        port = l.get("listen")
        for sl in l.get("slots") or []:
            row = dict(sl)
            row["lane"] = port
            row["slot"] = sl.get("id")
            # Unique across lanes: the page keys its rate history by id.
            row["id"] = f"{port}:{sl.get('id')}"
            slots.append(row)
    out["slots"] = slots
    recent = []
    for l in lanes:
        recent.extend(l.get("recent_streams") or [])
    recent.sort(key=lambda r: r.get("ended") or 0, reverse=True)
    out["recent_streams"] = recent[:12]
    rows = []
    for l in lanes:
        rows.extend(l.get("lane_rows") or [])
    out["lane_rows"] = rows
    out["healthy"] = any(bool(l.get("healthy")) for l in lanes)
    out["live"] = any(bool(l.get("live")) for l in lanes)
    out["metrics_available"] = any(bool(l.get("metrics_available")) for l in lanes)
    names = []
    for l in lanes:
        m = l.get("model")
        if m and m != "-" and m not in names:
            names.append(m)
    out["model"] = " + ".join(names) if names else "-"
    days: dict = {}
    for l in lanes:
        for d in l.get("days") or []:
            row = days.setdefault(d.get("day"), {"day": d.get("day"),
                                                 "prompt": 0, "cached": 0, "gen": 0})
            for k in ("prompt", "cached", "gen"):
                row[k] += d.get(k) or 0
    out["days"] = [days[k] for k in sorted(days)]
    return out


def _launch_failed(fwd) -> bool:
    """True when a preset is assigned, the upstream is not healthy, and what
    was launched is no longer running. Read-only: a container status the
    monitor last polled, and a poll() on the launcher's Popen."""
    if not getattr(fwd, "_preset", None) or getattr(fwd, "_upstream_healthy", False):
        return False
    if getattr(fwd, "_operator_stopped", False):
        return False
    if getattr(fwd, "CONTAINER_NAME", None):
        return getattr(fwd, "_container_status", None) == "exited"
    child = getattr(fwd, "_upstream_child", None)
    if child is None:
        return False
    try:
        return child.poll() is not None
    except Exception:
        return False


def snapshot(fwd, stats: dict) -> dict:
    """One JSON sample. Cumulative counters are returned raw -- the page
    differences successive samples for live rates, and differences a stored
    baseline for session figures. Both need the raw counter."""
    port = getattr(fwd, "_upstream", None)
    # Ask a dead upstream nothing. On this Windows a connect to a closed
    # loopback port times out instead of being refused, so an unloaded
    # lane paid ~4 s per poll here, and a peer's 2 s read of this snapshot
    # never succeeded: the Lanes panel showed every unloaded lane as
    # unreachable. The sampler's /health verdict gates both readers once it
    # has run; before its first pass (and in the tests) nothing changes.
    live = (not getattr(fwd, "_health_sampled", False)
            or getattr(fwd, "_upstream_healthy", False))
    m = upstream_metrics(port) if live else {}
    # The page's poll doubles as a sample for the token tally, so the Tokens
    # card is live while the dashboard is open. See forwarder._tally_tokens.
    tally = getattr(fwd, "_tally_tokens", None)
    if tally:
        tally(port, m)
    today = getattr(fwd, "_today_tokens", None)
    today_p, today_c, today_g = today() if today else (0, 0, 0)
    days_fn = getattr(fwd, "recent_days", None)
    lat = sorted(list(stats.get("latency", ()))[-200:])
    g = m.get
    slots = upstream_slots(port) if live else []
    # The keepalive child's pid when container mode is on, else None.
    # SGLang has no /slots and never will: it reports counts, not slots.
    # A lane running it therefore contributes ONE row to the per-stream
    # table -- what the engine itself reports about all its work -- instead
    # of disappearing from it. Never call these rows streams.
    lane_rows = []
    if any(k.startswith("sglang:") for k in m):
        lane_rows.append({
            "lane": getattr(fwd, "LISTEN_PORT", None),
            "engine": "sglang",
            "running": g("sglang:num_running_reqs", 0),
            "queued": g("sglang:num_queue_reqs", 0),
            "rate": g("sglang:gen_throughput", 0.0),
            "cached": g("sglang:cache_hit_rate", 0.0),
            "ctx": g("sglang:context_len", 0.0),
            "kv": g("sglang:token_usage", 0.0),
        })
    ensure_stream_watcher(fwd)
    _ka = getattr(fwd, "_container_keepalive", None)
    keepalive_pid = _ka.pid if _ka is not None else None
    return {
        # Context window: llama-server's n_ctx, the same on every slot.
        "ctx": max((s["n_ctx"] for s in slots), default=0),
        "t": time.time(),
        "uptime_s": int(time.time() - stats.get("started", _STARTED)),
        "upstream": port,
        # Which llama-server executable answered. Discovery cannot tell two
        # of them apart by port, so this is how a wrong one becomes visible.
        "upstream_exe": getattr(fwd, "_upstream_exe", None),
        # How the current upstream was found: executable-matched
        # ("llama-server"), a --candidate-port / derived container port
        # ("candidate"), or a pinned --upstream-port ("explicit"). null when
        # there is no upstream.
        "upstream_kind": getattr(fwd, "_upstream_kind", None),
        # The configured choice between the two kinds when both are healthy.
        "prefer": getattr(fwd, "PREFER", "llama-server"),
        "container": getattr(fwd, "_container_status", None),
        "container_name": getattr(fwd, "CONTAINER_NAME", None),
        "model": stats.get("model") or "-",
        "listen": getattr(fwd, "LISTEN_PORT", None),
        "live": bool(m),
        # /health (not /metrics) is what every engine answers, so this is the
        # honest "is the upstream up" light. SGLang has no /metrics, so the
        # old live==bool(m) read a healthy serving upstream as unreachable.
        "healthy": bool(getattr(fwd, "_upstream_healthy", False)),
        # True when the /metrics sample succeeded. When false, every card
        # that depends on /metrics or /slots must read "not provided by this
        # upstream" rather than "nothing finished yet".
        "metrics_available": bool(m),
        # The deployment facts the Deployment panel shows, refreshed by the
        # sampler thread; the request path only copies them.
        "facts": getattr(fwd, "_upstream_facts", {}) or {},
        # The keepalive child's pid when container mode is on, else null.
        "keepalive_pid": keepalive_pid,
        # --- lane identity: name this forwarder and link its peers ---
        "name": getattr(fwd, "FWD_NAME", None),
        # GPU index this lane runs on, from --gpu. A label: no server
        # exposes it, so the dashboard's GPU panel cannot infer it either.
        "gpu": getattr(fwd, "FWD_GPU", None),
        # Per-peer state the monitor thread last read, one dict per --peer.
        # The request path only copies the dict; it never re-probes.
        "peers": [dict(v) for v in getattr(fwd, "_peer_state", {}).values()],
        # GPU rows the monitor thread last read from nvidia-smi. Empty when
        # the binary is missing or failed; the panel says "no GPU data".
        "gpus": [dict(v) for v in getattr(fwd, "_gpu_state", [])],
        # The /__control token, so the dashboard can authorise its buttons.
        # See forwarder._control_token for why exposing it is safe.
        "control_token": getattr(fwd, "_control_token", None),
        # Whether a missing upstream falls back to Studio (off by default).
        "studio_fallback": bool(getattr(fwd, "STUDIO_FALLBACK", False)),
        # The operator pressed stop; auto-start stays off until start.
        "operator_stopped": bool(getattr(fwd, "_operator_stopped", False)),
        # PID behind a process upstream, from the netstat scan; null for a
        # container or when unknown. Read-only lookup, safe on this path.
        "upstream_pid": (fwd._upstream_pid()
                         if hasattr(fwd, "_upstream_pid") else None),
        # True when --upstream-cmd was given, so start/restart exist on a
        # process lane; without it only stop is offered.
        "upstream_cmd": bool(getattr(fwd, "UPSTREAM_CMD", None)),
        # Model presets this lane can assign to its GPU, and the current one.
        # "loading" is a preset assigned but not yet healthy and not stopped:
        # the page shows the wait instead of a red light that looks broken.
        "presets": sorted(getattr(fwd, "_presets", {}) or {}),
        "preset": getattr(fwd, "_preset", None),
        "loading": bool(getattr(fwd, "_preset", None)
                        and not getattr(fwd, "_upstream_healthy", False)
                        and not getattr(fwd, "_operator_stopped", False)
                        and not _launch_failed(fwd)),
        # The assigned server is gone: its container exited, or the process
        # the launcher started has already returned. Live 2026-09-05 20:52:
        # an SGLang preset died in seconds (no room for its KV cache beside
        # another model) and the page said "loading..." for as long as anyone
        # cared to watch. A failed launch must read as failed.
        "launch_failed": _launch_failed(fwd),
        # --- forwarder-only: /metrics has no error counter ---
        "conns": stats.get("conns", 0),
        "requests": stats.get("requests", 0),
        "status_2xx": stats.get("2xx", 0),
        "status_4xx": stats.get("4xx", 0),
        "status_5xx": stats.get("5xx", 0),
        "latency_p50_ms": round(lat[len(lat) // 2] * 1000, 1) if lat else None,
        "fallbacks": stats.get("fallbacks", 0),
        # Requests answered 503 because no llama-server was running.
        "unavailable": stats.get("unavailable", 0),
        # The forwarder's own tally: survives model reloads, which restart
        # every llama-server counter at zero.
        "tok_prompt": stats.get("tok_prompt", 0),
        "tok_cached": stats.get("tok_cached", 0),
        "tok_gen": stats.get("tok_gen", 0),
        # Today's totals across restarts, from the forwarder's tokens.json.
        "tok_today_prompt": today_p,
        "tok_today_cached": today_c,
        "tok_today_gen": today_g,
        # Newest first. Feeds the usage page's history and its totals.
        "days": days_fn(30) if days_fn else [],
        # --- cumulative, straight from llama-server ---
        "gen_tokens": g("llamacpp:tokens_predicted_total", 0),
        "gen_seconds": g("llamacpp:tokens_predicted_seconds_total", 0),
        "prompt_tokens": g("llamacpp:prompt_tokens_total", 0),
        "prompt_seconds": g("llamacpp:prompt_seconds_total", 0),
        "prompt_cached": g("llamacpp:prompt_tokens_cached_total", 0),
        # n_decode_total ticks per forward pass, so unlike tokens_predicted_*
        # (which only moves when a request COMPLETES) it gives a live rate
        # mid-generation. Without it the page reads "idle" during a long reply.
        "decode_steps": g("llamacpp:n_decode_total", 0),
        "draft_total": g("llamacpp:spec_decode_num_draft_tokens_total", 0),
        "draft_accepted": g("llamacpp:spec_decode_num_accepted_tokens_total", 0),
        "drafts": g("llamacpp:spec_decode_num_drafts_total", 0),
        "processing": g("llamacpp:requests_processing", 0),
        "deferred": g("llamacpp:requests_deferred", 0),
        "tokens_max": g("llamacpp:n_tokens_max", 0),
        # --- SGLang: the same cards, different metric names ---
        # sglang:gen_throughput is a live gauge, not a counter: the page
        # shows it directly rather than differencing it.
        "gen_throughput": g("sglang:gen_throughput", 0.0),
        # num_running_reqs plus num_queue_reqs, shown as "queued N".
        "sglang_running": g("sglang:num_running_reqs", 0),
        "sglang_queued": g("sglang:num_queue_reqs", 0),
        # 0..1, shown as a percent.
        "sglang_cache_hit_rate": g("sglang:cache_hit_rate", 0.0),
        # An accepted LENGTH, not a percent: the page shows "tau 3.2".
        "sglang_spec_accept_length": g("sglang:spec_accept_length", 0.0),
        # The server's own context window, shown on the Largest context card.
        "sglang_context_len": g("sglang:context_len", 0.0),
        # 0..1, KV pool occupancy, shown on the Server row.
        "sglang_token_usage": g("sglang:token_usage", 0.0),
        # Counters the forwarder's tally folds: cumulative, restart at zero
        # on a server restart. The tally treats a backwards counter as a new
        # process, so its value is added whole.
        "sglang_prompt_tokens": g("sglang:prompt_tokens_total", 0),
        "sglang_gen_tokens": g("sglang:generation_tokens_total", 0),
        # Informational: the server's own request count, not the forwarder's.
        "sglang_requests_total": g("sglang:num_requests_total", 0),
        # Per-stream, from /slots. /metrics cannot give this.
        "slots": slots,
        # Streams that finished, so the panel says something between bursts.
        "recent_streams": recent_streams(getattr(fwd, "LISTEN_PORT", None)),
        "lane_rows": lane_rows,
    }


#: The header both pages share: the title mark, the tab switch, the lane
#: name and the peer pills. Kept in one place so the two pages cannot drift
#: apart -- the moment a style change lands on one page only, switching
#: pages jumps, which is the exact bug this fragment exists to prevent.
_HEADER_CSS = """:root{
  --bg:#0b141a; --panel:#101e26; --panel2:#0d1a21; --line:#1d3440;
  --ink:#dcecf1; --dim:#7f9ba7; --teal:#5ed6cb; --amber:#f0b25e;
  --red:#e2685f; --green:#5cc98c;
  --mono:ui-monospace,"Cascadia Mono","SF Mono",Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
/* Reserve the scrollbar even when the page is short, so switching
   between the two pages cannot shift the layout sideways. */
html{scrollbar-gutter:stable}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",system-ui,sans-serif;
  padding:26px 24px 40px;-webkit-font-smoothing:antialiased}
.wrap{max-width:1060px;margin:0 auto}
h1{margin:0;font-size:25px;letter-spacing:-.02em;font-weight:650;
  display:flex;align-items:center;gap:11px}
.mark{width:22px;height:22px;flex:none}
.sub{margin:5px 0 20px;color:var(--dim);font-size:13px}
/* Two pages, one nav. It rides on the title line and both pages carry an
   identical header, so switching moves nothing on screen but the highlight. */
.tabs{display:flex;border:1px solid var(--line);border-radius:8px;
  overflow:hidden;background:var(--panel2);flex:none}
.tabs a{color:var(--dim);text-decoration:none;font-size:12px;padding:6px 15px;
  line-height:1.5}
.tabs a+a{border-left:1px solid var(--line)}
.tabs a:hover{color:var(--ink)}
.tabs a.on{color:var(--teal);background:var(--panel)}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:7px}
.dot.on{background:var(--green);box-shadow:0 0 0 3px rgba(92,201,140,.16)}
.dot.off{background:var(--red);box-shadow:0 0 0 3px rgba(226,104,95,.16)}
.fname{font-family:var(--mono);font-size:13px;color:var(--dim);
  margin-left:8px;font-weight:400}
.tabsrow{display:flex;align-items:center;gap:16px;margin-top:10px}
.tabsrow .peers{margin-left:auto}
/* Peer pills: status dot + name (or ":PORT"), then a muted engine suffix. */
.peers{display:flex;gap:6px;align-items:center}
.peers a.lane{font-family:var(--mono);font-size:11px;color:var(--dim);
  text-decoration:none;border:1px solid var(--line);border-radius:6px;
  padding:2px 8px;display:inline-flex;align-items:center;gap:6px}
.peers a.lane:hover{color:var(--teal);border-color:var(--teal)}
.peers .pdot{width:6px;height:6px;border-radius:50%;display:inline-block}
.peers .pdot.on{background:var(--green)}
.peers .pdot.off{background:var(--red)}
.peers .pfx{color:var(--dim);font-size:10px}
/* Lanes panel: one row per card, this lane first. */
.lanes{display:flex;flex-direction:column;gap:6px;margin-bottom:22px}
.recent{margin-top:14px;padding-top:12px;border-top:1px solid var(--line)}
.recent .rk{color:var(--dim);font-size:10px;letter-spacing:.08em;
  text-transform:uppercase;margin-bottom:6px}
.lane{display:grid;grid-template-columns:minmax(150px,1.1fr) 62px minmax(210px,1.5fr) minmax(0,2fr) auto minmax(80px,auto);
  gap:14px;align-items:center;padding:10px 14px;background:var(--panel);border:1px solid var(--line);
  border-radius:8px;font-family:var(--mono);font-size:12px}
.lane.self{border-color:var(--teal)}
.lane .ln b{color:var(--ink);font-weight:600}
.lane .ln a{color:var(--teal);text-decoration:none}
.lane .ln a:hover{text-decoration:underline}
.lane .lg{color:var(--dim)}
.lane .lp.ok{color:var(--green)} .lane .lp.warn{color:var(--amber)} .lane .lp.dim{color:var(--dim)} .lane .lp.bad{color:var(--red)}
.lane .lm{color:var(--dim);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.lane .ctl{justify-content:flex-end}
@media (max-width:900px){.lane{grid-template-columns:1fr 1fr;} .lane .lm{display:none}}
/* GPU panel: one row per card, the matching lane marked. */
.gpus{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  overflow:hidden}
.gpus table{width:100%;border-collapse:collapse}
.gpus th{font-size:9.5px;letter-spacing:.1em;color:var(--dim);
  text-transform:uppercase;font-weight:600;text-align:right;
  padding:10px 16px 8px;border-bottom:1px solid var(--line)}
.gpus th:first-child{text-align:left}
.gpus td{font-family:var(--mono);font-size:13px;text-align:right;
  padding:9px 16px;border-bottom:1px solid rgba(29,52,64,.5);
  font-variant-numeric:tabular-nums}
.gpus tr:last-child td{border-bottom:0}
.gpus td:first-child{text-align:left;color:var(--dim);font-size:12px}
/* The lane marker is plain text, never a glyph: an icon-font character
   renders as a tofu box where the font is missing. The marker rides the
   cell in the same small dim style as the other sub-labels. */
.gpus .gmark{color:var(--dim);font-size:10px;margin-left:8px}
.gpus .meter{height:4px;background:var(--panel2);border-radius:3px;
  overflow:hidden;border:1px solid var(--line);margin-top:0;width:90px;
  display:inline-block;vertical-align:middle;margin-left:8px}
.gpus .meter i{display:block;height:100%;background:var(--teal);width:0;
  transition:width .45s ease}
.quiet{padding:16px;color:var(--dim);font-size:12px}
"""

#: The header markup, one function because the two pages differ in only two
#: places: which tab carries the "on" class, and the page title. Everything
#: else -- the mark, the lane name, the peer pills, the tabs themselves --
#: is byte-identical on both pages.
_HEADER_HTML = """<div class="head">
<h1>
<svg class="mark" viewBox="0 0 24 24" aria-hidden="true">
  <rect x="1.3" y="1.3" width="21.4" height="21.4" rx="5.8" fill="#0f1e26" stroke="#1d3440"/>
  <path d="M5.6 12h12.2M13.3 7.8L18.1 12l-4.8 4.2" fill="none" stroke="#5ed6cb"
        stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
<span>omp forwarder</span><span class="fname" id="fname"></span></h1>
<div class="tabsrow">
<nav class="tabs">{tabs}</nav>
<div class="peers" id="peers"></div>
</div>
</div>"""

_SVG_SYMBOLS = """<svg style="display:none">
  <symbol id="ico-rst" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
    <path d="M20.5 12a8.5 8.5 0 1 1-2.6-6.1"/><path d="M20.6 4.2v5.2h-5.2"/>
  </symbol>
</svg>"""


def header(on: str) -> str:
    """The shared header: title, lane name, tab switch, peer pills.
    `on` is the tab that is lit on this page ("stats" or "usage")."""
    live = '<a class="on" href="/__stats">Live</a><a href="/__usage">Usage</a>'
    usage = '<a href="/__stats">Live</a><a class="on" href="/__usage">Usage</a>'
    tabs = live if on == "stats" else usage
    return _HEADER_HTML.format(tabs=tabs)


PAGE = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title id="pg_title">omp forwarder</title>
<style>
__HEADER_CSS__


/* The status bar layout belongs to this page only: the usage page uses
   .bar for its by-day rows, so these rules must not move into the
   shared header fragment. */
.bar{display:flex;flex-wrap:wrap;gap:24px;align-items:center;
  background:var(--panel);border:1px solid var(--line);border-radius:12px;
  padding:12px 18px;margin-bottom:16px}
.bit{display:flex;flex-direction:column;gap:1px;min-width:0}
/* .bit sets display, which beats the hidden attribute's default; without
   this the Container bit shows an empty label on a lane with no container. */
.bit[hidden]{display:none}
.bk{font-size:9.5px;letter-spacing:.1em;color:var(--dim);text-transform:uppercase}
.bv{font-family:var(--mono);font-size:13px;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap}
.grid{display:grid;gap:11px;grid-template-columns:repeat(auto-fit,minmax(180px,1fr))}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  padding:13px 16px 14px}
.k{font-size:9.5px;letter-spacing:.1em;color:var(--dim);text-transform:uppercase}
.v{font-family:var(--mono);font-size:26px;line-height:1.2;margin-top:6px;
  font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.u{font-size:12.5px;color:var(--dim);margin-left:3px}
.n{font-size:11px;color:var(--dim);margin-top:5px;min-height:15px}
.teal{color:var(--teal)} .amber{color:var(--amber)} .red{color:var(--red)}
.meter{height:4px;background:var(--panel2);border-radius:3px;margin-top:9px;
  overflow:hidden;border:1px solid var(--line)}
.meter i{display:block;height:100%;background:var(--teal);width:0;
  transition:width .45s ease}
.meter i.warn{background:var(--amber)} .meter i.bad{background:var(--red)}
h2{font-size:10px;letter-spacing:.1em;color:var(--dim);text-transform:uppercase;
  margin:24px 0 10px;font-weight:600;display:flex;align-items:center;gap:9px}
h2 .rule{flex:1;height:1px;background:var(--line)}
.tag{font-family:var(--mono);font-size:9.5px;letter-spacing:.04em;
  border:1px solid var(--line);border-radius:4px;padding:1px 5px;
  color:var(--dim);margin-left:7px;vertical-align:2px;white-space:nowrap}
/* Reset controls: quiet until hovered, so they never compete with the data. */
.rst{appearance:none;background:transparent;border:1px solid var(--line);
  border-radius:6px;color:var(--dim);cursor:pointer;padding:3px 7px 2px;
  display:inline-flex;align-items:center;gap:5px;font:inherit;font-size:9.5px;
  letter-spacing:.09em;text-transform:uppercase;transition:.15s}
.rst svg{width:11px;height:11px;display:block}
.rst:hover{color:var(--teal);border-color:var(--teal);background:rgba(94,214,203,.07)}
.rst.armed{color:var(--teal);border-color:var(--teal)}
.since{font-family:var(--mono);font-size:9.5px;color:var(--teal);
  letter-spacing:.02em;text-transform:none}
.stale{opacity:.4;transition:opacity .3s}
/* Per-stream rows. A table rather than cards: the point is comparing slots
   against each other, and columns do that better than tiles. */
.slots{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  overflow:hidden}
.slots table{width:100%;border-collapse:collapse}
.slots th{font-size:9.5px;letter-spacing:.1em;color:var(--dim);
  text-transform:uppercase;font-weight:600;text-align:right;
  padding:10px 16px 8px;border-bottom:1px solid var(--line)}
.slots th:first-child{text-align:left}
.slots td{font-family:var(--mono);font-size:14px;text-align:right;
  padding:9px 16px;border-bottom:1px solid rgba(29,52,64,.5);
  font-variant-numeric:tabular-nums}
.slots tr:last-child td{border-bottom:0}
.slots td:first-child{text-align:left;color:var(--dim);font-size:12px}
.slots .rate{color:var(--teal);font-size:17px}
.slots .dim{color:var(--dim)}
.slots tfoot td{border-top:1px solid var(--line);border-bottom:0;
  color:var(--ink)}
.quiet{padding:16px;color:var(--dim);font-size:12px}
.foot{margin-top:24px;padding-top:15px;border-top:1px solid var(--line);
  color:var(--dim);font-size:12px;line-height:1.65}
.foot b{color:var(--ink);font-weight:600}
.foot code{font-family:var(--mono);font-size:11px;color:var(--ink)}
/* Deployment panel: label/value pairs in one block. Unknowns read as an
   em dash (rendered by the JS), so a missing fact never reads as a guess. */
.facts{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  padding:13px 16px;display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
  gap:8px 24px}
.facts .row{display:flex;justify-content:space-between;gap:12px}
.facts .fk{font-size:9.5px;letter-spacing:.1em;color:var(--dim);
  text-transform:uppercase}
.facts .fv{font-family:var(--mono);font-size:13px;text-align:right;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.facts .fv.dim{color:var(--dim)}
/* The Upstream fact can read a whole sentence ("candidate :30000 - prefer
   llama-server"), so it spans the row instead of truncating in one cell. */
.facts .row.full{grid-column:1/-1}
/* Container controls: three quiet buttons next to the Container bit. */
.ctl{display:inline-flex;gap:5px;margin-left:10px}
.ctl button{appearance:none;background:transparent;border:1px solid var(--line);
  border-radius:6px;color:var(--dim);cursor:pointer;padding:2px 8px;
  font:inherit;font-size:10px;letter-spacing:.05em;transition:.15s}
.ctl button:hover{color:var(--teal);border-color:var(--teal);
  background:rgba(94,214,203,.07)}
.ctl button.danger:hover{color:var(--red);border-color:var(--red);
  background:rgba(226,104,95,.07)}
.ctl_msg{font-size:11px;color:var(--dim);margin-left:8px}
.pill{display:inline-block;font-family:var(--mono);font-size:11px;
  border:1px solid var(--line);background:transparent;border-radius:10px;
  padding:1px 7px;color:var(--dim);letter-spacing:.04em}
.pill.on{color:var(--green);border-color:var(--green)}
.pill.off{color:var(--red);border-color:var(--red)}
</style>
__SVG_SYMBOLS__
<div class="wrap">
__HEADER__
<div class="sub">Traffic omp sends straight to llama-server &mdash; the requests Unsloth&rsquo;s own API panel cannot see.</div>

<div class="bar">
  <div class="bit"><span class="bk">Listening</span><span class="bv" id="listen">&mdash;</span></div>
  <div class="bit"><span class="bk">Upstream</span><span class="bv" id="up">&mdash;</span></div>
  <div class="bit"><span class="bk">Status</span><span class="bv"><span class="dot off" id="dot"></span><span id="state">connecting</span></span></div>
  <div class="bit" style="flex:1"><span class="bk">Model</span><span class="bv" id="model">&mdash;</span></div>
  <div class="bit"><span class="bk">Uptime</span><span class="bv" id="uptime">&mdash;</span></div>
  <div class="bit" id="contbit" hidden><span class="bk">Container</span><span class="bv" id="cont" title="">&mdash;</span><span class="ctl"><button data-act="start">start</button><button data-act="stop">stop</button><button data-act="restart" class="danger">restart</button></span><span class="ctl_msg"></span></div>
  <div class="bit" id="procbit" hidden><span class="bk">Upstream process</span><span class="bv" id="proc" title="">&mdash;</span><span class="ctl"><button data-act="start" data-needs-cmd>start</button><button data-act="stop" class="danger">stop</button><button data-act="restart" class="danger" data-needs-cmd>restart</button></span><span class="ctl_msg"></span></div>
  <button class="rst" id="rst-all" title="Reset every section to start from now">
    <svg><use href="#ico-rst"/></svg>Reset all</button>
</div>

<h2>Lanes<span class="rule"></span></h2>
<div class="lanes" id="lanes"></div>

<h2>Deployment<span class="rule"></span></h2>
<div class="facts" id="facts">
  <div class="row"><span class="fk">Engine</span><span class="fv dim" id="f_engine">&mdash;</span></div>
  <div class="row"><span class="fk">Thinking</span><span class="fv" id="f_thinking">&mdash;</span></div>
  <div class="row"><span class="fk">Speculative</span><span class="fv dim" id="f_spec">&mdash;</span></div>
  <div class="row"><span class="fk">Parallel</span><span class="fv dim" id="f_par">&mdash;</span></div>
  <div class="row"><span class="fk">Model path</span><span class="fv dim" id="f_model" title="">&mdash;</span></div>
  <div class="row"><span class="fk">Keepalive PID</span><span class="fv dim" id="f_ka">&mdash;</span></div>
  <div class="row full"><span class="fk">Upstream</span><span class="fv dim" id="f_up" title="How the current upstream was found; the configured preference">&mdash;</span></div>
</div>

<h2 id="model_h">Model &mdash; live, from llama-server<span class="rule"></span>
  <span class="since" id="since-model"></span>
  <button class="rst" data-sect="model" title="Count from now"><svg><use href="#ico-rst"/></svg>Session</button></h2>
<div class="grid">
  <div class="card"><div class="k">Throughput</div>
    <div class="v"><span id="tput" class="teal">&mdash;</span><span class="u">tok/s</span></div>
    <div class="n" id="tput_n">&nbsp;</div></div>
  <div class="card"><div class="k">Decode</div>
    <div class="v"><span id="dec">&mdash;</span><span class="u">tok/s</span></div>
    <div class="n" id="dec_n">pure generation speed</div></div>
  <div class="card"><div class="k">Prefill</div>
    <div class="v"><span id="pre">&mdash;</span><span class="u">tok/s</span></div>
    <div class="n" id="pre_n">prompt evaluation</div></div>
  <div class="card"><div class="k">Draft acceptance</div>
    <div class="v"><span id="acc">&mdash;</span><span class="u" id="acc_u">%</span></div>
    <div class="meter"><i id="acc_m"></i></div>
    <div class="n" id="acc_n">MTP speculation</div></div>
</div>

<h2>Server<span class="rule"></span>
  <span class="since" id="since-server"></span>
  <button class="rst" data-sect="server" title="Count from now"><svg><use href="#ico-rst"/></svg>Session</button></h2>
<div class="grid">
  <div class="card"><div class="k">In flight</div>
    <div class="v" id="inflight">&mdash;</div><div class="n" id="inflight_n">&nbsp;</div></div>
  <div class="card"><div class="k">Prompt cache</div>
    <div class="v"><span id="hit">&mdash;</span><span class="u">%</span></div>
    <div class="meter"><i id="hit_m"></i></div>
    <div class="n" id="hit_n">prefix reused</div></div>
  <div class="card"><div class="k">Tokens per pass</div>
    <div class="v" id="tpp">&mdash;</div>
    <div class="n" id="tpp_n">speculation payoff</div></div>
  <div class="card"><div class="k">Largest context</div>
    <div class="v" id="tmax">&mdash;</div><div class="n" id="tmax_n">tokens, high water</div></div>
</div>

<h2>GPUs &mdash; nvidia-smi<span class="rule"></span><span class="since" id="gpus_n"></span></h2>
<div id="gpus" class="gpus"><div class="quiet">no GPU data</div></div>

<h2>Per stream &mdash; <span id="slots_src">from /slots</span><span class="rule"></span>
  <span class="since" id="slots_n"></span></h2>
<div id="slots" class="slots"><div class="quiet">no stream is generating</div></div>

<h2>Forwarder &mdash; what only it can see<span class="rule"></span>
  <span class="since" id="since-fwd"></span>
  <button class="rst" data-sect="fwd" title="Count from now"><svg><use href="#ico-rst"/></svg>Session</button></h2>
<div class="grid">
  <div class="card"><div class="k">Tokens</div>
    <div class="v" id="tok">&mdash;</div><div class="n" id="tok_n">prompt + generated</div></div>
  <div class="card"><div class="k">Requests</div>
    <div class="v" id="req">&mdash;</div><div class="n" id="conn_n">&nbsp;</div></div>
  <div class="card"><div class="k">Round trip</div>
    <div class="v"><span id="p50">&mdash;</span><span class="u" id="p50u">ms</span></div>
    <div class="n">median, last 200</div></div>
  <div class="card"><div class="k">Server errors</div>
    <div class="v" id="e5">&mdash;</div><div class="n" id="e5_n">HTTP 5xx</div></div>
  <div class="card"><div class="k">Client errors</div>
    <div class="v" id="e4">&mdash;</div><div class="n">HTTP 4xx</div></div>
</div>

<div class="foot">
  <b>Why these and not Studio&rsquo;s numbers.</b> Prefill rate, draft acceptance
  and prompt-cache hit are all absent from the Unsloth API panel, and they are
  the ones that find real faults. Acceptance collapsing toward 35% means
  speculation fell back to n&#8209;gram and throughput has roughly halved. A
  falling cache-hit rate means prompts stopped being stable, which costs far
  more than any decode tuning. Errors are counted here because
  llama&#8209;server&rsquo;s <code>/metrics</code> has no error counter, so a
  chat&#8209;template 500 is otherwise completely silent.<br>
  <b>Session vs lifetime.</b> Every counter above is cumulative since the server
  started. <b>Session</b> stores a baseline in this browser and subtracts it, so
  the numbers start from when you clicked; the reset is per-browser and changes
  nothing on the server. A <span class="tag">lifetime</span> tag means nothing
  finished in the last few seconds, so that figure is a running average rather
  than a live rate. Round trip is already a rolling median, and largest context
  is a high&#8209;water mark, so neither is baselined.<br>
  <b>Tokens</b> is the forwarder&rsquo;s own tally, folded from successive
  <code>/metrics</code> samples, so it survives model reloads &mdash; every
  llama&#8209;server counter restarts at zero on one. It counts only traffic
  since the forwarder started; a reload loses at most the few seconds since the
  last sample.
</div>
</div>
<script>
const $=i=>document.getElementById(i);
let prev=null, missed=0;
const fmt=(n,d=1)=>(n==null||!isFinite(n))?"—":(n>=1000?Math.round(n).toLocaleString():n.toFixed(d));
const dur=s=>{const h=Math.floor(s/3600),m=Math.floor(s%3600/60);
  return h?(h+"h "+m+"m"):(m?(m+"m "+(s%60)+"s"):(s+"s"));};
const LIFE='<span class="tag">lifetime</span>';

// Which cumulative fields each section's baseline applies to. Anything not
// listed is already a live or rolling figure and must NOT be baselined.
const KEYS={
  model:["gen_tokens","gen_seconds","prompt_tokens","prompt_seconds",
         "draft_total","draft_accepted","drafts","decode_steps"],
  server:["prompt_cached","prompt_tokens","gen_tokens","decode_steps"],
  fwd:["requests","conns","status_2xx","status_4xx","status_5xx","fallbacks",
       "tok_prompt","tok_gen"]
};
const STORE="ompfwd.baseline.v1";
let base={model:null,server:null,fwd:null};
try{ const raw=localStorage.getItem(STORE); if(raw) base=Object.assign(base,JSON.parse(raw)); }
catch(e){ /* private window, or site data blocked -- run without a baseline */ }
function persist(){ try{ localStorage.setItem(STORE,JSON.stringify(base)); }catch(e){} }

function adj(s,sect){
  const b=base[sect];
  if(!b) return s;
  const o=Object.assign({},s);
  for(const k of KEYS[sect]){
    if(typeof s[k]==="number" && typeof b[k]==="number") o[k]=Math.max(0,s[k]-b[k]);
  }
  return o;
}
function stampSince(){
  for(const sect of ["model","server","fwd"]){
    const el=$("since-"+sect), b=base[sect];
    el.textContent = b ? ("since "+new Date(b.t*1000).toLocaleTimeString()) : "";
  }
  document.querySelectorAll(".rst[data-sect]").forEach(btn=>{
    btn.classList.toggle("armed", !!base[btn.dataset.sect]);
  });
}
function resetSect(sect){
  if(!prev) return;
  // Clicking an already-armed section clears it back to lifetime, so the
  // control is a toggle rather than a one-way door.
  base[sect] = base[sect] ? null : Object.assign({},prev);
  persist(); stampSince(); tick();
}
document.querySelectorAll(".rst[data-sect]").forEach(btn=>{
  btn.addEventListener("click",()=>resetSect(btn.dataset.sect));
});
$("rst-all").addEventListener("click",()=>{
  if(!prev) return;
  const any = base.model||base.server||base.fwd;
  const snap = any ? null : Object.assign({},prev);
  base={model:snap&&Object.assign({},snap), server:snap&&Object.assign({},snap),
        fwd:snap&&Object.assign({},snap)};
  persist(); stampSince(); tick();
});
stampSince();

async function tick(){
  let s;
  try{ s=await (await fetch("/__stats.json",{cache:"no-store"})).json(); }
  catch(e){ if(++missed>2) document.body.classList.add("stale"); return; }
  missed=0; document.body.classList.remove("stale");
  const M=adj(s,"model"), S=adj(s,"server"), F=adj(s,"fwd");
  const noMetrics = !s.metrics_available;

  // --- engine-aware section titles and subtitle ---
  const engine = (s.facts||{}).engine || "unknown";
  const engineSuffix = engine==="llama-server" ? "llama-server"
                      : engine==="sglang"       ? "SGLang" : "upstream";
  const modelH=$("model_h");
  modelH.childNodes[0].textContent =
    "Model — live, from "+engineSuffix+" ";
  const subEl=document.querySelector(".sub");
  subEl.innerHTML = "Traffic omp sends straight to "+engineSuffix
    + " &mdash; the requests Unsloth&rsquo;s own API panel cannot see.";
  const slotsSrc=$("slots_src");
  slotsSrc.textContent = engine==="llama-server" ? "from /slots"
    : "not provided by this upstream";

  // --- bar ---
  const nl=(s.fleet&&s.fleet.lanes)||1;
  // One page for every lane: the header names the fleet, the Lanes panel
  // below names each card. A lone lane reads as before.
  $("listen").textContent=nl>1?("fleet \u00b7 "+nl+" lanes"):("127.0.0.1:"+s.listen);
  _ctrlToken=s.control_token||"";
  // No upstream means 503 with Retry-After, not a silent hop to Studio,
  // unless --studio-fallback was given: say which.
  $("up").textContent=nl>1?(s.fleet.serving+" of "+nl+" serving"):(s.upstream?("127.0.0.1:"+s.upstream):(s.studio_fallback?"Studio :8888 (fallback)":"none"));
  $("up").title = s.upstream_exe || "";
  $("model").textContent=s.model; $("uptime").textContent=dur(s.uptime_s);
  $("dot").className="dot "+(s.healthy?"on":"off");
  $("state").textContent=s.healthy?"ready":(s.upstream?"unreachable":"no server");
  // The whole Container bit, label included, exists only in container mode.
  // A lane fronting a plain llama-server showed an empty "CONTAINER" label.
  $("contbit").hidden=!s.container_name;
  const latched=s.operator_stopped?" · stopped by operator":"";
  if(s.container_name){
    $("cont").textContent=s.container_name+" · "+(s.container||"unknown")+latched;
    $("cont").title="container "+s.container_name+" in WSL distro; status: "+(s.container||"unknown");
  } else {
    $("cont").textContent=""; $("cont").title="";
  }
  // Process lane: shown when the scan knows the upstream's PID or a start
  // command exists, and never beside a container. start/restart need the
  // command; stop needs only the PID.
  const proc=!s.container_name && (s.upstream_pid || s.upstream_cmd);
  $("procbit").hidden=!proc;
  if(proc){
    $("proc").textContent=(s.upstream_pid?("pid "+s.upstream_pid):"not running")+latched;
    $("proc").title=s.upstream_cmd?"start runs --upstream-cmd on the host":"no --upstream-cmd: stop only";
    document.querySelectorAll("#procbit [data-needs-cmd]").forEach(b=>{ b.hidden=!s.upstream_cmd; });
  }

  // --- forwarder-only cards: always live ---
  const tok=F.tok_prompt+F.tok_gen;
  $("tok").textContent=Math.round(tok).toLocaleString();
  let tokNote = tok>0
    ? Math.round(F.tok_prompt).toLocaleString()+" prompt · "+Math.round(F.tok_gen).toLocaleString()+" generated"
    : "prompt + generated";
  const today=(s.tok_today_prompt||0)+(s.tok_today_gen||0);
  if(today>0 && Math.round(today)!==Math.round(s.tok_prompt+s.tok_gen))
    tokNote += " · today "+Math.round(today).toLocaleString();
  $("tok_n").textContent = tokNote;
  $("req").textContent=F.requests;
  $("conn_n").textContent=F.conns+" connections"
    +(F.fallbacks?(" · "+F.fallbacks+" via Studio"):"")
    +(F.unavailable?(" · "+F.unavailable+" no model"):"");
  $("e5").textContent=F.status_5xx; $("e4").textContent=F.status_4xx;
  $("e5").className="v "+(F.status_5xx>0?"red":"");
  const tot=F.status_2xx+F.status_4xx+F.status_5xx;
  $("e5_n").textContent=tot?("HTTP 5xx · "+(100*F.status_5xx/tot).toFixed(1)+"% of replies"):"HTTP 5xx";
  const ms=s.latency_p50_ms;
  if(ms==null){ $("p50").textContent="—"; $("p50u").textContent="ms"; }
  else if(ms>=1000){ $("p50").textContent=(ms/1000).toFixed(1); $("p50u").textContent="s"; }
  else { $("p50").textContent=Math.round(ms); $("p50u").textContent="ms"; }

  // --- model section: throughput, decode, prefill, draft ---
  const NP = "not provided by this upstream";
  if(engine==="sglang" && s.metrics_available){
    // SGLang exposes different metric names; map onto the same cards.
    // Throughput is a live gauge (tok/s), not a counter to difference.
    // Mixed fleet seen from an SGLang lane: gen_throughput is already the
    // sum of every SGLang lane, and renderSlots returns the llama half.
    const mix=renderSlots(s);
    const both=(s.gen_throughput||0)+(mix.haveRate?mix.total:0);
    $("tput").textContent=fmt(both);
    $("tput_n").textContent=mix.haveRate?"SGLang + per-stream, all lanes":"live, from SGLang";
    // Decode and Prefill are not separately exposed by SGLang.
    $("dec").textContent="—"; $("dec_n").textContent=NP;
    $("pre").textContent="—"; $("pre_n").textContent=NP;
    // Draft acceptance: SGLang reports a length, not a percent.
    // Show it as "tau 3.2" without the % unit; the unit follows the
    // value's meaning, so it is dropped here and restored for
    // llama-server percentages below.
    if(s.sglang_spec_accept_length>0){
      $("acc").textContent="tau "+s.sglang_spec_accept_length.toFixed(1);
      $("acc_u").textContent="";
      meterSet($("acc_m"),0,50,40);
      $("acc_n").textContent="spec accept length";
    } else { $("acc").textContent="—"; $("acc_u").textContent="";
      $("acc_n").textContent=NP; }
  } else if(noMetrics){
    $("tput").textContent="—"; $("tput_n").textContent=NP;
    $("dec").textContent="—";  $("dec_n").textContent=NP;
    $("pre").textContent="—";  $("pre_n").textContent=NP;
    $("acc").textContent="—";  $("acc_u").textContent=""; $("acc_n").textContent=NP;
    meterSet($("acc_m"),0,50,40);
  } else if(prev){
    const dg=s.gen_tokens-prev.gen_tokens, dsec=s.gen_seconds-prev.gen_seconds;
    const dp=s.prompt_tokens-prev.prompt_tokens, dps=s.prompt_seconds-prev.prompt_seconds;
    const dd=s.draft_total-prev.draft_total, da=s.draft_accepted-prev.draft_accepted;
    const dsteps=s.decode_steps-prev.decode_steps, dw=s.t-prev.t;
    const live=renderSlots(s);

    // Throughput: prefer per-stream decode rates from /slots; fall back to
    // the completion counter when /slots is unavailable.
    if((s.slots||[]).length || (s.lane_rows||[]).length){
      if(live.decoding>0 && live.haveRate){
        $("tput").textContent=fmt(live.total);
        $("tput_n").textContent = live.decoding+(live.decoding>1?" streams":" stream")+" decoding"
          +(live.prefilling?(" · "+live.prefilling+" prefilling"):""); }
      else if(live.decoding>0){ $("tput").textContent="…"; $("tput_n").textContent="settling"; }
      else if(live.prefilling>0){ $("tput").textContent="0";
        $("tput_n").textContent=live.prefilling+(live.prefilling>1?" streams":" stream")+" prefilling"; }
      else { $("tput").textContent="0"; $("tput_n").textContent="idle"; }
    }
    else if(dg>0){ $("tput").textContent=fmt(dg/dw);
      $("tput_n").textContent=Math.round(dg)+" tokens in "+dw.toFixed(0)+"s"; }
    else if(dsteps>0 && s.decode_steps>0){
      const tpp=s.gen_tokens/s.decode_steps;
      $("tput").textContent=fmt(dsteps*tpp/dw);
      $("tput_n").innerHTML="estimated, mid&#8209;request"; }
    else { $("tput").textContent="0"; $("tput_n").textContent="idle"; }

    const sess = !!base.model;
    if(dsec>0){ $("dec").textContent=fmt(dg/dsec); $("dec_n").textContent="pure generation speed"; }
    else if(M.gen_seconds>0){ $("dec").textContent=fmt(M.gen_tokens/M.gen_seconds);
      $("dec_n").innerHTML="pure generation speed"+(sess?'<span class="tag">session</span>':LIFE); }
    else { $("dec").textContent="—"; $("dec_n").textContent="nothing finished yet"; }

    if(dps>0){ $("pre").textContent=fmt(dp/dps,0);
      $("pre_n").textContent=Math.round(dp).toLocaleString()+" prompt tokens"; }
    else if(M.prompt_seconds>0){ $("pre").textContent=fmt(M.prompt_tokens/M.prompt_seconds,0);
      $("pre_n").innerHTML="prompt evaluation"+(sess?'<span class="tag">session</span>':LIFE); }
    else { $("pre").textContent="—"; $("pre_n").textContent="nothing finished yet"; }

    let a=null, avg=false;
    if(dd>0){ a=100*da/dd; }
    else if(M.draft_total>0){ a=100*M.draft_accepted/M.draft_total; avg=true; }
    if(a!=null){ $("acc").textContent=a.toFixed(1); $("acc_u").textContent="%";
      meterSet($("acc_m"),a,50,40);
      $("acc").className = a<40?"red":(a<50?"amber":"teal");
      $("acc_n").innerHTML = (a<40 ? "LOW — check for n-gram fallback" : "MTP speculation")
        + (avg ? (sess?'<span class="tag">session</span>':LIFE) : ""); }
    else { $("acc").textContent="—"; $("acc_u").textContent="";
      $("acc_n").textContent="no drafts yet"; }
  }

  // --- server section: in flight, prompt cache, tokens per pass, largest ctx ---
  if(engine==="sglang" && s.metrics_available){
    // SGLang maps onto the same cards, different numbers.
    // In flight: running + queued requests, from SGLang's own counters.
    $("inflight").textContent=s.sglang_running;
    $("inflight_n").textContent = s.sglang_queued>0
      ? (s.sglang_queued+" queued")
      : (s.sglang_running>0?"busy":"idle");
    // Prompt cache: SGLang's cache_hit_rate is a 0..1 ratio, shown as %.
    const ch=100*(s.sglang_cache_hit_rate||0);
    $("hit").textContent=ch.toFixed(1);
    meterSet($("hit_m"),ch,60,35);
    $("hit_n").textContent="prefix reused";
    // Tokens per pass: not split per pass by SGLang; show the KV pool.
    $("tpp").textContent="—"; $("tpp_n").textContent=NP;
    // Largest context: SGLang's context_len, its own window, not a high-water.
    $("tmax").textContent=Math.round(s.sglang_context_len).toLocaleString();
    $("tmax_n").textContent = s.sglang_context_len>0
      ? "window · KV "+Math.round(100*(s.sglang_token_usage||0))+"%"
      : "tokens, high water";
  } else if(noMetrics){
    $("inflight").textContent="—"; $("inflight_n").textContent=NP;
    $("hit").textContent="—";  $("hit_n").textContent=NP;
    meterSet($("hit_m"),0,60,35);
    $("tpp").textContent="—";  $("tpp_n").textContent=NP;
    $("tmax").textContent="—"; $("tmax_n").textContent=NP;
  } else {
    $("inflight").textContent=s.processing;
    $("inflight_n").textContent=s.deferred>0?(s.deferred+" queued"):(s.processing>0?"busy":"idle");
    $("tmax").textContent=Math.round(s.tokens_max).toLocaleString();
    $("tmax_n").textContent = s.ctx>0
      ? "high water · window "+Math.round(s.ctx).toLocaleString()
      : "tokens, high water";

    const seen=S.prompt_cached+S.prompt_tokens;
    if(seen>0){ const h=100*S.prompt_cached/seen;
      $("hit").textContent=h.toFixed(1); meterSet($("hit_m"),h,60,35);
      $("hit_n").textContent=Math.round(S.prompt_cached).toLocaleString()+" tokens reused"; }
    else { $("hit").textContent="—"; $("hit_n").textContent="no prompts yet"; }

    if(S.decode_steps>0){ const tpp=S.gen_tokens/S.decode_steps;
      $("tpp").textContent=tpp.toFixed(2);
      $("tpp_n").textContent = tpp<1.15 ? "speculation buying ~nothing"
                                        : "vs 1.00 without speculation"; }
    else { $("tpp").textContent="—"; $("tpp_n").textContent="no passes yet"; }
  }

  // --- GPU panel: one row per card, this lane's card marked ---
  const gbox=document.getElementById("gpus");
  const gnote=document.getElementById("gpus_n");
  if(s.gpus && s.gpus.length){
    const thisGpu=s.gpu;
    // The lane marker is plain text: an icon-font glyph drew as a tofu box.
    const lane=g=>{
      if(g.index===thisGpu) return "this lane";
      return (s.peers||[]).filter(pp=>pp.gpu===g.index)
        .map(pp=>pp.name||("lane :"+pp.port)).join(" · ");
    };
    const rows=s.gpus.map(g=>{
      const mem=(g.mem_used_mib/1024).toFixed(1)+"/"+(g.mem_total_mib/1024).toFixed(1)+" GiB";
      const utilPct=Math.min(100,g.util_pct);
      const mark=lane(g);
      return `<tr><td>gpu ${g.index} · ${g.name}`+(mark?` <span class="gmark">${mark}</span>`:"")
        +`</td><td>${mem}</td>`
        +`<td>${g.util_pct}%<span class="meter"><i style="width:${utilPct}%"></i></span></td></tr>`;
    });
    gbox.innerHTML="<table><thead><tr><th>card</th><th>memory</th><th>util</th></tr></thead><tbody>"
      +rows.join("")+"</tbody></table>";
    gnote.textContent=s.gpus.length+" card"+(s.gpus.length>1?"s":"");
  } else {
    gbox.innerHTML='<div class="quiet">no GPU data</div>';
    gnote.textContent="";
  }


  // --- per-stream table: render or show not-provided ---
  if(noMetrics){
    const box=document.getElementById("slots");
    box.innerHTML='<div class="quiet">not provided by this upstream</div>';
    document.getElementById("slots_n").textContent="";
  } else if(!prev){
    renderSlots(s);
  }

  // --- lane identity: name, peers, title ---
  if(s.name){ $("fname").textContent="· "+s.name;
    $("pg_title").textContent="omp forwarder · "+s.name; }
  else { $("fname").textContent="";
    $("pg_title").textContent="omp forwarder"; }
  const pb=$("peers");
  if(s.peers && s.peers.length){
    pb.innerHTML=s.peers.map(pp=>{
      const name=pp.name?("lane "+pp.name):("lane :"+pp.port);
      const fx=pp.reachable
        ? ((pp.engine&&pp.engine!=="unknown"?" "+pp.engine:"")
           +((pp.thinking&&pp.thinking!=="unknown")?(" · "+pp.thinking):""))
        : " unreachable";
      const dot=pp.healthy?"on":"off";
      return `<a class="lane" href="http://127.0.0.1:${pp.port}/__stats" target="_blank">`
        +`<span class="pdot ${dot}"></span><span>${name}</span>`
        +`<span class="pfx">${fx}</span></a>`;
    }).join("");
  } else { pb.innerHTML=""; }


  // --- deployment facts ---
  const F2=s.facts||{};
  const setF=(id,v)=>{ const el=$(id);
    if(v && v!=="unknown" && v!=="none"){ el.textContent=v; el.classList.remove("dim"); }
    else { el.innerHTML=(v==="none")?"none":"&mdash;"; el.classList.add("dim"); } };
  setF("f_engine", F2.engine);
  // Thinking renders as a pill so on/off reads at a glance.
  const thEl=$("f_thinking");
  if(F2.thinking==="on"||F2.thinking==="off"){
    thEl.className="fv pill "+F2.thinking;
    thEl.textContent=F2.thinking;
  } else {
    thEl.className="fv dim";
    thEl.innerHTML=F2.thinking==="none"?"none":"&mdash;";
  }
  setF("f_spec", F2.speculative==="none"?"none":(F2.speculative||"unknown"));
  setF("f_par", F2.parallel);
  // Lanes: this lane first, then every --peer, one row per card with the
  // same buttons. A peer's button posts to THIS forwarder with lane=<port>;
  // the forwarder relays it with the peer's token, which never reaches the
  // page. Rows are rebuilt only when the set of lanes or presets changes,
  // so a click never lands on a freshly re-rendered button.
  const esc=t=>String(t==null?"":t).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
  const lanesEl=$("lanes");
  const me={port:String(location.port||"80"),name:s.name,gpu:s.gpu,preset:s.preset,loading:s.loading,
    operator_stopped:s.operator_stopped,healthy:s.healthy,model:s.model,presets:s.presets||[],reachable:true,self:true};
  const lanes=[me].concat((s.peers||[]).map(pp=>Object.assign({},pp,{port:String(pp.port)})));
  const lkey=lanes.map(l=>l.port+":"+(l.presets||[]).join(",")).join("|");
  if(lanesEl.dataset.built!==lkey){
    lanesEl.innerHTML=lanes.map(l=>{
      const ln=l.self?"":' data-lane="'+esc(l.port)+'"';
      const btns=(l.presets||[]).map(p=>'<button data-act="assign" data-preset="'+esc(p)+'"'+ln+'>'+esc(p)+'</button>').join("")
        +'<button data-act="stop" class="danger"'+ln+'>unload</button>';
      return '<div class="lane'+(l.self?" self":"")+'" data-port="'+esc(l.port)+'"><span class="ln"></span><span class="lg"></span>'
        +'<span class="lp"></span><span class="lm"></span><span class="ctl">'+btns+'</span><span class="ctl_msg"></span></div>';
    }).join("");
    lanesEl.dataset.built=lkey; wireCtl(lanesEl);
  }
  lanes.forEach(l=>{
    const row=lanesEl.querySelector('.lane[data-port="'+l.port+'"]'); if(!row) return;
    const nm=esc(l.name||("lane :"+l.port));
    row.querySelector(".ln").innerHTML=(l.self?"<b>"+nm+"</b> <span class=\"pfx\">this page</span>"
      :'<a href="http://127.0.0.1:'+esc(l.port)+'/__stats">'+nm+'</a>')+' <span class="pfx">:'+esc(l.port)+'</span>';
    row.querySelector(".lg").textContent=(l.gpu===null||l.gpu===undefined)?"no gpu":("GPU "+l.gpu);
    const st=!l.reachable?"unreachable":(l.preset?(l.loading?"loading\u2026":(l.operator_stopped?"unloaded":(l.healthy?"serving":"failed \u00b7 not running"))):"none assigned");
    const lp=row.querySelector(".lp"); lp.textContent=(l.preset||"\u2014")+" \u00b7 "+st;
    lp.className="lp "+(st==="serving"?"ok":(st==="loading\u2026"?"warn":(st==="unreachable"||st.startsWith("failed")?"bad":"dim")));
    row.querySelector(".lm").textContent=l.model||"";
    row.querySelectorAll("button").forEach(b=>{
      b.disabled=!!l.loading||!l.reachable;
      if(b.dataset.act==="assign") b.classList.toggle("on", b.dataset.preset===l.preset && !l.operator_stopped && !!l.healthy);
    });
  });
  // The Model row that used to sit here moved into the Lanes panel above:
  // its first row is this lane, same state, same buttons. One place to press.
  setF("f_model", F2.model_path);
  const kaEl=$("f_ka");
  if(s.keepalive_pid){ kaEl.textContent=s.keepalive_pid; kaEl.classList.remove("dim"); }
  else { kaEl.innerHTML="&mdash;"; kaEl.classList.add("dim"); }
  // One line for both: the kind that was chosen and the configured
  // preference, so a candidate upstream next to "prefer llama-server"
  // reads as "the preferred kind was down, this is the fallback".
  const upEl=$("f_up");
  if(s.upstream_kind && s.upstream){
    upEl.textContent=s.upstream_kind+" :"+s.upstream+" · prefer "+(s.prefer||"llama-server");
    upEl.title="chosen: "+s.upstream_kind+" on port "+s.upstream
      +"; configured preference: "+(s.prefer||"llama-server");
    upEl.classList.remove("dim");
  } else { upEl.innerHTML="&mdash;"; upEl.classList.add("dim"); }

  // --- control buttons: only visible when container mode is on ---
  const ctl=$("ctl");
  if(s.container_name && s.control_token){
    ctl.style.display="";
  } else { ctl.style.display="none"; }

  // --- dim metric cards when /metrics is unavailable ---
  if(!s.metrics_available){
    document.querySelectorAll(".grid .v").forEach(el=>el.classList.add("dim"));
  } else {
    document.querySelectorAll(".grid .v").forEach(el=>el.classList.remove("dim"));
  }
  prev=s;
}
function meterSet(el,pct,warn,bad){el.style.width=Math.max(0,Math.min(100,pct))+"%";
  el.className = pct<bad?"bad":(pct<warn?"warn":"");}

// Per-slot rates need their own history: n_decoded restarts whenever a slot
// takes a new request, so a delta is only valid while id_task is unchanged.
const slotHist=new Map();
// Returns {decoding, prefilling, total, haveRate} so the Throughput card can
// use the same per-stream rates instead of the completion-based counter.
// Streams that have finished. A slot is busy only while it generates and
// the page polls every 3 s, so short agent turns leave no live trace; this
// is what shows the panel is working between bursts.
function recentHtml(s){
  const r=s.recent_streams||[];
  if(!r.length) return "";
  const ago=t=>{const d=Math.max(0,Math.round(s.t-t));
    return d<60?(d+"s ago"):(Math.round(d/60)+"m ago");};
  return '<div class="recent"><div class="rk">recent</div><table><thead><tr>'
    +"<th>stream</th><th>finished</th><th>tok/s</th><th>generated</th>"
    +"<th>context</th><th>cached</th><th>took</th></tr></thead><tbody>"
    +r.map(x=>"<tr>"
      +`<td>${x.lane!=null?('<span class="dim">:'+x.lane+' \u00b7 </span>'):""}slot ${x.slot}</td>`
      +`<td class="dim">${ago(x.ended)}</td>`
      +`<td class="rate">${x.rate==null?'<span class="dim">\u2014</span>':fmt(x.rate)}</td>`
      +`<td>${Math.round(x.tokens).toLocaleString()}</td>`
      +`<td>${Math.round(x.prompt).toLocaleString()}</td>`
      +`<td>${x.prompt>0?Math.round(100*x.cached/x.prompt)+"%":'<span class="dim">\u2014</span>'}</td>`
      +`<td class="dim">${x.seconds==null?"":(x.seconds+"s")}</td>`
      +"</tr>").join("")
    +"</tbody></table></div>";
}
function renderSlots(s){
  const rows=[], slots=(s.slots||[]).filter(x=>x.busy);
  let total=0, haveRate=false, decoding=0, prefilling=0;
  for(const sl of slots){
    const h=slotHist.get(sl.id);
    // Also require the phase to be unchanged. A window that straddles the
    // prefill -> decode boundary is mostly prefill, so it reports a decode
    // rate of ~1 tok/s on a long prompt, which is true of the window and
    // badly misleading about the stream.
    const wasDecoding = h && h.decoded>0;
    const same = h && h.task===sl.task && s.t>h.t
                 && wasDecoding===(sl.decoded>0);
    // decoded==0 means the slot is still evaluating its prompt, not idle.
    // Reporting "0 tok/s" there would be a lie; its prompt is growing, and
    // that growth rate IS its prefill speed.
    const phase = sl.decoded>0 ? "decode" : "prefill";
    if(phase==="decode") decoding++; else prefilling++;
    let rate=null;
    if(same){
      rate = phase==="decode" ? (sl.decoded-h.decoded)/(s.t-h.t)
                              : (sl.prompt-h.prompt)/(s.t-h.t);
      if(!(rate>0)) rate=null;
      if(rate!=null && phase==="decode"){ total+=rate; haveRate=true; }
    }
    slotHist.set(sl.id,{task:sl.task,decoded:sl.decoded,prompt:sl.prompt,t:s.t});
    const hit = sl.prompt>0 ? (100*sl.cached/sl.prompt) : null;
    rows.push("<tr>"
      +`<td>${sl.lane!=null?('<span class="dim">:'+sl.lane+' \u00b7 </span>'):""}slot ${sl.slot!=null?sl.slot:sl.id}${sl.spec?"":' <span class="dim">no spec</span>'}</td>`
      +`<td class="dim">${phase}${(phase==="prefill"&&rate==null)?" · queued":""}</td>`
      +`<td class="rate">${rate==null?'<span class="dim">…</span>':fmt(rate)}</td>`
      +`<td>${Math.round(sl.decoded).toLocaleString()}</td>`
      +`<td>${Math.round(sl.prompt).toLocaleString()}</td>`
      +`<td>${hit==null?'<span class="dim">—</span>':hit.toFixed(0)+"%"}</td>`
      +`<td class="dim">${Math.round(sl.remain).toLocaleString()}</td>`
      +"</tr>");
  }
  // Slots that finished should not keep stale history around.
  const live=new Set(slots.map(x=>x.id));
  for(const k of Array.from(slotHist.keys())) if(!live.has(k)) slotHist.delete(k);

  // A lane with no /slots (SGLang) reports counts, not slots. Its row says
  // what the engine says about all its work at once, and its throughput
  // joins the fleet total: both halves are tok/s, and a page that shows
  // only the llama half of a mixed fleet is not a fleet view.
  for(const lr of (s.lane_rows||[])){
    if(lr.rate>0){ total+=lr.rate; haveRate=true; }
    if(lr.running>0) decoding+=lr.running;
    rows.push("<tr>"
      +`<td><span class="dim">:${lr.lane} \u00b7 </span>${lr.engine} <span class="dim">whole lane</span></td>`
      +`<td class="dim">${lr.running>0?"running":"idle"}${lr.queued>0?(" \u00b7 "+lr.queued+" queued"):""}</td>`
      +`<td class="rate">${lr.rate>0?fmt(lr.rate):'<span class="dim">\u2014</span>'}</td>`
      +`<td>${Math.round(lr.running)} <span class="dim">reqs</span></td>`
      +`<td>${Math.round(lr.ctx).toLocaleString()}</td>`
      +`<td>${Math.round(100*(lr.cached||0))}%</td>`
      +`<td class="dim">KV ${Math.round(100*(lr.kv||0))}%</td>`
      +"</tr>");
  }
  const box=document.getElementById("slots");
  const note=document.getElementById("slots_n");
  const out={decoding,prefilling,total,haveRate};
  if(!rows.length){
    box.innerHTML='<div class="quiet">no stream is generating</div>'+recentHtml(s);
    note.textContent = (s.slots||[]).length ? ((s.slots||[]).length+" slots idle") : "";
    return out;
  }
  const nlr=(s.lane_rows||[]).length;
  note.textContent = (rows.length-nlr)+" of "+(s.slots||[]).length+" slots busy"
    + (nlr ? (" \u00b7 "+nlr+" lane"+(nlr>1?"s":"")+" without /slots") : "");
  box.innerHTML="<table><thead><tr>"
    +"<th>stream</th><th>phase</th><th>tok/s</th><th>generated</th>"
    +"<th>context</th><th>cached</th><th>budget left</th></tr></thead><tbody>"
    +rows.join("")+"</tbody>"
    // Only decode rates are summed. Adding a prefill rate to a decode rate
    // would produce a number that means nothing.
    +(rows.length>1 && haveRate
       ? '<tfoot><tr><td>aggregate</td><td class="dim">decode</td>'
         +`<td class="rate">${total.toFixed(1)}</td><td colspan="4"></td></tr></tfoot>`
       : "")
    +"</table>"+recentHtml(s);
  return out;
}
// Control buttons: POST to /__control with the token from the last sample.
let _ctrlToken="";
function wireCtl(root){
  root.querySelectorAll("button").forEach(btn=>{
    if(btn.dataset.wired) return; btn.dataset.wired="1";
    btn.addEventListener("click",async ()=>{
      if(!_ctrlToken) return;
      const act=btn.dataset.act, preset=btn.dataset.preset;
      // Scope the disable and the message to this group: the Container bit,
      // the Upstream-process bit and the Model row each carry their own.
      const lane=btn.dataset.lane;
      const grp=btn.closest(".bit")||btn.closest(".row")||btn.closest(".lane"), msg=grp.querySelector(".ctl_msg");
      const where=lane?("lane :"+lane):"this GPU";
      const q=act==="assign"?("assign "+preset+" to "+where+"? Whatever it fronts now is unloaded first."):(act+" the upstream on "+where+"? This frees its GPU.");
      if(act!=="start" && !confirm(q)) return;
      grp.querySelectorAll("button").forEach(b=>b.disabled=true);
      try{
        const r=await fetch("/__control?token="+encodeURIComponent(_ctrlToken)
          +"&action="+act+(preset?("&preset="+encodeURIComponent(preset)):"")+(lane?("&lane="+encodeURIComponent(lane)):""),{method:"POST",cache:"no-store"});
        const j=await r.json();
        msg.textContent=j.status?("-> "+j.status+(j.port?(" on :"+j.port):"")):(j.error||"");
      }catch(e){ msg.textContent="request failed"; }
      setTimeout(()=>{ msg.textContent="";
        grp.querySelectorAll("button").forEach(b=>b.disabled=false); },3000);
    });
  });
}
document.querySelectorAll(".ctl").forEach(wireCtl);
tick(); setInterval(tick,3000);
</script>
"""
PAGE = (PAGE.replace("__HEADER_CSS__", _HEADER_CSS)
           .replace("__SVG_SYMBOLS__", _SVG_SYMBOLS)
           .replace("__HEADER__", header("stats")))
