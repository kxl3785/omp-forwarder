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
_STARTED = time.time()


def parse_metrics(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        key, _, val = line.partition(" ")
        try:
            out[key] = float(val)
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

def snapshot(fwd, stats: dict) -> dict:
    """One JSON sample. Cumulative counters are returned raw -- the page
    differences successive samples for live rates, and differences a stored
    baseline for session figures. Both need the raw counter."""
    port = getattr(fwd, "_upstream", None)
    m = upstream_metrics(port)
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
    slots = upstream_slots(port)
    # The keepalive child's pid when container mode is on, else None.
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
        "peers": list(getattr(fwd, "PEERS", [])),
        # The /__control token, so the dashboard can authorise its buttons.
        # See forwarder._control_token for why exposing it is safe.
        "control_token": getattr(fwd, "_control_token", None),
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
        # Per-stream, from /slots. /metrics cannot give this.
        "slots": slots,
    }


PAGE = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title id="pg_title">omp forwarder</title>
<style>
:root{
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
.bar{display:flex;flex-wrap:wrap;gap:24px;align-items:center;
  background:var(--panel);border:1px solid var(--line);border-radius:12px;
  padding:12px 18px;margin-bottom:16px}
.bit{display:flex;flex-direction:column;gap:1px;min-width:0}
.bk{font-size:9.5px;letter-spacing:.1em;color:var(--dim);text-transform:uppercase}
.bv{font-family:var(--mono);font-size:13px;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:7px}
/* Two pages, one nav. It rides on the title line and both pages carry an
   identical header, so switching moves nothing on screen but the highlight. */
.tabs{display:flex;border:1px solid var(--line);border-radius:8px;
  overflow:hidden;background:var(--panel2);flex:none}
.tabs a{color:var(--dim);text-decoration:none;font-size:12px;padding:6px 15px;
  line-height:1.5}
.tabs a+a{border-left:1px solid var(--line)}
.tabs a:hover{color:var(--ink)}
.tabs a.on{color:var(--teal);background:var(--panel)}
.dot.on{background:var(--green);box-shadow:0 0 0 3px rgba(92,201,140,.16)}
.dot.off{background:var(--red);box-shadow:0 0 0 3px rgba(226,104,95,.16)}
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
/* Container controls: three quiet buttons next to the Container bit. */
.ctl{display:inline-flex;gap:5px;margin-left:10px}
.ctl button{appearance:none;background:transparent;border:1px solid var(--line);
  border-radius:6px;color:var(--dim);cursor:pointer;padding:2px 8px;
  font:inherit;font-size:10px;letter-spacing:.05em;transition:.15s}
.ctl button:hover{color:var(--teal);border-color:var(--teal);
  background:rgba(94,214,203,.07)}
.ctl button.danger:hover{color:var(--red);border-color:var(--red);
  background:rgba(226,104,95,.07)}
#ctl_msg{font-size:11px;color:var(--dim);margin-left:8px}
/* Peer pills ride the tabs row, right-aligned. */
.peers{display:flex;gap:6px;align-items:center}
.peers a.lane{font-family:var(--mono);font-size:11px;color:var(--dim);
  text-decoration:none;border:1px solid var(--line);border-radius:6px;
  padding:2px 8px}
.peers a.lane:hover{color:var(--teal);border-color:var(--teal)}
.fname{font-family:var(--mono);font-size:13px;color:var(--dim);
  margin-left:8px;font-weight:400}
.tabsrow{display:flex;align-items:center;gap:16px;margin-top:10px}
.tabsrow .peers{margin-left:auto}
.pill{display:inline-block;font-family:var(--mono);font-size:11px;
  border:1px solid var(--line);background:transparent;border-radius:10px;
  padding:1px 7px;color:var(--dim);letter-spacing:.04em}
.pill.on{color:var(--green);border-color:var(--green)}
.pill.off{color:var(--red);border-color:var(--red)}
</style>
<svg style="display:none">
  <symbol id="ico-rst" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
    <path d="M20.5 12a8.5 8.5 0 1 1-2.6-6.1"/><path d="M20.6 4.2v5.2h-5.2"/>
  </symbol>
</svg>
<div class="wrap">
<div class="head">
<h1>
<svg class="mark" viewBox="0 0 24 24" aria-hidden="true">
  <rect x="1.3" y="1.3" width="21.4" height="21.4" rx="5.8" fill="#0f1e26" stroke="#1d3440"/>
  <path d="M5.6 12h12.2M13.3 7.8L18.1 12l-4.8 4.2" fill="none" stroke="#5ed6cb"
        stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
<span>omp forwarder</span><span class="fname" id="fname"></span></h1>
<div class="tabsrow">
<nav class="tabs"><a class="on" href="/__stats">Live</a><a href="/__usage">Usage</a></nav>
<div class="peers" id="peers"></div>
</div>
</div>
<div class="sub">Traffic omp sends straight to llama-server &mdash; the requests Unsloth&rsquo;s own API panel cannot see.</div>

<div class="bar">
  <div class="bit"><span class="bk">Listening</span><span class="bv" id="listen">&mdash;</span></div>
  <div class="bit"><span class="bk">Upstream</span><span class="bv" id="up">&mdash;</span></div>
  <div class="bit"><span class="bk">Status</span><span class="bv"><span class="dot off" id="dot"></span><span id="state">connecting</span></span></div>
  <div class="bit" style="flex:1"><span class="bk">Model</span><span class="bv" id="model">&mdash;</span></div>
  <div class="bit"><span class="bk">Uptime</span><span class="bv" id="uptime">&mdash;</span></div>
  <div class="bit"><span class="bk">Container</span><span class="bv" id="cont" title="">&mdash;</span><span class="ctl" id="ctl"><button data-act="start">start</button><button data-act="stop">stop</button><button data-act="restart" class="danger">restart</button></span><span id="ctl_msg"></span></span></div>
  <button class="rst" id="rst-all" title="Reset every section to start from now">
    <svg><use href="#ico-rst"/></svg>Reset all</button>
</div>

<h2>Deployment<span class="rule"></span></h2>
<div class="facts" id="facts">
  <div class="row"><span class="fk">Engine</span><span class="fv dim" id="f_engine">&mdash;</span></div>
  <div class="row"><span class="fk">Thinking</span><span class="fv" id="f_thinking">&mdash;</span></div>
  <div class="row"><span class="fk">Speculative</span><span class="fv dim" id="f_spec">&mdash;</span></div>
  <div class="row"><span class="fk">Parallel</span><span class="fv dim" id="f_par">&mdash;</span></div>
  <div class="row"><span class="fk">Model path</span><span class="fv dim" id="f_model" title="">&mdash;</span></div>
  <div class="row"><span class="fk">Keepalive PID</span><span class="fv dim" id="f_ka">&mdash;</span></div>
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
    <div class="v"><span id="acc">&mdash;</span><span class="u">%</span></div>
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
  $("listen").textContent="127.0.0.1:"+s.listen;
  _ctrlToken=s.control_token||"";
  $("up").textContent=s.upstream?("127.0.0.1:"+s.upstream):"Studio :8888 (fallback)";
  $("up").title = s.upstream_exe || "";
  $("model").textContent=s.model; $("uptime").textContent=dur(s.uptime_s);
  $("dot").className="dot "+(s.healthy?"on":"off");
  $("state").textContent=s.healthy?"ready":(s.upstream?"unreachable":"no direct server");
  if(s.container_name){
    $("cont").textContent=s.container_name+" · "+(s.container||"unknown");
    $("cont").title="container "+s.container_name+" in WSL distro; status: "+(s.container||"unknown");
  } else {
    $("cont").textContent=""; $("cont").title="";
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
  if(noMetrics){
    $("tput").textContent="—"; $("tput_n").textContent=NP;
    $("dec").textContent="—";  $("dec_n").textContent=NP;
    $("pre").textContent="—";  $("pre_n").textContent=NP;
    $("acc").textContent="—";  $("acc_n").textContent=NP;
    meterSet($("acc_m"),0,50,40);
  } else if(prev){
    const dg=s.gen_tokens-prev.gen_tokens, dsec=s.gen_seconds-prev.gen_seconds;
    const dp=s.prompt_tokens-prev.prompt_tokens, dps=s.prompt_seconds-prev.prompt_seconds;
    const dd=s.draft_total-prev.draft_total, da=s.draft_accepted-prev.draft_accepted;
    const dsteps=s.decode_steps-prev.decode_steps, dw=s.t-prev.t;
    const live=renderSlots(s);

    // Throughput: prefer per-stream decode rates from /slots; fall back to
    // the completion counter when /slots is unavailable.
    if((s.slots||[]).length){
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
    if(a!=null){ $("acc").textContent=a.toFixed(1); meterSet($("acc_m"),a,50,40);
      $("acc").className = a<40?"red":(a<50?"amber":"teal");
      $("acc_n").innerHTML = (a<40 ? "LOW — check for n-gram fallback" : "MTP speculation")
        + (avg ? (sess?'<span class="tag">session</span>':LIFE) : ""); }
    else { $("acc").textContent="—"; $("acc_n").textContent="no drafts yet"; }
  }

  // --- server section: in flight, prompt cache, tokens per pass, largest ctx ---
  if(noMetrics){
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
    pb.innerHTML=s.peers.map(pp=>'<a class="lane" href="http://127.0.0.1:'+pp+'/__stats" target="_blank">lane :'+pp+'</a>').join("");
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
  setF("f_model", F2.model_path);
  const kaEl=$("f_ka");
  if(s.keepalive_pid){ kaEl.textContent=s.keepalive_pid; kaEl.classList.remove("dim"); }
  else { kaEl.innerHTML="&mdash;"; kaEl.classList.add("dim"); }

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
      +`<td>slot ${sl.id}${sl.spec?"":' <span class="dim">no spec</span>'}</td>`
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

  const box=document.getElementById("slots");
  const note=document.getElementById("slots_n");
  const out={decoding,prefilling,total,haveRate};
  if(!rows.length){
    box.innerHTML='<div class="quiet">no stream is generating</div>';
    note.textContent = (s.slots||[]).length ? ((s.slots||[]).length+" slots idle") : "";
    return out;
  }
  note.textContent = rows.length+" of "+(s.slots||[]).length+" slots busy";
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
    +"</table>";
  return out;
}
// Control buttons: POST to /__control with the token from the last sample.
let _ctrlToken="";
document.querySelectorAll("#ctl button").forEach(btn=>{
  btn.addEventListener("click",async ()=>{
    if(!_ctrlToken) return;
    const act=btn.dataset.act;
    document.querySelectorAll("#ctl button").forEach(b=>b.disabled=true);
    try{
      const r=await fetch("/__control?token="+encodeURIComponent(_ctrlToken)
        +"&action="+act,{method:"POST",cache:"no-store"});
      const j=await r.json();
      $("ctl_msg").textContent=j.status?("-> "+j.status):(j.error||"");
    }catch(e){ $("ctl_msg").textContent="request failed"; }
    setTimeout(()=>{ $("ctl_msg").textContent="";
      document.querySelectorAll("#ctl button").forEach(b=>b.disabled=false); },3000);
  });
});
tick(); setInterval(tick,3000);
</script>
"""
