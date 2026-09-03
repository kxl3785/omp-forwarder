"""The usage page served at http://127.0.0.1:<port>/__usage

WHY IT IS SEPARATE FROM /__stats: the live dashboard answers "is the model
healthy right now". This answers "how much work did the local model do instead
of a paid API". Different question, different refresh rate, and a different
shape -- accounting rather than telemetry. It is its own module because the
two pages share no markup, and stats.py is long enough already.

WHY THE ROWS ARE THE ROWS: llama-server counts every prompt token in exactly
one of two counters, freshly read or reused from a warm slot. Those are the
same two lines a paid API bills as "input" and "cache read", so the comparison
is a mapping rather than an estimate. There is deliberately no cache-write
row: a paid API charges a premium to populate its cache and a local server
does not, which means a real bill would be a little higher than what this
page shows. Saying that is more useful than inventing a number for it.

Rates are editable and stored per browser. They change, and the right
comparison depends on which model you would otherwise have used.

It polls /__stats.json, the same snapshot the live dashboard uses. There is no
second endpoint to keep in step.
"""
from __future__ import annotations

PAGE = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>omp forwarder &mdash; usage</title>
<style>
:root{
  --bg:#0b141a; --panel:#101e26; --panel2:#0d1a21; --line:#1d3440;
  --ink:#dcecf1; --dim:#7f9ba7; --teal:#5ed6cb; --amber:#f0b25e;
  --red:#e2685f; --green:#5cc98c;
  --mono:ui-monospace,"Cascadia Mono","SF Mono",Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5
  ui-sans-serif,"Segoe UI",Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:880px;margin:0 auto;padding:26px 22px 60px}
h1{font-size:19px;margin:0 0 3px;font-weight:600}
h2{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--dim);
  font-weight:600;margin:30px 0 12px;display:flex;align-items:center;gap:12px}
h2 .rule{flex:1;height:1px;background:var(--line)}
h2 .note{color:var(--dim);font-weight:400;letter-spacing:0;text-transform:none;font-size:11.5px}
.sub{color:var(--dim);font-size:12.5px;margin:0 0 15px}
/* Same nav as the live dashboard, so the two pages read as one thing. */
.tabs{display:flex;gap:8px;margin:0 0 24px}
.tabs a{color:var(--dim);text-decoration:none;font-size:12.5px;padding:6px 14px;
  border:1px solid var(--line);border-radius:8px;background:var(--panel2)}
.tabs a:hover{color:var(--ink);border-color:#2a4a58}
.tabs a.on{color:var(--teal);border-color:var(--teal);background:var(--panel)}
.grid{display:grid;gap:11px;grid-template-columns:repeat(auto-fit,minmax(170px,1fr))}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  padding:13px 15px 14px}
.k{font-size:9.5px;letter-spacing:.1em;color:var(--dim);text-transform:uppercase}
.v{font-family:var(--mono);font-size:25px;line-height:1.2;margin-top:6px}
.v .u{font-size:12px;color:var(--dim);margin-left:5px;font-family:inherit}
.n{font-size:11px;color:var(--dim);margin-top:5px;min-height:15px}
.teal{color:var(--teal)} .amber{color:var(--amber)} .dim{color:var(--dim)}
.tbl{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}
table{width:100%;border-collapse:collapse}
th,td{padding:9px 15px;text-align:right;font-size:12.5px;border-bottom:1px solid var(--line)}
th{font-size:9.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--dim);
  font-weight:600;background:var(--panel2)}
th:first-child,td:first-child{text-align:left}
tbody tr:last-child td{border-bottom:none}
td.num{font-family:var(--mono)}
tr.total td{background:var(--panel2);font-weight:600}
tr.na td{color:var(--dim)}
.bars{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.bar{display:grid;grid-template-columns:52px 1fr 104px;gap:11px;align-items:center;margin-bottom:7px}
.bar:last-of-type{margin-bottom:0}
.bar .d{font-family:var(--mono);font-size:11px;color:var(--dim)}
.bar .t{font-family:var(--mono);font-size:11.5px;text-align:right}
.track{height:15px;background:var(--panel2);border-radius:4px;overflow:hidden;display:flex}
.track i{display:block;height:100%}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin-top:13px;font-size:11px;color:var(--dim)}
.legend span{display:flex;align-items:center;gap:6px}
.sw{width:9px;height:9px;border-radius:2px;display:inline-block}
.rates{display:flex;gap:10px;flex-wrap:wrap;align-items:flex-end}
.rate{display:flex;flex-direction:column;gap:4px}
.rate label{font-size:9.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--dim)}
.rate input{width:100px;background:var(--panel2);border:1px solid var(--line);
  color:var(--ink);font-family:var(--mono);font-size:13px;padding:6px 9px;border-radius:7px}
.rate input:focus{outline:none;border-color:var(--teal)}
.presets{display:flex;gap:7px;margin-left:auto;flex-wrap:wrap}
.presets button{background:var(--panel2);border:1px solid var(--line);color:var(--dim);
  font:inherit;font-size:11.5px;padding:6px 11px;border-radius:7px;cursor:pointer}
.presets button:hover{color:var(--ink);border-color:#2a4a58}
.presets button.on{color:var(--teal);border-color:var(--teal)}
.foot{margin-top:26px;color:var(--dim);font-size:11.5px;line-height:1.75;
  border-top:1px solid var(--line);padding-top:16px}
.foot b{color:var(--ink);font-weight:600}
.foot code{font-family:var(--mono);font-size:11px;color:var(--teal)}
.quiet{color:var(--dim);font-size:12.5px;padding:12px 0}
body.stale .wrap{opacity:.45;transition:opacity .3s}
</style>
<div class="wrap">
<h1>omp forwarder &mdash; usage</h1>
<p class="sub">What the local model did instead of a paid API.</p>
<div class="tabs"><a href="/__stats">Live</a><a class="on" href="/__usage">Usage</a></div>

<h2>Today<span class="rule"></span><span class="note" id="day"></span></h2>
<div class="grid">
  <div class="card"><div class="k">Tokens</div>
    <div class="v" id="t_all">&mdash;</div>
    <div class="n" id="t_all_n">prompt + generated</div></div>
  <div class="card"><div class="k">Would have cost</div>
    <div class="v"><span class="teal">$</span><span id="t_cost">&mdash;</span></div>
    <div class="n">bought at the rates below</div></div>
  <div class="card"><div class="k">Prompt cache</div>
    <div class="v"><span id="t_hit">&mdash;</span><span class="u">%</span></div>
    <div class="n" id="t_hit_n">of prompt tokens reused</div></div>
  <div class="card"><div class="k">Generated</div>
    <div class="v" id="t_gen">&mdash;</div>
    <div class="n">tokens written by the model</div></div>
</div>

<h2>Breakdown<span class="rule"></span></h2>
<div class="tbl"><table>
  <thead><tr><th>&nbsp;</th><th>Today</th><th>Last 7 days</th><th>All recorded</th></tr></thead>
  <tbody id="bd"><tr><td colspan="4" class="quiet">loading&hellip;</td></tr></tbody>
</table></div>

<h2>By day<span class="rule"></span><span class="note" id="hist_n"></span></h2>
<div class="bars" id="bars"><div class="quiet">no days recorded yet</div></div>

<h2>Comparison rates<span class="rule"></span><span class="note">list prices, editable, stored in this browser</span></h2>
<div class="card">
  <div class="rates">
    <div class="rate"><label for="r_in">Input $/M</label>
      <input id="r_in" type="number" step="0.01" min="0"></div>
    <div class="rate"><label for="r_cache">Cache read $/M</label>
      <input id="r_cache" type="number" step="0.01" min="0"></div>
    <div class="rate"><label for="r_out">Output $/M</label>
      <input id="r_out" type="number" step="0.01" min="0"></div>
    <div class="presets" id="presets"></div>
  </div>
</div>

<div class="foot">
  <b>How this lines up with a paid bill.</b> llama&#8209;server counts every
  prompt token in exactly one of two places: read fresh, or reused from a warm
  slot. Those are the two lines a paid API bills as <b>input</b> and
  <b>cache&nbsp;read</b>, so the rows here are a mapping rather than an
  estimate. There is no cache&#8209;write row, because populating a local cache
  costs nothing extra. A paid API charges a premium for it, so a real bill
  would come out a little above the figure shown.<br>
  <b>Read the dollar figure as an order of magnitude.</b> Two models split the
  same text into different tokens, so the counts are comparable in scale, not
  exactly.<br>
  <b>It measures the server, not one client.</b> Everything that reached
  <code>llama-server</code> while the forwarder was running is counted,
  including requests sent from Studio&rsquo;s own interface. Daily totals live
  in <code>tokens.json</code> beside the log and survive restarts. Work done
  before the forwarder first ran cannot be known.
</div>
</div>
<script>
const $=i=>document.getElementById(i);
const NUM=n=>Math.round(n).toLocaleString();
const money=n=>n>=100?n.toFixed(0):(n>=1?n.toFixed(2):n.toFixed(3));
let missed=0, last=null;

// List prices per million tokens. A paid API charges a fraction of the input
// rate to READ its cache; the local server charges nothing, which is most of
// what this page exists to show.
const PRESETS={
  "Fable 5.1":{rin:10, rcache:0.25, rout:50},
  "Opus 5":{rin:5, rcache:0.50, rout:25},
  "Sonnet 5":{rin:2, rcache:0.20, rout:10},
};
const STORE="ompfwd.rates.v1";
let rates=Object.assign({}, PRESETS["Fable 5.1"]);
try{ const raw=localStorage.getItem(STORE); if(raw) rates=Object.assign(rates,JSON.parse(raw)); }
catch(e){ /* private window, or site data blocked -- run on the defaults */ }

function writeInputs(){ $("r_in").value=rates.rin; $("r_cache").value=rates.rcache; $("r_out").value=rates.rout; }
function readInputs(){
  rates={rin:+$("r_in").value||0, rcache:+$("r_cache").value||0, rout:+$("r_out").value||0};
  try{ localStorage.setItem(STORE,JSON.stringify(rates)); }catch(e){}
  markPreset(); if(last) render(last);
}
function markPreset(){
  for(const b of $("presets").children){
    const p=PRESETS[b.textContent];
    b.classList.toggle("on", p && p.rin===rates.rin && p.rcache===rates.rcache && p.rout===rates.rout);
  }
}
for(const name of Object.keys(PRESETS)){
  const b=document.createElement("button"); b.textContent=name;
  b.onclick=()=>{ rates=Object.assign({},PRESETS[name]); writeInputs(); readInputs(); };
  $("presets").appendChild(b);
}
for(const id of ["r_in","r_cache","r_out"]) $(id).addEventListener("input",readInputs);
writeInputs(); markPreset();

const cost=t=>t.prompt/1e6*rates.rin + t.cached/1e6*rates.rcache + t.gen/1e6*rates.rout;
const sum=ds=>ds.reduce((a,d)=>({prompt:a.prompt+d.prompt, cached:a.cached+d.cached,
                                 gen:a.gen+d.gen}), {prompt:0,cached:0,gen:0});

function render(s){
  const days=(s.days||[]).slice();
  const today={prompt:s.tok_today_prompt||0, cached:s.tok_today_cached||0,
               gen:s.tok_today_gen||0};
  // The in-memory tally leads the day file, which is written every 30 s. When
  // this run has counted more than the file records, trust the live figure --
  // otherwise the page reads stale for half a minute after every request.
  const live={prompt:s.tok_prompt||0, cached:s.tok_cached||0, gen:s.tok_gen||0};
  if(live.prompt+live.cached+live.gen > today.prompt+today.cached+today.gen){
    Object.assign(today, live);
    if(days.length) days[0]=Object.assign({}, days[0], today);
  }
  const week=sum(days.slice(0,7)), all=sum(days);

  $("day").textContent = days.length ? days[0].day : "";
  const tAll=today.prompt+today.gen;
  $("t_all").textContent=NUM(tAll);
  $("t_all_n").textContent = tAll>0
    ? NUM(today.prompt)+" prompt · "+NUM(today.gen)+" generated"
    : "nothing yet today";
  $("t_cost").textContent=money(cost(today));
  $("t_gen").textContent=NUM(today.gen);
  const seen=today.prompt+today.cached;
  if(seen>0){
    $("t_hit").textContent=(100*today.cached/seen).toFixed(1);
    $("t_hit_n").textContent=NUM(today.cached)+" tokens reused";
  } else { $("t_hit").textContent="—"; $("t_hit_n").textContent="no prompts yet"; }

  const row=(label,note,pick)=>"<tr><td>"+label
    +(note?' <span class="dim">'+note+"</span>":"")
    +'</td><td class="num">'+NUM(pick(today))
    +'</td><td class="num">'+NUM(pick(week))
    +'</td><td class="num">'+NUM(pick(all))+"</td></tr>";
  $("bd").innerHTML =
      row("Input","fresh",t=>t.prompt)
    + row("Cache read","reused",t=>t.cached)
    + '<tr class="na"><td>Cache write <span class="dim">not charged locally</span></td>'
    +   "<td>—</td><td>—</td><td>—</td></tr>"
    + row("Output","generated",t=>t.gen)
    + '<tr class="total"><td>Cost if bought</td>'
    +   '<td class="num teal">$'+money(cost(today))+"</td>"
    +   '<td class="num teal">$'+money(cost(week))+"</td>"
    +   '<td class="num teal">$'+money(cost(all))+"</td></tr>";

  const show=days.slice(0,14);
  $("hist_n").textContent = days.length
    ? (days.length===1 ? "1 day recorded" : days.length+" days recorded") : "";
  if(!show.length){ $("bars").innerHTML='<div class="quiet">no days recorded yet</div>'; return; }
  const peak=Math.max(1, ...show.map(d=>d.prompt+d.cached+d.gen));
  const w=x=>(100*x/peak).toFixed(2)+"%";
  $("bars").innerHTML = show.map(d=>
      '<div class="bar"><div class="d">'+d.day.slice(5)+"</div>"
    + '<div class="track">'
    +   '<i style="background:var(--teal);width:'+w(d.prompt)+'"></i>'
    +   '<i style="background:#2a6c78;width:'+w(d.cached)+'"></i>'
    +   '<i style="background:var(--amber);width:'+w(d.gen)+'"></i>'
    + '</div><div class="t">$'+money(cost(d))+"</div></div>"
  ).join("")
    + '<div class="legend">'
    + '<span><i class="sw" style="background:var(--teal)"></i>input</span>'
    + '<span><i class="sw" style="background:#2a6c78"></i>cache read</span>'
    + '<span><i class="sw" style="background:var(--amber)"></i>output</span>'
    + "<span>bar width is tokens · the figure is what it would have cost</span>"
    + "</div>";
}

async function tick(){
  let s;
  try{ s=await (await fetch("/__stats.json",{cache:"no-store"})).json(); }
  catch(e){ if(++missed>2) document.body.classList.add("stale"); return; }
  missed=0; document.body.classList.remove("stale");
  last=s; render(s);
}
tick(); setInterval(tick,5000);
</script>
"""
