"""One page drives every lane. A peer's preset state rides along in
_peer_state for the Lanes panel; its control token goes to _peer_tokens and
never into the snapshot. `lane=<peer port>` on /__control relays the action
to that peer with the peer's own token, server-side, loopback only."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

from omp_forwarder import forwarder as fwd
from omp_forwarder import stats

from .helpers import ForwarderCase, RelayCase, raw_request


def _body(out: bytes) -> dict:
    return json.loads(out.split(b"\r\n\r\n", 1)[1])


PEER_SNAPSHOT = {
    "name": "GPU 1", "healthy": True, "gpu": 1, "facts": {"engine": "sglang"},
    "preset": "sglang-nothink", "loading": False, "operator_stopped": False,
    "presets": ["llama-tune", "sglang-nothink"], "model": "qwen",
    "control_token": "peer-secret",
}


class PeerStateTests(ForwarderCase):

    def test_preset_state_is_kept_and_token_is_not_published(self):
        fwd.PEERS = [8891]
        fwd.LISTEN_PORT = 8892
        with mock.patch.object(stats, "_http_get_json", return_value=dict(PEER_SNAPSHOT)):
            fwd._sample_peers()
        p = fwd._peer_state[8891]
        self.assertEqual((p["preset"], p["loading"], p["operator_stopped"]),
                         ("sglang-nothink", False, False))
        self.assertEqual(p["presets"], ["llama-tune", "sglang-nothink"])
        self.assertNotIn("control_token", p)
        self.assertEqual(fwd._peer_tokens[8891], "peer-secret")
        snap = stats.snapshot(fwd, dict(fwd._stats))
        self.assertNotIn("peer-secret", json.dumps(snap))


class LaneRelayTests(RelayCase):

    def setUp(self):
        super().setUp()
        fwd._control_token = "tok"
        fwd.PEERS = [8891]
        fwd._peer_tokens = {8891: "peer-secret"}
        fwd._presets = {"llama-tune": {"kind": "process", "port": 1, "cmd": "x"}}

    def _post(self, q: str) -> bytes:
        return raw_request(self.port, [f"POST /__control?{q} HTTP/1.1\r\n\r\n".encode()])

    def test_relays_with_the_peer_token(self):
        seen = []
        with mock.patch.object(fwd, "_peer_control",
                               side_effect=lambda port, token, action, preset, timeout=90.0:
                               seen.append((port, token, action, preset)) or
                               (200, {"ok": True, "status": "loading", "port": 49501})):
            out = self._post("token=tok&lane=8891&action=assign&preset=llama-tune")
        self.assertTrue(out.startswith(b"HTTP/1.1 200"), out[:80])
        self.assertEqual(seen, [(8891, "peer-secret", "assign", "llama-tune")])
        b = _body(out)
        self.assertEqual((b["status"], b["lane"]), ("loading", 8891))

    def test_stale_peer_token_is_refreshed_and_retried_once(self):
        # 8891 restarted between two sampler ticks: its token changed, the
        # first relay got 403, and the operator saw "bad token".
        calls = []

        def relay(port, token, action, preset, timeout=90.0):
            calls.append(token)
            if token == "peer-secret":
                return 403, {"ok": False, "error": "bad token"}
            return 200, {"ok": True, "status": "stopped"}

        def resample():
            fwd._peer_tokens[8891] = "new-secret"

        with mock.patch.object(fwd, "_peer_control", side_effect=relay):
            with mock.patch.object(fwd, "_sample_peers", side_effect=resample):
                out = self._post("token=tok&lane=8891&action=stop")
        self.assertTrue(out.startswith(b"HTTP/1.1 200"), out[:80])
        self.assertEqual(calls, ["peer-secret", "new-secret"])

    def test_peer_403_with_unchanged_token_is_not_retried(self):
        calls = []

        def relay(port, token, action, preset, timeout=90.0):
            calls.append(token)
            return 403, {"ok": False, "error": "bad token"}

        with mock.patch.object(fwd, "_peer_control", side_effect=relay):
            with mock.patch.object(fwd, "_sample_peers"):
                out = self._post("token=tok&lane=8891&action=stop")
        self.assertTrue(out.startswith(b"HTTP/1.1 403"), out[:80])
        self.assertEqual(calls, ["peer-secret"])

    def test_peer_status_passes_through(self):
        with mock.patch.object(fwd, "_peer_control",
                               return_value=(409, {"ok": False, "error": "no-gpu"})):
            out = self._post("token=tok&lane=8891&action=assign&preset=llama-tune")
        self.assertTrue(out.startswith(b"HTTP/1.1 409"), out[:80])

    def test_unknown_lane_is_400(self):
        out = self._post("token=tok&lane=9999&action=stop")
        self.assertTrue(out.startswith(b"HTTP/1.1 400"), out[:80])
        self.assertEqual(_body(out)["lanes"], [8891])

    def test_peer_token_not_yet_read_is_503(self):
        fwd._peer_tokens = {}
        out = self._post("token=tok&lane=8891&action=stop")
        self.assertTrue(out.startswith(b"HTTP/1.1 503"), out[:80])

    def test_local_token_still_gates_the_relay(self):
        out = self._post("token=wrong&lane=8891&action=stop")
        self.assertTrue(out.startswith(b"HTTP/1.1 403"), out[:80])

    def test_get_with_lane_never_relays(self):
        with mock.patch.object(fwd, "_peer_control") as pc:
            out = raw_request(self.port, [b"GET /__control?token=tok&lane=8891&action=stop HTTP/1.1\r\n\r\n"])
        self.assertTrue(out.startswith(b"HTTP/1.1 405"), out[:80])
        pc.assert_not_called()

    def test_own_port_as_lane_acts_locally(self):
        fwd.PEERS = [8891, self.port]
        fwd.FWD_GPU = None
        with mock.patch.object(fwd, "_peer_control") as pc:
            out = self._post(f"token=tok&lane={self.port}&action=assign&preset=llama-tune")
        pc.assert_not_called()
        self.assertTrue(out.startswith(b"HTTP/1.1 409"), out[:80])  # no --gpu, local answer


class PeerControlHttpTests(ForwarderCase):
    """The one real HTTP call, against a tiny loopback server."""

    def test_posts_query_and_parses_json(self):
        seen = []

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                seen.append(self.path)
                body = json.dumps({"ok": True, "status": "stopped"}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        code, body = fwd._peer_control(srv.server_address[1], "t0k", "stop", "", timeout=5)
        self.assertEqual((code, body["status"]), (200, "stopped"))
        self.assertEqual(seen, ["/__control?token=t0k&action=stop"])

    def test_dead_peer_is_502(self):
        from .helpers import free_port
        code, body = fwd._peer_control(free_port(), "t", "stop", "", timeout=2)
        self.assertEqual(code, 502)
        self.assertFalse(body["ok"])


class ModelNameSamplerTests(ForwarderCase):
    """The header's model name is the operator's proof that the right model
    loaded. A dash from a not-yet-ready server must not stick."""

    def test_dash_is_retried_once_healthy(self):
        fwd._upstream = 49500
        fwd._stats["model"] = "-"          # what an early poll stored
        fwd._upstream_healthy = True
        with mock.patch.object(stats, "upstream_model", return_value="Qwen3.8-27B"):
            fwd._sample_model()
        self.assertEqual(fwd._stats["model"], "Qwen3.8-27B")

    def test_unhealthy_upstream_is_not_asked_and_dash_is_cleared(self):
        fwd._upstream = 49500
        fwd._stats["model"] = "-"
        fwd._upstream_healthy = False
        with mock.patch.object(stats, "upstream_model") as um:
            fwd._sample_model()
        um.assert_not_called()
        self.assertEqual(fwd._stats["model"], "")

    def test_known_name_is_kept(self):
        fwd._upstream = 49500
        fwd._stats["model"] = "known"
        fwd._upstream_healthy = True
        with mock.patch.object(stats, "upstream_model") as um:
            fwd._sample_model()
        um.assert_not_called()
        self.assertEqual(fwd._stats["model"], "known")


class DeadUpstreamSnapshotTests(ForwarderCase):
    """An unloaded lane must answer /__stats.json at once. A connect to a
    closed loopback port can take the full timeout on Windows, and the
    snapshot used to pay it twice per poll."""

    def test_readers_skipped_once_sampler_says_unhealthy(self):
        fwd._upstream = 49501
        fwd._health_sampled = True
        fwd._upstream_healthy = False
        with mock.patch.object(stats, "upstream_metrics") as m:
            with mock.patch.object(stats, "upstream_slots") as sl:
                stats.snapshot(fwd, dict(fwd._stats))
        m.assert_not_called()
        sl.assert_not_called()

    def test_readers_run_before_the_first_sample_and_when_healthy(self):
        fwd._upstream = 49501
        for sampled, healthy in ((False, False), (True, True)):
            fwd._health_sampled, fwd._upstream_healthy = sampled, healthy
            with mock.patch.object(stats, "upstream_metrics", return_value={}) as m:
                with mock.patch.object(stats, "upstream_slots", return_value=[]):
                    stats.snapshot(fwd, dict(fwd._stats))
            m.assert_called_once()


class LaneOwnershipTests(ForwarderCase):
    """A lane never discovers a server that another lane fronts, nor a port
    that a preset maps to another GPU."""

    def _scan(self, healthy_ports):
        return (mock.patch.object(fwd, "_server_pids", return_value={"1": "x"}),
                mock.patch.object(fwd, "_listening_ports", return_value=list(healthy_ports)),
                mock.patch.object(fwd, "_healthy", side_effect=lambda p: p in healthy_ports))

    def test_peer_upstream_is_skipped(self):
        fwd._peer_state = {8891: {"port": 8891, "upstream": 49501, "reachable": True}}
        a, b, c = self._scan([49501, 49500])
        with a, b, c:
            self.assertEqual(fwd._choose_upstream(), (49500, "llama-server"))

    def test_other_gpu_preset_port_is_skipped_even_with_no_peer(self):
        fwd.FWD_GPU = 0
        fwd._presets = {"llama-tune": {"kind": "process", "port": "4950{gpu}", "cmd": "x"}}
        a, b, c = self._scan([49501])
        with a, b, c:
            self.assertIsNone(fwd._choose_upstream())

    def test_own_gpu_preset_port_is_taken(self):
        fwd.FWD_GPU = 1
        fwd._presets = {"llama-tune": {"kind": "process", "port": "4950{gpu}", "cmd": "x"}}
        a, b, c = self._scan([49501])
        with a, b, c:
            self.assertEqual(fwd._choose_upstream(), (49501, "llama-server"))

    def test_no_gpu_flag_means_no_preset_exclusion(self):
        fwd.FWD_GPU = None
        fwd._presets = {"llama-tune": {"kind": "process", "port": "4950{gpu}", "cmd": "x"}}
        a, b, c = self._scan([49501])
        with a, b, c:
            self.assertEqual(fwd._choose_upstream(), (49501, "llama-server"))


class MergeSnapshotTests(ForwarderCase):
    """One snapshot for every lane."""

    def _lane(self, port, **kw):
        d = {"listen": port, "requests": 0, "tok_gen": 0, "gen_tokens": 0,
             "ctx": 0, "healthy": False, "model": "-", "slots": [], "days": []}
        d.update(kw)
        return d

    def test_counters_add_and_marks_take_max(self):
        own = self._lane(8890, requests=3, tok_gen=100, ctx=4096, healthy=False)
        peer = self._lane(8891, requests=5, tok_gen=50, ctx=262144, healthy=True)
        m = stats.merge_snapshots(own, [peer])
        self.assertEqual((m["requests"], m["tok_gen"], m["ctx"]), (8, 150, 262144))
        self.assertTrue(m["healthy"])
        self.assertEqual(m["fleet"], {"lanes": 2, "serving": 1, "ports": [8890, 8891]})

    def test_slots_concatenate_with_lane_tags_and_unique_ids(self):
        own = self._lane(8890, slots=[{"id": 0, "busy": True, "decoded": 5}])
        peer = self._lane(8891, slots=[{"id": 0, "busy": True, "decoded": 9}])
        m = stats.merge_snapshots(own, [peer])
        self.assertEqual([(r["id"], r["lane"], r["slot"]) for r in m["slots"]],
                         [("8890:0", 8890, 0), ("8891:0", 8891, 0)])

    def test_models_join_and_days_sum_by_day(self):
        own = self._lane(8890, model="tune", days=[{"day": "2026-09-05", "prompt": 1, "cached": 2, "gen": 3}])
        peer = self._lane(8891, model="candidate", days=[{"day": "2026-09-05", "prompt": 10, "cached": 20, "gen": 30},
                                                         {"day": "2026-09-04", "prompt": 1, "cached": 1, "gen": 1}])
        m = stats.merge_snapshots(own, [peer])
        self.assertEqual(m["model"], "tune + candidate")
        self.assertEqual(m["days"], [{"day": "2026-09-04", "prompt": 1, "cached": 1, "gen": 1},
                                     {"day": "2026-09-05", "prompt": 11, "cached": 22, "gen": 33}])

    def test_own_lane_fields_stay_own(self):
        own = self._lane(8890, control_token="mine", preset="llama-tune", gpu=0)
        peer = self._lane(8891, control_token="theirs", preset="sglang-nothink", gpu=1)
        m = stats.merge_snapshots(own, [peer])
        self.assertEqual((m["control_token"], m["preset"], m["gpu"], m["listen"]),
                         ("mine", "llama-tune", 0, 8890))

    def test_single_lane_is_unchanged_but_named(self):
        own = self._lane(8890, requests=3, slots=[{"id": 0}])
        m = stats.merge_snapshots(own, [])
        self.assertEqual(m["requests"], 3)
        self.assertEqual(m["slots"], [{"id": 0}])
        self.assertEqual(m["fleet"]["lanes"], 1)


class FleetJsonTests(RelayCase):
    """/__stats.json is the fleet; ?self=1 is this lane alone."""

    PEER = {"listen": 8891, "requests": 5, "healthy": True, "model": "candidate",
            "slots": [{"id": 0, "busy": True, "decoded": 1}], "days": [],
            "control_token": "peer-secret"}

    def setUp(self):
        super().setUp()
        fwd._stats["requests"] = 2
        fwd._peer_state = {8891: {"port": 8891, "reachable": True}}

    def _get(self, path):
        out = raw_request(self.port, [f"GET {path} HTTP/1.1\r\n\r\n".encode()])
        return _body(out)

    def test_default_view_folds_reachable_peers_in(self):
        asked = []
        with mock.patch.object(stats, "_http_get_json",
                               side_effect=lambda port, path, timeout=4: asked.append((port, path)) or dict(self.PEER)):
            d = self._get("/__stats.json")
        self.assertEqual(asked, [(8891, "/__stats.json?self=1")])
        self.assertEqual(d["requests"], 7)
        self.assertEqual(d["fleet"]["lanes"], 2)
        self.assertEqual(d["slots"][0]["lane"], 8891)
        self.assertNotIn("peer-secret", json.dumps(d))

    def test_self_view_asks_no_peer(self):
        with mock.patch.object(stats, "_http_get_json") as g:
            d = self._get("/__stats.json?self=1")
        g.assert_not_called()
        self.assertEqual(d["requests"], 2)
        self.assertNotIn("fleet", d)   # a lane alone is not a fleet

    def test_unreachable_peer_is_not_asked(self):
        fwd._peer_state = {8891: {"port": 8891, "reachable": False}}
        with mock.patch.object(stats, "_http_get_json") as g:
            d = self._get("/__stats.json")
        g.assert_not_called()
        self.assertEqual(d["requests"], 2)


class LaunchFailedTests(ForwarderCase):
    """A dead launch reads as failed, not as loading."""

    def setUp(self):
        super().setUp()
        fwd._preset = "sglang-nothink"
        fwd._upstream_healthy = False
        fwd._operator_stopped = False

    def test_exited_container_is_failed_not_loading(self):
        fwd.WSL_DISTRO, fwd.CONTAINER_NAME = "d", "sgl1"
        fwd._container_status = "exited"
        s = stats.snapshot(fwd, dict(fwd._stats))
        self.assertTrue(s["launch_failed"])
        self.assertFalse(s["loading"])

    def test_running_container_is_still_loading(self):
        fwd.WSL_DISTRO, fwd.CONTAINER_NAME = "d", "sgl1"
        fwd._container_status = "running"
        s = stats.snapshot(fwd, dict(fwd._stats))
        self.assertFalse(s["launch_failed"])
        self.assertTrue(s["loading"])

    def test_returned_process_is_failed(self):
        fwd._upstream_child = mock.Mock(poll=mock.Mock(return_value=2))
        s = stats.snapshot(fwd, dict(fwd._stats))
        self.assertTrue(s["launch_failed"])
        self.assertFalse(s["loading"])

    def test_live_process_is_loading(self):
        fwd._upstream_child = mock.Mock(poll=mock.Mock(return_value=None))
        s = stats.snapshot(fwd, dict(fwd._stats))
        self.assertFalse(s["launch_failed"])
        self.assertTrue(s["loading"])

    def test_stopped_lane_is_neither(self):
        fwd._operator_stopped = True
        fwd._upstream_child = mock.Mock(poll=mock.Mock(return_value=1))
        s = stats.snapshot(fwd, dict(fwd._stats))
        self.assertFalse(s["launch_failed"])
        self.assertFalse(s["loading"])


def _slot(sid, task, decoded, prompt=100, cached=0, busy=True):
    return {"id": sid, "task": task, "decoded": decoded, "prompt": prompt,
            "cached": cached, "busy": busy}


class RecentStreamTests(ForwarderCase):
    """Finished streams are kept, so the panel shows traffic between bursts.
    A slot keeps its last task and token count after release, so a stream
    that ran entirely between two samples is still recorded."""

    def test_a_stream_watched_to_the_end_reports_its_average_rate(self):
        stats.note_slots([_slot(0, 7, 10)], 100.0)
        stats.note_slots([_slot(0, 7, 210)], 102.0)
        stats.note_slots([_slot(0, 8, 0, busy=False)], 103.0)
        r = stats.recent_streams(8890)
        self.assertEqual((r[0]["slot"], r[0]["tokens"], r[0]["lane"]), (0, 210, 8890))
        self.assertEqual((r[0]["seconds"], r[0]["rate"]), (2.0, 105.0))

    def test_a_stream_that_ran_between_samples_is_still_recorded(self):
        # Never seen busy: at the next sample the slot is idle and already
        # carries the finished task's id and its final token count.
        stats.note_slots([_slot(0, 7, 40, busy=False)], 100.0)
        stats.note_slots([_slot(0, 9, 88, busy=False)], 101.0)
        r = stats.recent_streams(8890)
        self.assertEqual([(x["tokens"], x["rate"]) for x in r], [(88, None), (40, None)])

    def test_the_first_sample_only_seeds(self):
        # A live forwarder starts with every slot already carrying the task
        # it last ran, months of uptime ago. Those are history, not traffic.
        stats._seeded = False
        stats.note_slots([_slot(0, 7, 40, busy=False)], 100.0)
        self.assertEqual(stats.recent_streams(8890), [])
        stats.note_slots([_slot(0, 9, 88, busy=False)], 101.0)
        self.assertEqual([x["tokens"] for x in stats.recent_streams(8890)], [88])

    def test_each_slot_reports_without_waiting_for_new_work(self):
        # Three requests over three slots, none of which is used again.
        # The first version only recorded a stream when its slot took new
        # work, so two of these were never seen.
        stats.note_slots([_slot(0, 1, 5), _slot(1, 2, 5), _slot(2, 3, 5)], 100.0)
        stats.note_slots([_slot(0, 1, 30, busy=False),
                          _slot(1, 2, 40, busy=False),
                          _slot(2, 3, 50, busy=False)], 101.0)
        r = stats.recent_streams(8890)
        self.assertEqual(sorted(x["tokens"] for x in r), [30, 40, 50])

    def test_a_new_task_on_the_same_slot_ends_the_old_stream(self):
        stats.note_slots([_slot(0, 7, 100)], 100.0)
        stats.note_slots([_slot(0, 9, 5)], 101.0)
        r = stats.recent_streams(8890)
        self.assertEqual([(x["slot"], x["tokens"]) for x in r], [(0, 100)])

    def test_the_same_stream_is_recorded_once(self):
        stats.note_slots([_slot(0, 7, 50, busy=False)], 100.0)
        for i in range(4):
            stats.note_slots([_slot(0, 7, 50, busy=False)], 101.0 + i)
        self.assertEqual(len(stats.recent_streams(8890)), 1)

    def test_a_stream_that_generated_nothing_is_not_recorded(self):
        stats.note_slots([_slot(0, 7, 0)], 100.0)
        stats.note_slots([_slot(0, 8, 0, busy=False)], 101.0)
        self.assertEqual(stats.recent_streams(8890), [])

    def test_newest_first_and_capped(self):
        for i in range(15):
            stats.note_slots([_slot(0, i, 10, busy=False)], 100.0 + i)
        r = stats.recent_streams(8890)
        self.assertEqual(len(r), 12)
        self.assertGreater(r[0]["ended"], r[-1]["ended"])

    def test_snapshot_carries_them(self):
        stats.note_slots([_slot(1, 3, 50, busy=False)], 10.0)
        stats.note_slots([_slot(1, 4, 0, busy=False)], 11.0)
        fwd.LISTEN_PORT = 8890
        d = stats.snapshot(fwd, dict(fwd._stats))
        self.assertEqual([(x["slot"], x["lane"]) for x in d["recent_streams"]], [(1, 8890)])

class RecentMergeTests(ForwarderCase):

    def test_merge_interleaves_lanes_newest_first(self):
        own = {"listen": 8890, "slots": [], "days": [], "healthy": False,
               "recent_streams": [{"slot": 0, "lane": 8890, "ended": 5.0, "tokens": 1}]}
        peer = {"listen": 8891, "slots": [], "days": [], "healthy": False,
                "recent_streams": [{"slot": 0, "lane": 8891, "ended": 9.0, "tokens": 2}]}
        m = stats.merge_snapshots(own, [peer])
        self.assertEqual([x["lane"] for x in m["recent_streams"]], [8891, 8890])


class LaneRowTests(ForwarderCase):
    """SGLang has no /slots, so it contributes one row per lane."""

    def test_sglang_metrics_make_a_lane_row(self):
        fwd.LISTEN_PORT = 8891
        fwd._upstream = 30001
        fwd._upstream_healthy = True
        m = {"sglang:num_running_reqs": 2.0, "sglang:num_queue_reqs": 1.0,
             "sglang:gen_throughput": 161.7, "sglang:cache_hit_rate": 0.42,
             "sglang:context_len": 262144.0, "sglang:token_usage": 0.3}
        with mock.patch.object(stats, "upstream_metrics", return_value=m):
            with mock.patch.object(stats, "upstream_slots", return_value=[]):
                d = stats.snapshot(fwd, dict(fwd._stats))
        self.assertEqual(len(d["lane_rows"]), 1)
        r = d["lane_rows"][0]
        self.assertEqual((r["lane"], r["engine"], r["running"], r["rate"]),
                         (8891, "sglang", 2.0, 161.7))

    def test_a_llama_lane_makes_no_lane_row(self):
        fwd._upstream = 49500
        fwd._upstream_healthy = True
        with mock.patch.object(stats, "upstream_metrics",
                               return_value={"llamacpp:n_decode_total": 5}):
            with mock.patch.object(stats, "upstream_slots", return_value=[]):
                d = stats.snapshot(fwd, dict(fwd._stats))
        self.assertEqual(d["lane_rows"], [])

    def test_merge_collects_rows_from_every_lane(self):
        own = {"listen": 8890, "slots": [], "days": [], "healthy": True,
               "lane_rows": []}
        peer = {"listen": 8891, "slots": [], "days": [], "healthy": True,
                "lane_rows": [{"lane": 8891, "engine": "sglang", "rate": 161.7}]}
        m = stats.merge_snapshots(own, [peer])
        self.assertEqual([r["lane"] for r in m["lane_rows"]], [8891])
