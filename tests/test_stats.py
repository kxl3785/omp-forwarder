"""stats.py: the /metrics parser, the /slots and /v1/models readers, and the
JSON snapshot the dashboard polls."""
from __future__ import annotations

import json

from omp_forwarder import forwarder as fwd
from omp_forwarder import stats

from .helpers import ForwarderCase, free_port, http_response

METRICS = """# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 1234
llamacpp:n_decode_total 56

llamacpp:requests_processing 0
llamacpp:some_text_value not-a-number
"""


def json_response(obj) -> bytes:
    return http_response(200, json.dumps(obj).encode(), "application/json")


class ParseMetricsTests(ForwarderCase):

    def test_skips_comments_blanks_and_non_numeric(self):
        self.assertEqual(stats.parse_metrics(METRICS), {
            "llamacpp:prompt_tokens_total": 1234.0,
            "llamacpp:n_decode_total": 56.0,
            "llamacpp:requests_processing": 0.0,
        })

    def test_empty(self):
        self.assertEqual(stats.parse_metrics(""), {})


class UpstreamReadersTests(ForwarderCase):

    def test_metrics_from_a_live_upstream(self):
        up = self.fake(lambda m, p, r: [http_response(200, METRICS.encode())])
        self.assertEqual(stats.upstream_metrics(up.port)
                         ["llamacpp:n_decode_total"], 56.0)

    def test_metrics_without_an_upstream(self):
        self.assertEqual(stats.upstream_metrics(None), {})
        self.assertEqual(stats.upstream_metrics(free_port()), {})

    def test_model_name_drops_the_org_prefix(self):
        up = self.fake(lambda m, p, r: [json_response(
            {"data": [{"id": "esatapedico/Qwen-27B-GGUF"}]})])
        self.assertEqual(stats.upstream_model(up.port), "Qwen-27B-GGUF")

    def test_model_name_from_the_models_key(self):
        up = self.fake(lambda m, p, r: [json_response(
            {"models": [{"name": "a/b/c"}]})])
        self.assertEqual(stats.upstream_model(up.port), "c")

    def test_model_name_fallbacks(self):
        self.assertEqual(stats.upstream_model(None), "-")
        self.assertEqual(stats.upstream_model(free_port()), "-")
        empty = self.fake(lambda m, p, r: [json_response({"data": []})])
        self.assertEqual(stats.upstream_model(empty.port), "-")
        not_found = self.fake()                    # 404 for /v1/models
        self.assertEqual(stats.upstream_model(not_found.port), "-")

    def test_slots_shape(self):
        up = self.fake(lambda m, p, r: [json_response([
            {"id": 0, "id_task": 7, "is_processing": True,
             "n_prompt_tokens": 100, "n_prompt_tokens_cache": 90,
             "speculative": True,
             "next_token": [{"n_decoded": 12, "n_remain": 88}]},
            # An idle slot: no next_token at all, nothing in flight.
            {"id": 1, "id_task": -1, "is_processing": False},
        ])])
        self.assertEqual(stats.upstream_slots(up.port), [
            {"id": 0, "task": 7, "busy": True, "decoded": 12, "remain": 88,
             "prompt": 100, "cached": 90, "spec": True, "n_ctx": 0},
            {"id": 1, "task": -1, "busy": False, "decoded": 0, "remain": 0,
             "prompt": 0, "cached": 0, "spec": False, "n_ctx": 0},
        ])

    def test_slots_fallbacks(self):
        self.assertEqual(stats.upstream_slots(None), [])
        self.assertEqual(stats.upstream_slots(free_port()), [])
        not_a_list = self.fake(lambda m, p, r: [json_response(
            {"error": "slots endpoint disabled"})])
        self.assertEqual(stats.upstream_slots(not_a_list.port), [])


class SnapshotTests(ForwarderCase):

    def test_snapshot_with_a_live_upstream(self):
        def responder(method, path, raw):
            if path == "/metrics":
                return [http_response(200, METRICS.encode())]
            if path == "/slots":
                return [json_response([])]
            return [http_response(404)]

        up = self.fake(responder)
        fwd._upstream = up.port
        fwd.LISTEN_PORT = 8891
        fwd._stats.update({"conns": 5, "requests": 3, "2xx": 2, "5xx": 1,
                           "fallbacks": 1, "model": "m",
                           "latency": [0.3, 0.1, 0.2]})
        d = stats.snapshot(fwd, fwd._stats)
        self.assertEqual(d["upstream"], up.port)
        self.assertEqual(d["listen"], 8891)
        self.assertTrue(d["live"])
        self.assertEqual(d["model"], "m")
        self.assertEqual((d["conns"], d["requests"], d["status_2xx"],
                          d["status_4xx"], d["status_5xx"], d["fallbacks"]),
                         (5, 3, 2, 0, 1, 1))
        self.assertEqual(d["latency_p50_ms"], 200.0)
        self.assertEqual(d["prompt_tokens"], 1234.0)
        self.assertEqual(d["decode_steps"], 56.0)
        self.assertEqual(d["gen_tokens"], 0)      # absent metric reads 0
        self.assertEqual(d["slots"], [])
        self.assertEqual(d["ctx"], 0)             # no slots, no window known
        self.assertGreaterEqual(d["uptime_s"], 0)

    def test_snapshot_without_an_upstream(self):
        fwd._upstream = None
        d = stats.snapshot(fwd, fwd._stats)
        self.assertIsNone(d["upstream"])
        self.assertFalse(d["live"])
        self.assertEqual(d["model"], "-")
        self.assertIsNone(d["latency_p50_ms"])
        self.assertEqual(d["slots"], [])
