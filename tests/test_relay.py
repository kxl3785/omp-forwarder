"""The relay end to end, against a fake upstream socket.

These are the tests that guard the raw-TCP constraint: bytes past the request
line must reach the upstream untouched, and a streamed response must reach
the client as it is produced, with EOF at the end."""
from __future__ import annotations

import http.client
import json
import socket
import threading
import time
from unittest import mock

from omp_forwarder import forwarder as fwd

from .helpers import (HOST, RelayCase, free_port, http_response, raw_request,
                      wait_for)

REQUEST = (b"POST /v1/chat/completions HTTP/1.1\r\n"
           b"Host: 127.0.0.1\r\n"
           b"Content-Type: application/json\r\n"
           b"Content-Length: 17\r\n"
           b"\r\n"
           b'{"prompt":"pong"}')


def echo_body(method, path, raw):
    """Reply with the request bytes as the body, so a test can check exactly
    what the upstream received."""
    return [http_response(200, raw, "application/octet-stream")]


class RelayTests(RelayCase):

    def test_request_line_and_body_in_one_read_reach_upstream(self):
        up = self.fake(echo_body)
        fwd.FORCED_UPSTREAM = up.port
        out = raw_request(self.port, [REQUEST])
        self.wait_done(1)
        self.assertEqual(up.received, [REQUEST])
        self.assertTrue(out.startswith(b"HTTP/1.1 200 OK"))
        self.assertTrue(out.endswith(REQUEST))
        self.assertEqual(fwd._stats["requests"], 1)
        self.assertEqual(fwd._stats["2xx"], 1)
        self.assertEqual(fwd._stats["fallbacks"], 0)

    def test_body_arriving_after_the_request_line_is_forwarded_in_order(self):
        up = self.fake(echo_body)
        fwd.FORCED_UPSTREAM = up.port
        line, rest = REQUEST.split(b"\r\n", 1)
        # The relay reads only up to the first CRLF to route. Everything after
        # it must flow through the pump unchanged, whenever it arrives.
        raw_request(self.port, [line + b"\r\n", 0.2, rest])
        self.wait_done(1)
        self.assertEqual(up.received, [REQUEST])

    def test_binary_body_is_not_altered(self):
        up = self.fake(echo_body)
        fwd.FORCED_UPSTREAM = up.port
        body = bytes(range(256)) * 300      # 76.8 kB, more than one recv
        req = (b"POST /v1/embeddings HTTP/1.1\r\n"
               + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
        raw_request(self.port, [req])
        self.wait_done(1)
        self.assertEqual(len(up.received), 1)
        self.assertEqual(up.received[0], req)

    def test_streamed_response_is_relayed_as_it_arrives(self):
        head = (b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                b"Connection: close\r\n\r\n")
        chunks = [b"data: one\n\n", b"data: two\n\n", b"data: [DONE]\n\n"]

        def stream(method, path, raw):
            return [head, chunks[0], 0.6, chunks[1], chunks[2]]

        up = self.fake(stream)
        fwd.FORCED_UPSTREAM = up.port
        with socket.create_connection((HOST, self.port), timeout=10) as c:
            c.sendall(REQUEST)
            t0 = time.time()
            first = c.recv(65536)
            t_first = time.time() - t0
            rest = b""
            while True:
                d = c.recv(65536)
                if not d:
                    break
                rest += d
            t_all = time.time() - t0
        # The first bytes must arrive before the upstream's pause ends. A relay
        # that buffered until close would fail this.
        self.assertLess(t_first, 0.5, "first bytes were held back")
        self.assertGreater(t_all, 0.5)
        self.assertEqual(first + rest, head + b"".join(chunks))
        self.wait_done(1)
        self.assertEqual(fwd._stats["2xx"], 1)

    def test_status_counters_follow_upstream_status_lines(self):
        def by_path(method, path, raw):
            code = {"/ok": 200, "/missing": 404, "/boom": 500}.get(path, 200)
            return [http_response(code, b"x")]

        up = self.fake(by_path)
        fwd.FORCED_UPSTREAM = up.port
        for path in ("/ok", "/missing", "/boom", "/boom"):
            raw_request(self.port, [f"GET {path} HTTP/1.1\r\n\r\n".encode()])
        self.wait_done(4)
        self.assertEqual(fwd._stats["requests"], 4)
        self.assertEqual((fwd._stats["2xx"], fwd._stats["4xx"],
                          fwd._stats["5xx"]), (1, 1, 2))
        self.assertEqual(len(fwd._stats["latency"]), 4)

    def test_two_concurrent_requests_do_not_mix(self):
        def slow_echo(method, path, raw):
            return [0.3, http_response(200, raw)]

        up = self.fake(slow_echo)
        fwd.FORCED_UPSTREAM = up.port
        reqs = [f"GET /r{i} HTTP/1.1\r\nX-N: {i}\r\n\r\n".encode()
                for i in range(2)]
        results = {}

        def go(i):
            results[i] = raw_request(self.port, [reqs[i]])

        ts = [threading.Thread(target=go, args=(i,)) for i in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(10)
        self.wait_done(2)
        for i in range(2):
            self.assertTrue(results[i].endswith(reqs[i]),
                            f"request {i} got another request's reply")
        self.assertEqual(sorted(up.received), sorted(reqs))

    def test_stats_page_is_served_locally(self):
        up = self.fake()
        fwd.FORCED_UPSTREAM = up.port
        conn = http.client.HTTPConnection(HOST, self.port, timeout=10)
        conn.request("GET", "/__stats")
        r = conn.getresponse()
        body = r.read()
        self.assertEqual(r.status, 200)
        self.assertIn("text/html", r.getheader("Content-Type"))
        self.assertIn(b'<title id="pg_title">omp forwarder</title>', body)
        # Served by the forwarder itself: nothing reached the upstream, and it
        # does not count as a relayed request.
        self.assertEqual(up.received, [])
        self.assertEqual(fwd._stats["requests"], 0)
        self.assertEqual(fwd._stats["conns"], 1)

    def test_favicon_is_answered_locally_and_closes_the_connection(self):
        # Not cosmetic. The relay routes only the first request on a
        # connection, so a relayed /favicon.ico used to drag the page's own
        # /__stats.json fetch upstream on the same keep-alive connection --
        # a 404, and a dashboard of zeros on first load.
        up = self.fake()
        fwd.FORCED_UPSTREAM = up.port
        out = raw_request(self.port, [b"GET /favicon.ico HTTP/1.1\r\n"
                                      b"Host: 127.0.0.1\r\n\r\n"])
        self.assertTrue(out.startswith(b"HTTP/1.1 204 No Content"))
        self.assertIn(b"Connection: close", out)
        self.assertEqual(up.received, [])
        self.assertEqual(fwd._stats["requests"], 0)

    def test_usage_page_is_served_locally(self):
        up = self.fake()
        fwd.FORCED_UPSTREAM = up.port
        conn = http.client.HTTPConnection(HOST, self.port, timeout=10)
        conn.request("GET", "/__usage")
        r = conn.getresponse()
        body = r.read()
        self.assertEqual(r.status, 200)
        self.assertIn("text/html", r.getheader("Content-Type"))
        self.assertIn(b"omp forwarder &mdash; usage", body)
        # It polls the same snapshot the live dashboard uses, so there is no
        # second endpoint to keep in step.
        self.assertIn(b"/__stats.json", body)
        self.assertEqual(up.received, [])
        self.assertEqual(fwd._stats["requests"], 0)

    def test_usage_json_returns_the_snapshot(self):
        up = self.fake()
        fwd.FORCED_UPSTREAM = up.port
        conn = http.client.HTTPConnection(HOST, self.port, timeout=10)
        conn.request("GET", "/__usage.json")
        r = conn.getresponse()
        d = json.loads(r.read())
        self.assertEqual(r.status, 200)
        self.assertIn("tok_today_cached", d)
        self.assertIn("days", d)
        self.assertEqual(fwd._stats["requests"], 0)

    def test_stats_json_reports_upstream_and_counters(self):
        def responder(method, path, raw):
            if path == "/health":
                return [http_response(200)]
            if path == "/metrics":
                return [http_response(200, b"# HELP x\n"
                                      b"llamacpp:n_decode_total 7\n"
                                      b"llamacpp:requests_processing 1\n")]
            if path == "/v1/models":
                return [http_response(200, json.dumps(
                    {"data": [{"id": "some-org/model-x"}]}).encode(),
                    "application/json")]
            if path == "/slots":
                return [http_response(200, json.dumps([{
                    "id": 0, "id_task": 12, "is_processing": True,
                    "n_prompt_tokens": 500, "n_prompt_tokens_cache": 400,
                    "speculative": True, "n_ctx": 262144,
                    "next_token": [{"n_decoded": 33, "n_remain": 67}],
                }]).encode(), "application/json")]
            return [http_response(200, b"ok")]

        up = self.fake(responder)
        fwd.FORCED_UPSTREAM = up.port
        raw_request(self.port, [b"GET /v1/models HTTP/1.1\r\n\r\n"])
        self.wait_done(1)
        conn = http.client.HTTPConnection(HOST, self.port, timeout=10)
        conn.request("GET", "/__stats.json")
        r = conn.getresponse()
        d = json.loads(r.read())
        self.assertEqual(r.status, 200)
        self.assertEqual(d["listen"], self.port)
        self.assertEqual(d["upstream"], up.port)
        self.assertTrue(d["live"])
        self.assertEqual(d["model"], "model-x")
        self.assertEqual(d["decode_steps"], 7)
        self.assertEqual(d["processing"], 1)
        self.assertEqual(d["requests"], 1)
        self.assertEqual(d["status_2xx"], 1)
        self.assertIsInstance(d["latency_p50_ms"], float)
        self.assertEqual(d["slots"], [{
            "id": 0, "task": 12, "busy": True, "decoded": 33, "remain": 67,
            "prompt": 500, "cached": 400, "spec": True, "n_ctx": 262144}])
        self.assertEqual(d["ctx"], 262144)
        # The dashboard fetch itself is not a relayed request.
        self.assertEqual(fwd._stats["requests"], 1)

    def test_stale_cached_upstream_is_rediscovered(self):
        up = self.fake(echo_body)
        fwd._upstream = free_port()                 # what Studio used to run on
        with mock.patch.object(fwd, "_server_pids", return_value={"1"}), \
                mock.patch.object(fwd, "_listening_ports",
                                  return_value=[up.port]):
            out = raw_request(self.port, [REQUEST])
        self.wait_done(1)
        self.assertTrue(out.startswith(b"HTTP/1.1 200 OK"))
        self.assertEqual(fwd._upstream, up.port)
        self.assertEqual(fwd._stats["fallbacks"], 0)

    def test_failed_connect_forces_a_rescan(self):
        # The health check passes for the cached port, but the connect fails.
        # That is what a model reload looks like from here: Studio re-rolled
        # the port between the check and the connect.
        up = self.fake(echo_body)
        dead = free_port()
        fwd._upstream = dead
        with mock.patch.object(fwd, "_healthy", return_value=True), \
                mock.patch.object(fwd, "_server_pids", return_value={"1"}), \
                mock.patch.object(fwd, "_listening_ports",
                                  return_value=[up.port]):
            out = raw_request(self.port, [REQUEST])
        self.wait_done(1)
        self.assertTrue(out.startswith(b"HTTP/1.1 200 OK"))
        self.assertEqual(fwd._upstream, up.port)

    def test_no_model_returns_503_not_studios_401(self):
        # The bug this replaces: with no llama-server the request went to
        # Studio, which requires its own API key and answered 401. A client
        # reads 401 as "your config is wrong, stop trying" when the truth is
        # "the model is loading". It cost a real outage.
        studio = self.fake(lambda m, p, r: [http_response(401, b"no key")])
        fwd.STUDIO_PORT = studio.port
        with mock.patch.object(fwd, "_server_pids", return_value=set()):
            out = raw_request(self.port, [REQUEST])
        self.assertTrue(out.startswith(b"HTTP/1.1 503 Service Unavailable"))
        self.assertIn(b"Retry-After: 5", out)
        self.assertIn(b"model_not_loaded", out)
        body = json.loads(out.split(b"\r\n\r\n", 1)[1])
        self.assertEqual(body["error"]["type"], "service_unavailable")
        # Studio was never touched, so it cannot contribute a 401.
        self.assertEqual(studio.received, [])
        self.assertEqual(fwd._stats["unavailable"], 1)
        self.assertEqual(fwd._stats["fallbacks"], 0)
        self.assertEqual(fwd._stats["4xx"], 0)

    def test_studio_fallback_is_opt_in(self):
        studio = self.fake(echo_body)
        fwd.STUDIO_PORT = studio.port
        fwd.STUDIO_FALLBACK = True
        with mock.patch.object(fwd, "_server_pids", return_value=set()):
            out = raw_request(self.port, [REQUEST])
        self.wait_done(1)
        self.assertTrue(out.startswith(b"HTTP/1.1 200 OK"))
        self.assertEqual(studio.received, [REQUEST])
        self.assertEqual(fwd._stats["fallbacks"], 1)
        self.assertEqual(fwd._stats["unavailable"], 0)
        self.assertIsNone(fwd._upstream)

    def test_waits_for_a_model_that_appears_mid_request(self):
        # A reload leaves a gap of tens of seconds. Failing instantly there
        # turns "still loading" into an error.
        up = self.fake(echo_body)
        fwd.WAIT_FOR_MODEL = 20
        ready = []

        def pids():
            # Absent on the first two scans, then present.
            ready.append(1)
            return {"1"} if len(ready) > 2 else set()

        # The port lookup must honour its argument: no pids, no ports. A mock
        # that ignored it would report a server during the gap.
        with mock.patch.object(fwd, "_server_pids", side_effect=pids), \
                mock.patch.object(fwd, "_listening_ports",
                                  side_effect=lambda ps: [up.port] if ps else []):
            t0 = time.time()
            out = raw_request(self.port, [REQUEST], timeout=30)
        waited = time.time() - t0
        self.wait_done(1)
        self.assertTrue(out.startswith(b"HTTP/1.1 200 OK"))
        # up.received also holds the /health probes discovery made.
        self.assertIn(REQUEST, up.received)
        self.assertGreater(waited, 0.9, "did not actually wait")
        self.assertLess(waited, 20, "waited past the model appearing")
        self.assertEqual(fwd._stats["unavailable"], 0)

    def test_wait_gives_up_at_the_deadline(self):
        fwd.WAIT_FOR_MODEL = 2
        with mock.patch.object(fwd, "_server_pids", return_value=set()):
            t0 = time.time()
            out = raw_request(self.port, [REQUEST], timeout=30)
        waited = time.time() - t0
        self.assertTrue(out.startswith(b"HTTP/1.1 503"))
        self.assertGreaterEqual(waited, 2)
        self.assertLess(waited, 12)
        self.assertEqual(fwd._stats["unavailable"], 1)
        # A 503 we served ourselves is not a relayed round trip.
        self.assertEqual(fwd._stats["latency"], [])

    def test_client_that_sends_nothing_is_dropped(self):
        with socket.create_connection((HOST, self.port), timeout=10) as c:
            c.shutdown(socket.SHUT_WR)
            self.assertEqual(c.recv(10), b"")
        wait_for(lambda: fwd._stats["conns"] == 1, what="conn counted")
        self.assertEqual(fwd._stats["requests"], 0)


class ReadRequestHeadTests(RelayCase):

    def test_keeps_bytes_read_past_the_request_line(self):
        a, b = socket.socketpair()
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        a.sendall(REQUEST)
        head = fwd._read_request_head(b)
        self.assertTrue(head.startswith(b"POST /v1/chat/completions"))
        self.assertEqual(head, REQUEST)

    def test_returns_empty_on_immediate_close(self):
        a, b = socket.socketpair()
        self.addCleanup(b.close)
        a.close()
        self.assertEqual(fwd._read_request_head(b), b"")
