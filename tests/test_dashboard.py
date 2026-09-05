"""The dashboard control surface: honest status light, deployment facts,
container controls, and lane identity. All WSL/docker calls are faked; no
real processes or upstreams are started."""
from __future__ import annotations

import json
import socket
import subprocess
import time
from unittest import mock

from omp_forwarder import forwarder as fwd
from omp_forwarder import stats

from .helpers import ForwarderCase, RelayCase, free_port, http_response, raw_request


def _completed(stdout: str, rc: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=rc,
                                       stdout=stdout, stderr="")


def json_response(obj) -> bytes:
    return http_response(200, json.dumps(obj).encode(), "application/json")


# ----------------------------------------------------------------
# 1. Honest status light: _sample_health drives _upstream_healthy
# ----------------------------------------------------------------

class SampleHealthTests(ForwarderCase):
    """_sample_health: what ends up in _upstream_healthy."""

    def test_no_upstream(self):
        fwd._upstream = None
        fwd._sample_health()
        self.assertFalse(fwd._upstream_healthy)

    def test_healthy_upstream(self):
        up = self.fake()                     # /health -> 200
        fwd._upstream = up.port
        fwd._sample_health()
        self.assertTrue(fwd._upstream_healthy)

    def test_unhealthy_upstream(self):
        up = self.fake(healthy=False)         # /health -> 503
        fwd._upstream = up.port
        fwd._sample_health()
        self.assertFalse(fwd._upstream_healthy)

    def test_dead_port(self):
        fwd._upstream = free_port()
        fwd._sample_health()
        self.assertFalse(fwd._upstream_healthy)


class SnapshotHealthyTests(ForwarderCase):
    """snapshot() exposes healthy and metrics_available."""

    def test_healthy_true_when_swept(self):
        fwd._upstream_healthy = True
        fwd._upstream = None
        d = stats.snapshot(fwd, fwd._stats)
        self.assertTrue(d["healthy"])

    def test_healthy_false_by_default(self):
        # reset_state set it to False
        d = stats.snapshot(fwd, fwd._stats)
        self.assertFalse(d["healthy"])

    def test_metrics_available_with_upstream(self):
        up = self.fake(lambda m, p, r: [
            http_response(200, b"llamacpp:n_decode_total 42\n")])
        fwd._upstream = up.port
        d = stats.snapshot(fwd, fwd._stats)
        self.assertTrue(d["metrics_available"])

    def test_metrics_available_false_without_upstream(self):
        d = stats.snapshot(fwd, fwd._stats)
        self.assertFalse(d["metrics_available"])


# ----------------------------------------------------------------
# 2. Deployment facts: _sample_upstream_facts drives _upstream_facts
# ----------------------------------------------------------------

class SampleFactsTests(ForwarderCase):
    """_sample_upstream_facts: SGLang and llama-server fact shapes."""

    def test_no_upstream(self):
        fwd._upstream = None
        fwd._sample_upstream_facts()
        f = fwd._upstream_facts
        self.assertEqual(f["engine"], "unknown")
        self.assertEqual(f["thinking"], "unknown")
        self.assertEqual(f["speculative"], "none")
        self.assertEqual(f["model_path"], "")

    def test_sglang_facts(self):
        info = {"default_chat_template_kwargs": {"enable_thinking": True},
                "speculative_algorithm": "eagle3",
                "tp_size": 4, "pp_size": 2,
                "model_path": "/models/qwen3"}
        up = self.fake(lambda m, p, r: [
            json_response(info) if p == "/get_server_info"
            else http_response(404)])
        fwd._upstream = up.port
        fwd._sample_upstream_facts()
        f = fwd._upstream_facts
        self.assertEqual(f["engine"], "sglang")
        self.assertEqual(f["thinking"], "on")
        self.assertEqual(f["speculative"], "eagle3")
        self.assertEqual(f["parallel"], "tp=4 pp=2")
        self.assertEqual(f["model_path"], "/models/qwen3")

    def test_sglang_thinking_off(self):
        info = {"default_chat_template_kwargs": {"enable_thinking": False},
                "speculative_algorithm": "",
                "tp_size": 1, "pp_size": 1,
                "model_path": "/m"}
        up = self.fake(lambda m, p, r: [
            json_response(info) if p == "/get_server_info"
            else http_response(404)])
        fwd._upstream = up.port
        fwd._sample_upstream_facts()
        f = fwd._upstream_facts
        self.assertEqual(f["thinking"], "off")
        self.assertEqual(f["speculative"], "none")
        self.assertEqual(f["parallel"], "tp=1 pp=1")

    def test_llama_server_facts(self):
        props = {"chat_template_kwargs": {"enable_thinking": False},
                 "model": "Qwen-27B-GGUF"}
        up = self.fake(lambda m, p, r: [
            json_response(props) if p == "/props" else http_response(404)])
        fwd._upstream = up.port
        fwd._sample_upstream_facts()
        f = fwd._upstream_facts
        self.assertEqual(f["engine"], "llama-server")
        self.assertEqual(f["thinking"], "off")
        self.assertEqual(f["speculative"], "unknown")
        self.assertEqual(f["parallel"], "")
        self.assertEqual(f["model_path"], "Qwen-27B-GGUF")

    def test_neither_endpoint(self):
        # 404 both fact endpoints; only /health answers 200.
        up = self.fake(lambda m, p2, r: [
            http_response(200, b'{"status":"ok"}', "application/json")
            if p2 == "/health" else http_response(404)])
        fwd._upstream = up.port
        fwd._sample_upstream_facts()
        f = fwd._upstream_facts
        self.assertEqual(f["engine"], "unknown")
        self.assertEqual(f["thinking"], "unknown")


class SnapshotFactsTests(ForwarderCase):
    """snapshot() copies _upstream_facts into the JSON."""

    def test_facts_in_snapshot(self):
        fwd._upstream_facts = {"engine": "sglang", "thinking": "on",
                               "speculative": "eagle3",
                               "parallel": "tp=4 pp=2",
                               "model_path": "/models/qwen3"}
        fwd._upstream = None
        d = stats.snapshot(fwd, fwd._stats)
        self.assertEqual(d["facts"], fwd._upstream_facts)

    def test_facts_empty_by_default(self):
        # reset_state set it to {}
        d = stats.snapshot(fwd, fwd._stats)
        self.assertEqual(d["facts"], {})

    def test_keepalive_pid_in_snapshot(self):
        d = stats.snapshot(fwd, fwd._stats)
        self.assertIsNone(d["keepalive_pid"])

    def test_metrics_unavailable_but_facts_present(self):
        # SGLang has no /metrics, so metrics_available is false, but the
        # sampler still populates facts from /get_server_info. The dashboard
        # must show "not provided by this upstream" on the /metrics cards
        # while still rendering the Deployment panel with engine data.
        fwd._upstream_facts = {"engine": "sglang", "thinking": "on"}
        fwd._upstream = None
        d = stats.snapshot(fwd, fwd._stats)
        self.assertFalse(d["metrics_available"])
        self.assertEqual(d["facts"]["engine"], "sglang")


# ----------------------------------------------------------------
# 3. Container controls: POST /__control
# ----------------------------------------------------------------

class ControlEndpointTests(RelayCase):
    """The live /__control endpoint via raw_request."""

    def _post(self, query: str) -> tuple[str, dict]:
        """Send a POST /__control?<query> and return (status_line, body)."""
        raw = raw_request(self.port, [
            f"POST /__control{query} HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{self.port}\r\n"
            f"Content-Length: 0\r\n"
            f"\r\n".encode()])
        head, _, body = raw.partition(b"\r\n\r\n")
        status = head.split(b"\r\n", 1)[0].decode()
        parsed = json.loads(body)
        return status, parsed

    def _get(self, query: str) -> tuple[str, dict]:
        raw = raw_request(self.port, [
            f"GET /__control{query} HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{self.port}\r\n"
            f"\r\n".encode()])
        head, _, body = raw.partition(b"\r\n\r\n")
        status = head.split(b"\r\n", 1)[0].decode()
        parsed = json.loads(body)
        return status, parsed

    def test_container_mode_off_returns_404(self):
        fwd._control_token = "tok123"
        status, body = self._post("?token=tok123&action=start")
        self.assertIn("404", status)
        self.assertFalse(body["ok"])
        self.assertEqual(body["error"], "container mode off")

    def test_get_returns_405(self):
        fwd.WSL_DISTRO = "Ubuntu-24.04"
        fwd.CONTAINER_NAME = "sgl"
        fwd._control_token = "tok123"
        status, body = self._get("?token=tok123&action=start")
        self.assertIn("405", status)
        self.assertFalse(body["ok"])
        self.assertEqual(body["error"], "POST only")

    def test_bad_token_returns_403(self):
        fwd.WSL_DISTRO = "Ubuntu-24.04"
        fwd.CONTAINER_NAME = "sgl"
        fwd._control_token = "correct"
        status, body = self._post("?token=wrong&action=start")
        self.assertIn("403", status)
        self.assertFalse(body["ok"])
        self.assertEqual(body["error"], "bad token")

    def test_missing_token_returns_403(self):
        fwd.WSL_DISTRO = "Ubuntu-24.04"
        fwd.CONTAINER_NAME = "sgl"
        fwd._control_token = "correct"
        status, body = self._post("?action=start")
        self.assertIn("403", status)
        self.assertFalse(body["ok"])
        self.assertEqual(body["error"], "bad token")

    def test_unknown_action_returns_400(self):
        fwd.WSL_DISTRO = "Ubuntu-24.04"
        fwd.CONTAINER_NAME = "sgl"
        fwd._control_token = "tok123"
        status, body = self._post("?token=tok123&action=destroy")
        self.assertIn("400", status)
        self.assertFalse(body["ok"])
        self.assertEqual(body["error"], "unknown action")

    def test_start_success(self):
        fwd.WSL_DISTRO = "Ubuntu-24.04"
        fwd.CONTAINER_NAME = "sgl"
        fwd._control_token = "tok123"
        # _run_wsl returns "running" for the docker start AND for the
        # docker inspect that _poll_container_status issues.
        with mock.patch.object(fwd, "_run_wsl",
                               return_value=_completed("running\n")):
            status, body = self._post("?token=tok123&action=start")
        self.assertIn("200", status)
        self.assertTrue(body["ok"])
        self.assertEqual(body["action"], "start")
        self.assertEqual(body["status"], "running")

    def test_stop_success(self):
        fwd.WSL_DISTRO = "Ubuntu-24.04"
        fwd.CONTAINER_NAME = "sgl"
        fwd._control_token = "tok123"
        with mock.patch.object(fwd, "_run_wsl",
                               return_value=_completed("exited\n")):
            status, body = self._post("?token=tok123&action=stop")
        self.assertIn("200", status)
        self.assertTrue(body["ok"])
        self.assertEqual(body["action"], "stop")

    def test_restart_success(self):
        fwd.WSL_DISTRO = "Ubuntu-24.04"
        fwd.CONTAINER_NAME = "sgl"
        fwd._control_token = "tok123"
        with mock.patch.object(fwd, "_run_wsl",
                               return_value=_completed("running\n")):
            status, body = self._post("?token=tok123&action=restart")
        self.assertIn("200", status)
        self.assertTrue(body["ok"])
        self.assertEqual(body["action"], "restart")

    def test_start_calls_docker(self):
        fwd.WSL_DISTRO = "Ubuntu-24.04"
        fwd.CONTAINER_NAME = "sgl"
        fwd._control_token = "tok123"
        calls: list[list[str]] = []

        def fake_wsl(args, timeout=30.0):
            calls.append(args)
            return _completed("running\n")

        with mock.patch.object(fwd, "_run_wsl", side_effect=fake_wsl):
            self._post("?token=tok123&action=start")

        # Two calls: docker start, then docker inspect (from _poll_container_status)
        self.assertEqual(len(calls), 2)
        args = calls[0]
        self.assertIn("docker", args)
        self.assertIn("start", args)
        self.assertIn("sgl", args)
        self.assertIn("Ubuntu-24.04", args)


class ControlActionTests(ForwarderCase):
    """_control_action: the WSL docker seam."""

    def test_no_flags(self):
        self.assertEqual(fwd._control_action("start"), "error")

    def test_invalid_action(self):
        fwd.WSL_DISTRO = "Ubuntu-24.04"
        fwd.CONTAINER_NAME = "sgl"
        self.assertEqual(fwd._control_action("destroy"), "error")

    def test_start_calls_docker_and_polls(self):
        fwd.WSL_DISTRO = "Ubuntu-24.04"
        fwd.CONTAINER_NAME = "sgl"
        calls: list[list[str]] = []

        def fake_wsl(args, timeout=30.0):
            calls.append(args)
            return _completed("running\n")

        with mock.patch.object(fwd, "_run_wsl", side_effect=fake_wsl):
            result = fwd._control_action("start")

        # Two calls: docker start, then docker inspect (from _poll_container_status)
        self.assertEqual(len(calls), 2)
        self.assertIn("start", calls[0])
        self.assertIn("inspect", calls[1])
        self.assertEqual(result, "running")


# ----------------------------------------------------------------
# 4. Lane identity: name and peers
# ----------------------------------------------------------------

class LaneIdentityTests(ForwarderCase):
    """snapshot() exposes name and peers."""

    def test_name_and_peers_in_snapshot(self):
        fwd.FWD_NAME = "lane-a"
        fwd.PEERS = [9890, 9891]
        d = stats.snapshot(fwd, fwd._stats)
        self.assertEqual(d["name"], "lane-a")
        self.assertEqual(d["peers"], [9890, 9891])

    def test_name_and_peers_default_none(self):
        # reset_state set FWD_NAME to None, PEERS to []
        d = stats.snapshot(fwd, fwd._stats)
        self.assertIsNone(d["name"])
        self.assertEqual(d["peers"], [])

    def test_peers_returns_a_copy(self):
        fwd.PEERS = [1]
        d = stats.snapshot(fwd, fwd._stats)
        d["peers"].append(99)
        self.assertEqual(fwd.PEERS, [1])


class MainFlagsTests(ForwarderCase):
    """main() sets FWD_NAME, PEERS, and _control_token from CLI args."""

    def test_control_token_generated(self):
        with mock.patch.object(fwd, "threading"), \
             mock.patch("omp_forwarder.forwarder._serve_and_cleanup"), \
             mock.patch("omp_forwarder.forwarder.discover",
                        return_value=None), \
             mock.patch("omp_forwarder.forwarder._sample_tokens"):
            rc = fwd.main(["--upstream-port", str(free_port())])
            self.assertEqual(rc, 0)
        self.assertIsInstance(fwd._control_token, str)
        self.assertEqual(len(fwd._control_token), 32)  # 16 bytes hex

    def test_name_flag(self):
        with mock.patch.object(fwd, "threading"), \
             mock.patch("omp_forwarder.forwarder._serve_and_cleanup"), \
             mock.patch("omp_forwarder.forwarder.discover",
                        return_value=None), \
             mock.patch("omp_forwarder.forwarder._sample_tokens"):
            fwd.main(["--name", "lane-b",
                      "--upstream-port", str(free_port())])
        self.assertEqual(fwd.FWD_NAME, "lane-b")

    def test_peer_flags(self):
        with mock.patch.object(fwd, "threading"), \
             mock.patch("omp_forwarder.forwarder._serve_and_cleanup"), \
             mock.patch("omp_forwarder.forwarder.discover",
                        return_value=None), \
             mock.patch("omp_forwarder.forwarder._sample_tokens"):
            fwd.main(["--peer", "9890", "--peer", "9891",
                      "--upstream-port", str(free_port())])
        self.assertEqual(fwd.PEERS, [9890, 9891])

    def test_control_token_is_fresh_per_call(self):
        with mock.patch.object(fwd, "threading"), \
             mock.patch("omp_forwarder.forwarder._serve_and_cleanup"), \
             mock.patch("omp_forwarder.forwarder.discover",
                        return_value=None), \
             mock.patch("omp_forwarder.forwarder._sample_tokens"):
            fwd.main(["--upstream-port", str(free_port())])
            tok1 = fwd._control_token
            fwd.main(["--upstream-port", str(free_port())])
            tok2 = fwd._control_token
        self.assertNotEqual(tok1, tok2)
