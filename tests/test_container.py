"""Container-upstream mode: polling, auto-start throttle, keepalive, validation,
and dashboard snapshot. All subprocess calls are replaced with recorded
output, so these run on any platform."""
from __future__ import annotations

import subprocess
import time
from unittest import mock

from omp_forwarder import forwarder as fwd
from omp_forwarder import stats

from .helpers import ForwarderCase, free_port


def _completed(stdout: str, rc: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=rc,
                                       stdout=stdout, stderr="")


class PollContainerStatusTests(ForwarderCase):
    """_poll_container_status: what ends up in _container_status."""

    def test_running(self):
        fwd.WSL_DISTRO = "Ubuntu-24.04"
        fwd.CONTAINER_NAME = "sgl"
        with mock.patch.object(fwd, "_run_wsl",
                               return_value=_completed("running\n")):
            fwd._poll_container_status()
        self.assertEqual(fwd._container_status, "running")

    def test_exited(self):
        fwd.WSL_DISTRO = "Ubuntu-24.04"
        fwd.CONTAINER_NAME = "sgl"
        with mock.patch.object(fwd, "_run_wsl",
                               return_value=_completed("exited\n")):
            fwd._poll_container_status()
        self.assertEqual(fwd._container_status, "exited")

    def test_error(self):
        fwd.WSL_DISTRO = "Ubuntu-24.04"
        fwd.CONTAINER_NAME = "sgl"
        with mock.patch.object(fwd, "_run_wsl",
                               return_value=_completed("error\n")):
            fwd._poll_container_status()
        self.assertEqual(fwd._container_status, "error")

    def test_none_from_run(self):
        fwd.WSL_DISTRO = "Ubuntu-24.04"
        fwd.CONTAINER_NAME = "sgl"
        with mock.patch.object(fwd, "_run_wsl", return_value=None):
            fwd._poll_container_status()
        self.assertEqual(fwd._container_status, "error")

    def test_empty_stdout_is_error(self):
        fwd.WSL_DISTRO = "Ubuntu-24.04"
        fwd.CONTAINER_NAME = "sgl"
        with mock.patch.object(fwd, "_run_wsl", return_value=_completed("")):
            fwd._poll_container_status()
        self.assertEqual(fwd._container_status, "error")

    def test_noop_when_flags_not_set(self):
        # reset_state already set them to None
        with mock.patch.object(fwd, "_run_wsl",
                               return_value=_completed("running")) as fake:
            fwd._poll_container_status()
            fake.assert_not_called()


class AutoStartTests(ForwarderCase):
    """Auto-start fires once per minute, only when exited and no upstream."""

    def _setup(self):
        fwd.WSL_DISTRO = "Ubuntu-24.04"
        fwd.CONTAINER_NAME = "sgl"
        fwd._upstream = None
        fwd._container_last_start = 0.0

    def test_calls_docker_start_when_exited_and_no_upstream(self):
        self._setup()
        calls: list[list[str]] = []

        def fake_wsl(args, timeout=10.0):
            calls.append(args)
            # First call is inspect, second is start
            if len(calls) == 1:
                return _completed("exited\n")
            return _completed("", rc=0)

        with mock.patch.object(fwd, "_run_wsl", side_effect=fake_wsl):
            fwd._poll_container_status()

        # First call: docker inspect; second: docker start
        self.assertEqual(len(calls), 2)
        self.assertIn("inspect", calls[0])
        self.assertIn("start", calls[1])

    def test_does_not_call_start_when_running(self):
        self._setup()
        calls: list[list[str]] = []

        def fake_wsl(args, timeout=10.0):
            calls.append(args)
            return _completed("running\n")

        with mock.patch.object(fwd, "_run_wsl", side_effect=fake_wsl):
            fwd._poll_container_status()

        # Only the inspect call, no start
        self.assertEqual(len(calls), 1)
        self.assertIn("inspect", calls[0])

    def test_does_not_call_start_when_upstream_is_healthy(self):
        self._setup()
        up = free_port()
        fwd._upstream = up
        calls: list[list[str]] = []

        def fake_wsl(args, timeout=10.0):
            calls.append(args)
            return _completed("exited\n")

        with mock.patch.object(fwd, "_run_wsl", side_effect=fake_wsl):
            fwd._poll_container_status()

        # Only the inspect call, no start
        self.assertEqual(len(calls), 1)
        self.assertIn("inspect", calls[0])

    def test_does_not_call_start_within_one_minute(self):
        self._setup()
        # Pretend a start happened just now
        fwd._container_last_start = time.time() - 30  # 30 s ago
        calls: list[list[str]] = []

        def fake_wsl(args, timeout=10.0):
            calls.append(args)
            return _completed("exited\n")

        with mock.patch.object(fwd, "_run_wsl", side_effect=fake_wsl):
            fwd._poll_container_status()

        # Only the inspect call
        self.assertEqual(len(calls), 1)

    def test_calls_start_after_one_minute_has_elapsed(self):
        self._setup()
        # Pretend a start happened 61 s ago
        fwd._container_last_start = time.time() - 61
        calls: list[list[str]] = []

        def fake_wsl(args, timeout=10.0):
            calls.append(args)
            if "start" in " ".join(args):
                return _completed("")
            return _completed("exited\n")

        with mock.patch.object(fwd, "_run_wsl", side_effect=fake_wsl):
            fwd._poll_container_status()

        # inspect + start
        self.assertEqual(len(calls), 2)
        self.assertIn("start", calls[1])

    def test_start_sets_last_start(self):
        self._setup()
        fwd._container_last_start = 0.0
        with mock.patch.object(fwd, "_run_wsl",
                               side_effect=lambda *a, **k: _completed("exited\n")):
            fwd._poll_container_status()
        self.assertGreater(fwd._container_last_start, 0)


class KeepaliveTests(ForwarderCase):
    """_container_keepalive_start / _container_keepalive_stop."""

    def test_start_and_stop(self):
        fwd.WSL_DISTRO = "Ubuntu-24.04"
        with mock.patch.object(fwd, "_spawn_wsl",
                                return_value=mock.Mock()) as fake_spawn:
            fwd._container_keepalive_start()
            fake_spawn.assert_called_once()
            args = fake_spawn.call_args[0][0]
            self.assertEqual(args[0], "wsl.exe")
            self.assertIn("-d", args)
            self.assertIn("Ubuntu-24.04", args)
            self.assertIn("sleep", args)

            self.assertIsNotNone(fwd._container_keepalive)
            fwd._container_keepalive_stop()
            self.assertIsNone(fwd._container_keepalive)

    def test_stop_when_none_is_noop(self):
        # reset_state already set _container_keepalive to None
        fwd._container_keepalive_stop()  # must not raise

    def test_keepalive_stopped_even_when_serve_raises(self):
        """_serve_and_cleanup runs _container_keepalive_stop() in a finally,
        so an exception from the serve step still cleans up the child."""
        fwd.WSL_DISTRO = "Ubuntu-24.04"
        stopped = []

        def fake_stop():
            stopped.append(1)

        def fake_serve():
            raise RuntimeError("simulate serve crash")

        with mock.patch.object(fwd, "_container_keepalive_stop",
                              side_effect=fake_stop) as fake_stop_m:
            try:
                fwd._serve_and_cleanup(fake_serve)
            except RuntimeError:
                pass
            fake_stop_m.assert_called_once()


class MainValidationTests(ForwarderCase):
    """main() rejects the flags when only one of the two is given."""

    def test_wsl_distro_without_container(self):
        rc = fwd.main(["--wsl-distro", "Ubuntu-24.04"])
        self.assertEqual(rc, 1)

    def test_container_without_wsl_distro(self):
        rc = fwd.main(["--container", "sgl"])
        self.assertEqual(rc, 1)


class SnapshotContainerTests(ForwarderCase):
    """snapshot() exposes container and container_name in the JSON."""

    def test_container_fields_present_when_set(self):
        fwd.CONTAINER_NAME = "sgl"
        fwd._container_status = "running"
        fwd._upstream = None
        d = stats.snapshot(fwd, fwd._stats)
        self.assertEqual(d["container"], "running")
        self.assertEqual(d["container_name"], "sgl")

    def test_container_fields_none_when_unset(self):
        # reset_state already set them to None
        fwd._upstream = None
        d = stats.snapshot(fwd, fwd._stats)
        self.assertIsNone(d["container"])
        self.assertIsNone(d["container_name"])
