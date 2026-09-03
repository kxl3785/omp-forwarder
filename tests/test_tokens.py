"""The token tally: the forwarder's own count of prompt and generated tokens,
folded from successive /metrics samples so that it survives model reloads.

The property under test: the tally must equal the tokens produced on the
forwarder's watch, whether those came from one llama-server process or from
several in a row."""
from __future__ import annotations

import json
import os
import tempfile
from unittest import mock

from omp_forwarder import forwarder as fwd
from omp_forwarder import stats

from .helpers import ForwarderCase, RelayCase, http_response, raw_request


def m(prompt: float, gen: float) -> dict:
    return {"llamacpp:prompt_tokens_total": float(prompt),
            "llamacpp:tokens_predicted_total": float(gen)}


def tally() -> tuple[int, int]:
    return fwd._stats["tok_prompt"], fwd._stats["tok_gen"]


class TallyTests(ForwarderCase):

    def test_baseline_records_without_adding(self):
        fwd._tally_tokens(5000, m(100, 50), baseline=True)
        self.assertEqual(tally(), (0, 0))
        self.assertEqual(fwd._tok_last, (5000, 100.0, 50.0))

    def test_adds_deltas_on_the_same_port(self):
        fwd._tally_tokens(5000, m(100, 50), baseline=True)
        fwd._tally_tokens(5000, m(160, 80))
        self.assertEqual(tally(), (60, 30))
        fwd._tally_tokens(5000, m(160, 80))         # nothing new: no change
        self.assertEqual(tally(), (60, 30))
        fwd._tally_tokens(5000, m(200, 90))
        self.assertEqual(tally(), (100, 40))

    def test_a_new_port_is_a_new_process_and_counts_whole(self):
        fwd._tally_tokens(5000, m(100, 50), baseline=True)
        fwd._tally_tokens(5000, m(120, 60))
        # Studio reloaded the model: new process, new port, counters from 0.
        fwd._tally_tokens(6000, m(30, 10))
        self.assertEqual(tally(), (20 + 30, 10 + 10))
        fwd._tally_tokens(6000, m(45, 12))
        self.assertEqual(tally(), (65, 22))

    def test_counters_going_backwards_on_the_same_port_count_whole(self):
        fwd._tally_tokens(5000, m(100, 50), baseline=True)
        fwd._tally_tokens(5000, m(20, 5))           # restarted on the same port
        self.assertEqual(tally(), (20, 5))

    def test_first_server_seen_with_no_baseline_counts_whole(self):
        # No llama-server ran when the forwarder started, so the first one it
        # sees was started on its watch: all of its work is ours.
        fwd._tally_tokens(5000, m(30, 10))
        self.assertEqual(tally(), (30, 10))

    def test_no_port_or_no_metrics_is_a_noop(self):
        fwd._tally_tokens(None, m(30, 10))
        fwd._tally_tokens(5000, {})
        self.assertEqual(tally(), (0, 0))
        self.assertIsNone(fwd._tok_last)

    def test_missing_counters_read_as_zero(self):
        fwd._tally_tokens(5000, {"llamacpp:n_decode_total": 3.0})
        self.assertEqual(tally(), (0, 0))
        self.assertEqual(fwd._tok_last, (5000, 0.0, 0.0))


class DayTotalsTests(ForwarderCase):
    """Per-day totals: they follow the tally, roll over at the date change,
    and survive a restart through TOKENS_FILE."""

    def test_day_totals_follow_the_tally(self):
        with mock.patch.object(fwd, "_today", return_value="2026-09-03"):
            fwd._tally_tokens(5000, m(100, 50), baseline=True)
            fwd._tally_tokens(5000, m(160, 80))
            fwd._tally_tokens(5000, m(160, 80))     # no change: no entry churn
            fwd._tally_tokens(5000, m(200, 90))
            self.assertEqual(fwd._day_totals,
                             {"2026-09-03": {"prompt": 100, "gen": 40}})
            self.assertEqual(fwd._today_tokens(), (100, 40))
        self.assertTrue(fwd._days_dirty)

    def test_baseline_touches_no_day(self):
        with mock.patch.object(fwd, "_today", return_value="2026-09-03"):
            fwd._tally_tokens(5000, m(100, 50), baseline=True)
        self.assertEqual(fwd._day_totals, {})
        self.assertFalse(fwd._days_dirty)

    def test_rollover_logs_the_finished_day_and_starts_a_new_one(self):
        with mock.patch.object(fwd, "_today", return_value="2026-09-03"):
            fwd._tally_tokens(5000, m(100, 50), baseline=True)
            fwd._tally_tokens(5000, m(130, 60))
        with mock.patch.object(fwd, "_today", return_value="2026-09-04"), \
                mock.patch.object(fwd, "log") as logged:
            fwd._tally_tokens(5000, m(135, 62))
        self.assertEqual(fwd._day_totals, {
            "2026-09-03": {"prompt": 30, "gen": 10},
            "2026-09-04": {"prompt": 5, "gen": 2}})
        msgs = [c.args[0] for c in logged.call_args_list]
        self.assertEqual(msgs, ["tokens 2026-09-03: 30 prompt, 10 generated"])

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            fwd.TOKENS_FILE = os.path.join(tmp, "sub", "tokens.json")
            fwd._day_totals = {"2026-09-02": {"prompt": 7, "gen": 3},
                               "2026-09-03": {"prompt": 100, "gen": 40}}
            fwd._days_dirty = True
            fwd._save_days()
            self.assertFalse(fwd._days_dirty)
            self.assertFalse(os.path.exists(fwd.TOKENS_FILE + ".tmp"))
            with open(fwd.TOKENS_FILE, encoding="utf-8") as fh:
                on_disk = json.load(fh)
            self.assertEqual(on_disk["2026-09-03"], {"prompt": 100, "gen": 40})
            fwd._day_totals = {}
            fwd._load_days()
            self.assertEqual(fwd._day_totals, on_disk)

    def test_load_tolerates_a_missing_or_corrupt_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            fwd.TOKENS_FILE = os.path.join(tmp, "tokens.json")
            fwd._load_days()
            self.assertEqual(fwd._day_totals, {})
            with open(fwd.TOKENS_FILE, "w", encoding="utf-8") as fh:
                fh.write("{not json")
            fwd._load_days()
            self.assertEqual(fwd._day_totals, {})
            with open(fwd.TOKENS_FILE, "w", encoding="utf-8") as fh:
                json.dump({"2026-09-03": {"prompt": "12", "gen": 4},
                           "junk": 5}, fh)
            fwd._load_days()
            self.assertEqual(fwd._day_totals,
                             {"2026-09-03": {"prompt": 12, "gen": 4}})

    def test_no_tokens_file_means_no_disk_io(self):
        fwd.TOKENS_FILE = None
        fwd._day_totals = {"2026-09-03": {"prompt": 1, "gen": 1}}
        fwd._days_dirty = True
        fwd._save_days()                       # must not raise
        fwd._load_days()
        self.assertEqual(fwd._day_totals, {"2026-09-03": {"prompt": 1, "gen": 1}})

    def test_default_path_sits_beside_the_log(self):
        with mock.patch.dict(os.environ, {"OMP_FORWARDER_LOG":
                                          os.path.join("x", "logs", "f.log")}):
            self.assertEqual(fwd._default_tokens_file(),
                             os.path.join("x", "logs", "tokens.json"))


class SampleTests(ForwarderCase):

    def _metrics_fake(self, state: dict):
        def responder(method, path, raw):
            if path == "/metrics":
                body = (f"llamacpp:prompt_tokens_total {state['p']}\n"
                        f"llamacpp:tokens_predicted_total {state['g']}\n")
                return [http_response(200, body.encode())]
            return [http_response(404)]
        return self.fake(responder)

    def test_sample_reads_metrics_from_the_upstream(self):
        state = {"p": 1000, "g": 400}
        up = self._metrics_fake(state)
        fwd._upstream = up.port
        fwd._sample_tokens(baseline=True)
        self.assertEqual(tally(), (0, 0))
        state.update(p=1500, g=600)
        fwd._sample_tokens()
        self.assertEqual(tally(), (500, 200))

    def test_sample_without_an_upstream_is_a_noop(self):
        fwd._upstream = None
        fwd._sample_tokens()
        self.assertEqual(tally(), (0, 0))
        self.assertIsNone(fwd._tok_last)

    def test_snapshot_folds_its_sample_and_reports_the_tally(self):
        state = {"p": 70, "g": 30}
        up = self._metrics_fake(state)
        fwd._upstream = up.port
        d = stats.snapshot(fwd, fwd._stats)
        self.assertEqual((d["tok_prompt"], d["tok_gen"]), (70, 30))
        state.update(p=100, g=45)
        d = stats.snapshot(fwd, fwd._stats)
        self.assertEqual((d["tok_prompt"], d["tok_gen"]), (100, 45))
        self.assertEqual(tally(), (100, 45))
        # No baseline was taken, so everything counted today.
        self.assertEqual((d["tok_today_prompt"], d["tok_today_gen"]), (100, 45))

    def test_page_has_the_card_and_baselines_it(self):
        self.assertIn('id="tok"', stats.PAGE)
        self.assertIn('"tok_prompt","tok_gen"', stats.PAGE)


class RelayMarksDirtyTests(RelayCase):

    def test_completed_relay_wakes_the_sampler(self):
        up = self.fake(lambda meth, path, raw: [http_response(200, b"ok")])
        fwd.FORCED_UPSTREAM = up.port
        self.assertFalse(fwd._tok_dirty.is_set())
        raw_request(self.port, [b"GET /v1/models HTTP/1.1\r\n\r\n"])
        self.wait_done(1)
        self.assertTrue(fwd._tok_dirty.is_set())
