"""Unit tests for the app/engine JSON-lines protocol, with a fake engine."""

import io
import json
import os
import sys
import unittest
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from acestep_mlx.protocol import Emitter, ProtocolError, parse_request, serve  # noqa: E402


class FakeEngine:
    """Records requests; reports every phase; fails on the prompt "boom"."""

    def __init__(self):
        """Start with no recorded calls."""
        self.calls = []

    def generate(self, request_id, prompt, report_phase):
        """Return a minimal song record, or raise for "boom"."""
        self.calls.append((request_id, prompt))
        if prompt == "boom":
            raise RuntimeError("synthetic failure")
        for phase in ("lyrics", "composing", "composing", "rendering", "composing"):
            report_phase(phase)
        return {"prompt": prompt, "audio": "x.flac"}


def run(lines):
    """Serve ``lines`` with a FakeEngine; returns (decoded events, engine)."""
    out, engine = io.StringIO(), FakeEngine()
    serve(engine, iter(lines), Emitter(out), load_s=1.5)
    return [json.loads(line) for line in out.getvalue().splitlines()], engine


def gen(request_id, prompt):
    """A generate request line."""
    return json.dumps({"id": request_id, "cmd": "generate", "prompt": prompt}) + "\n"


class ParseRequestTests(unittest.TestCase):
    """parse_request accepts the two commands and rejects everything else."""

    def test_generate_and_shutdown_parse(self):
        """Valid requests come back as dicts."""
        self.assertEqual(parse_request(gen("a", "song"))["prompt"], "song")
        self.assertEqual(parse_request('{"cmd": "shutdown"}')["cmd"], "shutdown")

    def test_invalid_requests_raise(self):
        """Bad JSON, non-objects, unknown commands, missing id or blank prompt."""
        for line in ("{", "[]", '{"cmd": "dance"}', '{"cmd": "generate", "prompt": "x"}',
                     '{"cmd": "generate", "id": "a", "prompt": "   "}'):
            with self.assertRaises(ProtocolError, msg=line):
                parse_request(line)


class ServeTests(unittest.TestCase):
    """serve announces readiness, answers in order and survives failures."""

    def test_ready_then_phases_then_done(self):
        """Phases are emitted once each, in order, never going back."""
        events, _ = run([gen("r1", "  folk ballad  ")])
        self.assertEqual(events[0], {"event": "ready", "protocol": 1, "load_s": 1.5})
        self.assertEqual([e["phase"] for e in events if e["event"] == "phase"],
                         ["lyrics", "composing", "rendering"])
        self.assertEqual(events[-1]["event"], "done")
        self.assertEqual(events[-1]["id"], "r1")
        self.assertEqual(events[-1]["song"]["prompt"], "folk ballad")

    def test_failure_is_reported_and_serving_continues(self):
        """An engine exception becomes an error event for that id; the next request runs."""
        events, engine = run([gen("r1", "boom"), "not json\n", "\n", gen("r2", "ok")])
        errors = [e for e in events if e["event"] == "error"]
        self.assertEqual(errors[0]["id"], "r1")
        self.assertIn("synthetic failure", errors[0]["message"])
        self.assertIsNone(errors[1]["id"])
        self.assertEqual(events[-1]["event"], "done")
        self.assertEqual([c[0] for c in engine.calls], ["r1", "r2"])

    def test_shutdown_stops_before_later_requests(self):
        """Requests after shutdown are not served."""
        _, engine = run([gen("r1", "a"), '{"cmd": "shutdown"}\n', gen("r2", "b")])
        self.assertEqual([c[0] for c in engine.calls], ["r1"])

    def test_non_ascii_survives(self):
        """Lyrics in Italian round-trip unescaped."""
        events, _ = run([gen("r1", "ballata però")])
        self.assertEqual(events[-1]["song"]["prompt"], "ballata però")


class SongDirNameTests(unittest.TestCase):
    """song_dir_name sorts by time; imported lazily to keep this file model-free."""

    def test_format(self):
        """Timestamp then the first 8 characters of the id."""
        from acestep_mlx.engine_service import song_dir_name

        name = song_dir_name("0123456789abcdef", datetime(2026, 10, 23, 20, 5, 9))
        self.assertEqual(name, "20261023-200509-01234567")


if __name__ == "__main__":
    unittest.main()
