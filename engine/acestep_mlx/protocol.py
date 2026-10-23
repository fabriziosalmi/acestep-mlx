"""JSON-lines protocol between the macOS app and the engine process.

One JSON object per line in each direction. Requests (stdin):
    {"id": "<uuid>", "cmd": "generate", "prompt": "<description>", "options": {...}}
    {"cmd": "shutdown"}
Events (stdout):
    {"event": "ready", "protocol": 1, "load_s": 31.2}
    {"event": "phase", "id": ..., "phase": "lyrics" | "composing" | "rendering"}
    {"event": "done", "id": ..., "song": {...}}            # see engine_service.Song
    {"event": "error", "id": ... | null, "message": "..."}
``options`` is optional; its keys are validated by ``recipe.parse_options``.
The engine handles one request at a time; the app does not send a new one before
``done`` or ``error``.
"""

import json
import threading
from typing import Any, Callable, Dict, Optional, TextIO

PROTOCOL_VERSION = 1
PHASES = ("lyrics", "composing", "rendering")


class ProtocolError(ValueError):
    """A request line that is not a valid request."""


def parse_request(line: str) -> Dict[str, Any]:
    """Decode and validate one request line.

    Raises:
        ProtocolError: For invalid JSON, an unknown command or a missing field.
    """
    try:
        req = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"not JSON: {exc.msg}") from exc
    if not isinstance(req, dict):
        raise ProtocolError("request must be a JSON object")
    cmd = req.get("cmd")
    if cmd == "shutdown":
        return req
    if cmd != "generate":
        raise ProtocolError(f"unknown cmd {cmd!r}")
    if not isinstance(req.get("id"), str) or not req["id"]:
        raise ProtocolError("generate needs a non-empty string id")
    prompt = req.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ProtocolError("generate needs a non-empty prompt")
    if not isinstance(req.get("options", {}), dict):
        raise ProtocolError("options must be an object")
    return req


class Emitter:
    """Writes events as JSON lines; safe to call from progress threads."""

    def __init__(self, stream: TextIO):
        """``stream`` is the protocol channel (the process's original stdout)."""
        self._stream = stream
        self._lock = threading.Lock()

    def emit(self, event: str, **fields: Any) -> None:
        """Write ``{"event": event, **fields}`` and flush."""
        line = json.dumps({"event": event, **fields}, ensure_ascii=False)
        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()

    def phase_reporter(self, request_id: str) -> Callable[[str], None]:
        """A callback that emits each phase once, in order, for one request."""
        sent = []

        def report(phase: str) -> None:
            if phase not in PHASES:
                raise ValueError(f"unknown phase {phase!r}")
            if not sent or PHASES.index(phase) > PHASES.index(sent[-1]):
                sent.append(phase)
                self.emit("phase", id=request_id, phase=phase)

        return report


def serve(engine, lines, emitter: Emitter, load_s: Optional[float] = None) -> None:
    """Announce readiness, then answer requests from ``lines`` until shutdown or EOF.

    ``engine.generate(request_id, prompt, options, report_phase)`` returns a
    JSON-serializable song dict.
    A failing request produces an ``error`` event and the loop keeps serving.
    """
    emitter.emit("ready", protocol=PROTOCOL_VERSION, load_s=load_s)
    for raw in lines:
        if not raw.strip():
            continue
        try:
            req = parse_request(raw)
        except ProtocolError as exc:
            emitter.emit("error", id=None, message=str(exc))
            continue
        if req["cmd"] == "shutdown":
            return
        try:
            song = engine.generate(
                req["id"], req["prompt"].strip(), req.get("options") or {},
                emitter.phase_reporter(req["id"]),
            )
        except Exception as exc:  # noqa: BLE001 - reported to the app, loop continues
            emitter.emit("error", id=req["id"], message=f"{type(exc).__name__}: {exc}")
            continue
        emitter.emit("done", id=req["id"], song=song)
