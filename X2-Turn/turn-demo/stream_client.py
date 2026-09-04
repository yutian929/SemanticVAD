#!/usr/bin/env python3
"""Standalone streaming client for the X2-Turn browser demo server.

The X2-Turn model runs on the server (``demo_turn.server``, which exposes a
``/ws/stream`` WebSocket). This client runs on your own machine, opens that
socket over an SSH tunnel, streams 16 kHz mono PCM (from the microphone or a
WAV file), and renders the incremental ASR + turn states in a live terminal
dashboard.

    Server (GPU box):   cd turn-demo && MODEL=... bash run.sh   # binds 127.0.0.1:7860
    Tunnel (client):    ssh -N -L 7860:127.0.0.1:7860 user@server
    Client (this):      python stream_client.py --wav assets/sample_en.wav
                        python stream_client.py --mic

Wire protocol (see demo_turn/server.py ws_stream):
    -> {"type":"start","commit_ms":320}     text
    <- {"type":"ready","backend":"hf"}       text
    -> <int16 LE PCM bytes @16kHz mono>      binary, streamed in chunks
    <- {"type":"update", asr_text, turns[], turn_hist, last_turn, ...}  every commit
    -> {"type":"stop"}                       text
    <- {"type":"final", ...}                 text

The six X2-Turn turn states are idle / noidle / speaking / turn_end /
backchannel / uncertain (this differs from SoulX-Duplug's label set).

Client deps:  pip install -r requirements-client.txt
(``sounddevice`` is only needed for --mic; WAV replay does not import it.)
"""
from __future__ import annotations

import argparse
import json
import queue
import signal
import sys
import threading
import time
from typing import Dict, List, Optional

import numpy as np

# --- X2-Turn turn-state palette (mirrors demo_turn/viz.py TURN_COLOR) ---------
TURN_COLOR: Dict[str, str] = {
    "idle": "#9ca3af",        # gray
    "noidle": "#60a5fa",      # light blue
    "speaking": "#3b82f6",    # blue
    "turn_end": "#22c55e",    # green
    "backchannel": "#a855f7", # purple
    "uncertain": "#eab308",   # yellow
}
TURN_ORDER = ["idle", "noidle", "speaking", "turn_end", "backchannel", "uncertain"]

TARGET_SR = 16000


# ----------------------------------------------------------------- shared state
class SharedState:
    """Latest server update, guarded by a lock; read by the render loop."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.status = "connecting"
        self.backend = "?"
        self.asr_text = ""
        self.turns: List[str] = []
        self.turn_hist: Dict[str, int] = {}
        self.last_turn = "idle"
        self.n_frames = 0
        self.duration_s = 0.0
        self.infer_ms = 0.0
        self.sent_s = 0.0
        self.finished = False
        self.error: Optional[str] = None

    def apply(self, data: dict) -> None:
        with self.lock:
            if "error" in data:
                self.error = str(data["error"])
                return
            typ = data.get("type")
            if typ == "ready":
                self.status = "streaming"
                self.backend = data.get("backend", "?")
                return
            # update / final
            self.asr_text = data.get("asr_text", self.asr_text)
            self.turns = data.get("turns", self.turns) or self.turns
            self.turn_hist = data.get("turn_hist", self.turn_hist) or self.turn_hist
            self.last_turn = data.get("last_turn", self.last_turn)
            self.n_frames = data.get("n_frames", self.n_frames)
            self.duration_s = data.get("duration_s", self.duration_s)
            self.infer_ms = data.get("elapsed_infer_ms", self.infer_ms)
            if typ == "final":
                self.status = "final"
                self.finished = True


# ------------------------------------------------------------------ rendering
def build_dashboard(st: SharedState):
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    with st.lock:
        status = st.status
        backend = st.backend
        asr = st.asr_text
        turns = list(st.turns)
        hist = dict(st.turn_hist)
        last_turn = st.last_turn
        n_frames = st.n_frames
        duration_s = st.duration_s
        infer_ms = st.infer_ms
        sent_s = st.sent_s
        error = st.error

    # header ------------------------------------------------------------------
    head = Text()
    dot = {"streaming": "green", "final": "cyan", "connecting": "yellow"}.get(status, "red")
    head.append("● ", style=dot)
    head.append(f"{status}", style="bold")
    head.append(f"   backend={backend}", style="dim")
    head.append(f"   sent={sent_s:5.1f}s", style="dim")
    head.append(f"   decoded={duration_s:5.2f}s", style="dim")
    head.append(f"   frames={n_frames}", style="dim")
    head.append(f"   infer={infer_ms:6.1f}ms", style="dim")

    # current turn state (big, colored) --------------------------------------
    color = TURN_COLOR.get(last_turn, "#ffffff")
    cur = Text()
    cur.append("  CURRENT TURN  ", style="bold white on black")
    cur.append("   ")
    cur.append(f" {last_turn.upper()} ", style=f"bold white on {color}")

    # ASR --------------------------------------------------------------------
    asr_panel = Panel(
        Text(asr or "(waiting for speech…)", style="bold"),
        title="ASR",
        title_align="left",
        border_style="dim",
    )

    # timeline: colored blocks, one per 80 ms frame --------------------------
    legend = Text()
    for name in TURN_ORDER:
        legend.append("█ ", style=TURN_COLOR[name])
        legend.append(f"{name}  ", style="dim")
    tl = Text()
    tail = turns[-160:]
    for t in tail:
        tl.append("█", style=TURN_COLOR.get(t, "#ffffff"))
    if not tail:
        tl.append("(no frames yet)", style="dim")
    timeline_panel = Panel(
        Group(legend, Text(""), tl),
        title=f"Timeline · 80 ms/frame · last {len(tail)} of {len(turns)}",
        title_align="left",
        border_style="dim",
    )

    # histogram --------------------------------------------------------------
    table = Table.grid(padding=(0, 1))
    table.add_column(justify="right", style="bold")
    table.add_column()
    table.add_column(justify="right", style="dim")
    total = max(sum(hist.values()), 1)
    for name in TURN_ORDER:
        n = hist.get(name, 0)
        bar_len = int(round(30 * n / total))
        bar = Text("█" * bar_len, style=TURN_COLOR[name])
        table.add_row(name, bar, str(n))

    body = Group(head, Text(""), cur, Text(""), asr_panel, timeline_panel,
                 Panel(table, title="Turn histogram", title_align="left", border_style="dim"))
    if error:
        body = Group(body, Text(f"\n[server error] {error}", style="bold red"))
    return Panel(body, title="X2-Turn · live stream", border_style=color)


# --------------------------------------------------------------- audio sources
def iter_wav_chunks(path: str, chunk_ms: int, speed: float):
    """Yield (int16 bytes, seconds_sent) from a WAV, paced to ~realtime*speed."""
    import soundfile as sf

    audio, sr = sf.read(path, always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sr != TARGET_SR:
        import soxr

        audio = soxr.resample(audio, sr, TARGET_SR)
    i16 = np.clip(audio, -1.0, 1.0)
    i16 = (i16 * 32767.0).astype("<i2")
    step = max(int(TARGET_SR * chunk_ms / 1000.0), 1)
    dt = (chunk_ms / 1000.0) / max(speed, 1e-6)
    for start in range(0, len(i16), step):
        chunk = i16[start:start + step]
        yield chunk.tobytes(), (start + len(chunk)) / TARGET_SR
        time.sleep(dt)


def open_mic_stream(out_q: "queue.Queue", chunk_ms: int, device: Optional[int]):
    """Open a 16 kHz mono int16 input stream that pushes raw byte chunks to out_q.

    Returns the started sounddevice stream; the caller closes it. Raises a clear
    RuntimeError if sounddevice/PortAudio is unavailable so the error surfaces in
    the dashboard instead of silently killing a background thread.
    """
    try:
        import sounddevice as sd
    except ImportError as e:
        raise RuntimeError(
            "microphone mode needs sounddevice: pip install sounddevice "
            "(and a working PortAudio + input device)"
        ) from e

    block = max(int(TARGET_SR * chunk_ms / 1000.0), 1)

    def callback(indata, frames, time_info, status):  # noqa: ARG001
        if status:
            print(f"[mic] {status}", file=sys.stderr)
        out_q.put(bytes(indata))  # int16 mono little-endian

    stream = sd.RawInputStream(samplerate=TARGET_SR, channels=1, dtype="int16",
                               blocksize=block, device=device, callback=callback)
    stream.start()
    return stream


# ------------------------------------------------------------------------ main
def run(args) -> int:
    import websocket  # websocket-client
    from rich.live import Live

    st = SharedState()
    ws = websocket.create_connection(args.url, timeout=10)
    ws.settimeout(1.0)
    send_lock = threading.Lock()
    stop_event = threading.Event()

    def ws_send_text(obj: dict) -> None:
        with send_lock:
            ws.send(json.dumps(obj))

    def ws_send_bytes(b: bytes) -> None:
        with send_lock:
            ws.send_binary(b)

    # receiver ---------------------------------------------------------------
    def recv_loop() -> None:
        while not stop_event.is_set():
            try:
                msg = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except Exception as e:  # noqa: BLE001
                # A recv that fails while we're already shutting down (socket
                # closed after the final result) is expected, not an error.
                if not stop_event.is_set():
                    st.apply({"error": f"recv: {e}"})
                break
            if not msg:
                continue
            if isinstance(msg, bytes):
                continue
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue
            st.apply(data)
            if data.get("type") == "final":
                break

    recv_thread = threading.Thread(target=recv_loop, daemon=True)
    recv_thread.start()

    ws_send_text({"type": "start", "commit_ms": args.commit_ms})
    # wait briefly for "ready"
    t0 = time.time()
    while time.time() - t0 < 10 and st.status == "connecting":
        time.sleep(0.05)

    # sender -----------------------------------------------------------------
    def sender_loop() -> None:
        try:
            if args.wav:
                for chunk, sent_s in iter_wav_chunks(args.wav, args.chunk_ms, args.speed):
                    if stop_event.is_set():
                        break
                    ws_send_bytes(chunk)
                    with st.lock:
                        st.sent_s = sent_s
                time.sleep(0.3)  # let the last commit come back
            else:
                q: "queue.Queue" = queue.Queue()
                stream = open_mic_stream(q, args.chunk_ms, args.device)  # may raise
                sent = 0
                try:
                    while not stop_event.is_set():
                        try:
                            chunk = q.get(timeout=0.2)
                        except queue.Empty:
                            continue
                        ws_send_bytes(chunk)
                        sent += len(chunk) // 2
                        with st.lock:
                            st.sent_s = sent / TARGET_SR
                finally:
                    stream.stop()
                    stream.close()
        except Exception as e:  # noqa: BLE001
            st.apply({"error": f"send: {e}"})
        finally:
            # ask the server for the flushed final result
            try:
                ws_send_text({"type": "stop"})
            except Exception:  # noqa: BLE001
                pass

    sender_thread = threading.Thread(target=sender_loop, daemon=True)
    sender_thread.start()

    # Ctrl-C stops the sender (mic mode); WAV mode ends on its own.
    def on_sigint(signum, frame):  # noqa: ARG001
        stop_event.set()

    signal.signal(signal.SIGINT, on_sigint)

    # render loop ------------------------------------------------------------
    from rich.console import Console

    console = Console()
    exit_code = 0
    with Live(build_dashboard(st), console=console, refresh_per_second=12,
              screen=False) as live:
        while True:
            live.update(build_dashboard(st))
            with st.lock:
                done = st.finished
                err = st.error
            if done or err:
                # one last refresh, then leave the final frame on screen
                time.sleep(0.2)
                live.update(build_dashboard(st))
                break
            if stop_event.is_set() and not sender_thread.is_alive() \
                    and not recv_thread.is_alive():
                break
            time.sleep(0.08)

    stop_event.set()
    try:
        ws.close()
    except Exception:  # noqa: BLE001
        pass
    with st.lock:
        if st.error:
            console.print(f"[bold red]server error:[/] {st.error}")
            exit_code = 1
    return exit_code


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Stream mic/WAV audio to the X2-Turn server and show live turn states.",
    )
    p.add_argument("--url", default="ws://localhost:7860/ws/stream",
                   help="WebSocket URL of the server's /ws/stream (via SSH tunnel).")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--wav", help="Stream this WAV file (any samplerate; resampled to 16k).")
    src.add_argument("--mic", action="store_true", help="Capture and stream the microphone.")
    p.add_argument("--commit-ms", type=int, default=320,
                   help="Server decode cadence; how often it returns an update.")
    p.add_argument("--chunk-ms", type=int, default=80,
                   help="Size of each audio chunk sent to the server.")
    p.add_argument("--speed", type=float, default=1.0,
                   help="WAV replay speed multiplier (1.0 = realtime).")
    p.add_argument("--device", type=int, default=None,
                   help="Input device index for --mic (see python -m sounddevice).")
    args = p.parse_args(argv)
    if not args.wav and not args.mic:
        p.error("choose an audio source: --wav PATH or --mic")
    return args


if __name__ == "__main__":
    sys.exit(run(parse_args()))
