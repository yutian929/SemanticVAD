# X2 Turn Demo

Follow the repository [Quick start](../README.md#quick-start) first. This page
is extra detail: flags, TLS, microphone upload, and the optional vLLM backend.

The demo shows streaming ASR, one six-class turn prediction every 80 ms, a
timeline and histogram, and a raw frame table. It does not start an LLM or TTS
service, and it does not turn frames into product actions.

The six labels are `idle`, `noidle`, `speaking`, `turn_end`, `backchannel`, and
`uncertain`.

## Local Transformers

This is the same path as the root Quick start. From the repository root,
activate `x2-turn`, then:

```bash
cd turn-demo
MODEL=x-square-robot/X2-Turn-4B-0812 bash run.sh

# After huggingface-cli download, or a private checkpoint:
MODEL=/path/to/X2-Turn-4B-0812 bash run.sh
```

Open <http://localhost:7860>. The 4B checkpoint needs about **24 GB+ VRAM**.
The model loads on the first **Run scenario**, upload, or microphone request,
not when the process starts. That first inference can take several minutes.

Choose **[built-in] English question** and click **Run scenario**. You do not
need a microphone. Typical numbers for that clip are in the root
[Quick start](../README.md#quick-start). Sample provenance is in
[`assets/README.md`](assets/README.md).

The server binds to `127.0.0.1` and has no authentication. For development TLS:

```bash
HOST=0.0.0.0 \
SSL_CERTFILE=/path/to/cert.pem \
SSL_KEYFILE=/path/to/key.pem \
bash run.sh
```

Do not expose uploaded speech on an untrusted network. Prefer an authenticated
HTTPS reverse proxy for anything beyond localhost.

`DEVICE=cpu` is only for tiny tests. Uploads are capped at 20 MiB. Optional
scenario JSONL files can be passed with `--test_jsonl`.

## After Transformers works: realtime vLLM

Stock vLLM does not emit `turn.delta`. Start the patched runtime in
[`voxtral-realtime/integrations/vllm/README.md`](../voxtral-realtime/integrations/vllm/README.md),
then:

```bash
cd turn-demo
BACKEND=vllm \
VLLM_URL=ws://127.0.0.1:8011/v1/realtime \
VLLM_MODEL=x-square-robot/X2-Turn-4B-0812 \
bash run.sh
```

The vLLM backend forwards microphone PCM to `/v1/realtime`. The Transformers
backend re-decodes the accumulated buffer and is for inspection, not latency
benchmarks.

## Remote streaming client (SSH tunnel)

`stream_client.py` runs on **your** machine, streams 16 kHz mono PCM to the
server's `/ws/stream` WebSocket, and renders the incremental ASR and the six
turn states in a live terminal dashboard. The model stays on the GPU box; only
audio goes up and JSON comes back. Its label set is X2-Turn's own (`idle`,
`noidle`, `speaking`, `turn_end`, `backchannel`, `uncertain`) — not SoulX's.

**1. Server (GPU box)** — the default `127.0.0.1` bind is exactly what you want;
the tunnel reaches it, nothing else can:

```bash
cd turn-demo
MODEL=/path/to/X2-Turn-4B-0812 DEVICE=cuda:0 bash run.sh
```

**2. Tunnel (client machine)** — forward local 7860 to the server's 7860:

```bash
ssh -N -L 7860:127.0.0.1:7860 user@gpu-box
```

**3. Client (client machine)**:

```bash
pip install -r requirements-client.txt      # sounddevice only needed for --mic

# replay a WAV file (any samplerate; resampled to 16k), paced to realtime:
python stream_client.py --wav assets/sample_en.wav

# or stream the live microphone (Ctrl-C to stop and flush the final result):
python stream_client.py --mic
```

Both default to `--url ws://localhost:7860/ws/stream`. Useful flags: `--speed`
(WAV replay rate), `--commit-ms` (server decode cadence), `--chunk-ms` (send
size), `--device` (mic input index, from `python -m sounddevice`).

The first request loads the 4B weights and can take minutes; the dashboard
shows `connecting` until the server sends `ready`. As with the browser path,
the Transformers backend re-decodes the accumulated buffer each commit, so it
is for inspection rather than latency benchmarks — point `--url` at the vLLM
backend for realtime behavior.

## Validate

```bash
pytest
```
