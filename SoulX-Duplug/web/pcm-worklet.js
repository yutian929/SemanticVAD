/**
 * AudioWorklet processor: collects mono microphone PCM (Float32, native sample
 * rate) and posts it to the main thread in fixed-size blocks.
 *
 * Resampling / chunking into the 16kHz-2560-sample frames expected by the
 * SoulX-Duplug server is done on the main thread (see app.js).
 */
const BLOCK_SIZE = 1024;

class PCMCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf = new Float32Array(BLOCK_SIZE);
    this._filled = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;

    const channel = input[0];
    if (!channel) return true;

    let offset = 0;
    while (offset < channel.length) {
      const n = Math.min(BLOCK_SIZE - this._filled, channel.length - offset);
      this._buf.set(channel.subarray(offset, offset + n), this._filled);
      this._filled += n;
      offset += n;

      if (this._filled === BLOCK_SIZE) {
        const out = new Float32Array(this._buf);
        this.port.postMessage(out, [out.buffer]);
        this._filled = 0;
      }
    }

    return true;
  }
}

registerProcessor('pcm-capture', PCMCaptureProcessor);
