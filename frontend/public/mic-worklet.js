// Resamples mic input from the AudioContext rate to 16 kHz mono PCM16 LE
// (linear interpolation) and posts 20 ms chunks (320 samples = 640 bytes).
const TARGET_RATE = 16000
const CHUNK_SAMPLES = 320
const LEVEL_EVERY_BLOCKS = 16

class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super()
    this.step = sampleRate / TARGET_RATE
    // Position of the next output sample in input coordinates relative to the
    // current block start; -1 refers to the last sample of the previous block.
    this.pos = 0
    this.last = 0
    this.chunk = new Int16Array(CHUNK_SAMPLES)
    this.filled = 0
    this.blocks = 0
    this.peak = 0
  }

  process(inputs) {
    const x = inputs[0] && inputs[0][0]
    if (!x || x.length === 0) return true
    const n = x.length

    for (let i = 0; i < n; i++) {
      const a = Math.abs(x[i])
      if (a > this.peak) this.peak = a
    }

    while (true) {
      const i0 = Math.floor(this.pos)
      if (i0 + 1 > n - 1) break
      const s0 = i0 < 0 ? this.last : x[i0]
      const frac = this.pos - i0
      const v = s0 + (x[i0 + 1] - s0) * frac
      const c = v < -1 ? -1 : v > 1 ? 1 : v
      this.chunk[this.filled++] = c < 0 ? c * 32768 : c * 32767
      if (this.filled === CHUNK_SAMPLES) {
        const buf = this.chunk.buffer
        this.port.postMessage({ pcm: buf }, [buf])
        this.chunk = new Int16Array(CHUNK_SAMPLES)
        this.filled = 0
      }
      this.pos += this.step
    }
    this.pos -= n
    this.last = x[n - 1]

    if (++this.blocks >= LEVEL_EVERY_BLOCKS) {
      this.port.postMessage({ level: this.peak })
      this.blocks = 0
      this.peak = 0
    }
    return true
  }
}

registerProcessor('mic-processor', MicProcessor)
