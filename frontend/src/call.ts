import type { ServerMessage } from './types'

export class MicError extends Error {}

export interface CallHandlers {
  onMessage(msg: ServerMessage): void
  onLevel(level: number): void
  onClose(code: number): void
}

const RING_BEEP_SEC = 0.4
const RING_GAP_SEC = 0.25
const RING_BEEPS = 2

export class CallSession {
  private ctx: AudioContext
  private stream: MediaStream | null = null
  private micNode: AudioWorkletNode | null = null
  private micSource: MediaStreamAudioSourceNode | null = null
  private ws: WebSocket | null = null
  private sources = new Set<AudioBufferSourceNode>()
  private cursor = 0
  private playRate = 22050
  private leftover: number | null = null
  private disposed = false

  /** Must be constructed synchronously inside the click handler (iOS autoplay policy). */
  constructor(private handlers: CallHandlers) {
    this.ctx = new AudioContext()
    void this.ctx.resume()
  }

  async prepare(): Promise<void> {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new MicError(
        'Браузер не даёт доступ к микрофону. Откройте сайт по https:// (или localhost).',
      )
    }
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      })
    } catch (e) {
      const name = e instanceof DOMException ? e.name : ''
      throw new MicError(
        name === 'NotFoundError'
          ? 'Микрофон не найден. Подключите микрофон и попробуйте ещё раз.'
          : 'Нет доступа к микрофону. Разрешите его в настройках браузера и попробуйте ещё раз.',
      )
    }
    if (this.disposed) return
    await this.ctx.audioWorklet.addModule(`${import.meta.env.BASE_URL}mic-worklet.js`)
    if (this.disposed) return
    this.micSource = this.ctx.createMediaStreamSource(this.stream)
    this.micNode = new AudioWorkletNode(this.ctx, 'mic-processor', {
      numberOfInputs: 1,
      numberOfOutputs: 0,
      channelCount: 1,
      channelCountMode: 'explicit',
    })
    this.micNode.port.onmessage = (e: MessageEvent<{ pcm?: ArrayBuffer; level?: number }>) => {
      if (e.data.pcm) {
        if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(e.data.pcm)
      } else if (e.data.level !== undefined) {
        this.handlers.onLevel(e.data.level)
      }
    }
    this.micSource.connect(this.micNode)
  }

  /** Two short 425 Hz beeps (RU/KZ ringback). Resolves when finished. */
  playRing(): Promise<void> {
    const ctx = this.ctx
    const gain = ctx.createGain()
    gain.gain.value = 0
    gain.connect(ctx.destination)
    const osc = ctx.createOscillator()
    osc.frequency.value = 425
    osc.connect(gain)
    const t0 = ctx.currentTime + 0.05
    let end = t0
    for (let i = 0; i < RING_BEEPS; i++) {
      const start = t0 + i * (RING_BEEP_SEC + RING_GAP_SEC)
      end = start + RING_BEEP_SEC
      gain.gain.setValueAtTime(0, start)
      gain.gain.linearRampToValueAtTime(0.15, start + 0.02)
      gain.gain.setValueAtTime(0.15, end - 0.02)
      gain.gain.linearRampToValueAtTime(0, end)
    }
    osc.start(t0)
    osc.stop(end + 0.05)
    return new Promise((resolve) => {
      osc.onended = () => {
        osc.disconnect()
        gain.disconnect()
        resolve()
      }
    })
  }

  connect(url: string): void {
    const ws = new WebSocket(url)
    ws.binaryType = 'arraybuffer'
    ws.onmessage = (e) => {
      if (typeof e.data === 'string') {
        let msg: ServerMessage
        try {
          msg = JSON.parse(e.data) as ServerMessage
        } catch {
          return
        }
        if (msg.type === 'ready') this.playRate = msg.sampleRate
        else if (msg.type === 'interrupt') this.stopPlayback()
        this.handlers.onMessage(msg)
      } else {
        this.play(e.data as ArrayBuffer)
      }
    }
    ws.onclose = (e) => {
      if (this.ws !== ws) return
      this.handlers.onClose(e.code)
    }
    this.ws = ws
  }

  hangup(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'hangup' }))
    }
  }

  private play(data: ArrayBuffer): void {
    let bytes = new Uint8Array(data)
    if (this.leftover !== null) {
      const joined = new Uint8Array(bytes.length + 1)
      joined[0] = this.leftover
      joined.set(bytes, 1)
      bytes = joined
      this.leftover = null
    }
    if (bytes.length % 2) {
      this.leftover = bytes[bytes.length - 1]
      bytes = bytes.subarray(0, bytes.length - 1)
    }
    const n = bytes.length / 2
    if (n === 0) return

    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.length)
    const buffer = this.ctx.createBuffer(1, n, this.playRate)
    const out = buffer.getChannelData(0)
    for (let i = 0; i < n; i++) out[i] = view.getInt16(i * 2, true) / 32768

    const src = this.ctx.createBufferSource()
    src.buffer = buffer
    src.connect(this.ctx.destination)
    // Small lead on (re)start absorbs network jitter; afterwards chunks chain gaplessly.
    const startAt = Math.max(this.cursor, this.ctx.currentTime + 0.06)
    src.start(startAt)
    this.cursor = startAt + buffer.duration
    this.sources.add(src)
    src.onended = () => this.sources.delete(src)
  }

  private stopPlayback(): void {
    for (const src of this.sources) {
      src.onended = null
      try {
        src.stop()
      } catch {
        // already stopped
      }
      src.disconnect()
    }
    this.sources.clear()
    this.cursor = 0
    this.leftover = null
  }

  dispose(): void {
    if (this.disposed) return
    this.disposed = true
    this.stopPlayback()
    if (this.ws) {
      this.ws.onclose = null
      this.ws.onmessage = null
      this.ws.close()
      this.ws = null
    }
    if (this.micNode) this.micNode.port.onmessage = null
    this.micSource?.disconnect()
    this.micNode?.disconnect()
    this.stream?.getTracks().forEach((t) => t.stop())
    void this.ctx.close()
  }
}
