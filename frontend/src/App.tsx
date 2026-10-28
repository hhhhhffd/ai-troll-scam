import { useCallback, useEffect, useReducer, useRef, useState } from 'react'
import { CallSession, MicError } from './call'
import { loadPhoneVoice, resolveKey } from './storage'
import { emptyTranscript, transcriptReducer } from './transcript'
import type { ServerMessage } from './types'

const FALLBACK_NAME = 'Бабушка'
const ENDED_SHOW_MS = 2000

type Phase =
  | { name: 'idle'; note?: string }
  | { name: 'dialing' }
  | { name: 'call' }
  | { name: 'ended'; durationSec: number }

function buildUrl(key: string): string {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws'
  const q = key ? `?${new URLSearchParams({ key })}` : ''
  return `${scheme}://${location.host}/ws/call${q}`
}

function formatTime(sec: number): string {
  const m = Math.floor(sec / 60)
  return `${m}:${String(sec % 60).padStart(2, '0')}`
}

function PhoneIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path
        fill="currentColor"
        d="M6.6 10.8a15.1 15.1 0 0 0 6.6 6.6l2.2-2.2a1 1 0 0 1 1-.25c1.1.37 2.3.57 3.6.57a1 1 0 0 1 1 1V20a1 1 0 0 1-1 1A17 17 0 0 1 3 4a1 1 0 0 1 1-1h3.5a1 1 0 0 1 1 1c0 1.25.2 2.45.57 3.6a1 1 0 0 1-.25 1z"
      />
    </svg>
  )
}

function Timer({ startedAt }: { startedAt: number }) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 500)
    return () => clearInterval(id)
  }, [])
  return <>{formatTime(Math.max(0, Math.floor((now - startedAt) / 1000)))}</>
}

export function App() {
  const [phase, setPhase] = useState<Phase>({ name: 'idle' })
  const [name, setName] = useState(FALLBACK_NAME)
  const [transcript, dispatch] = useReducer(transcriptReducer, emptyTranscript)
  const [speaking, setSpeaking] = useState(false)
  const [startedAt, setStartedAt] = useState<number | null>(null)

  const sessionRef = useRef<CallSession | null>(null)
  const startedAtRef = useRef<number | null>(null)
  const key = useRef(resolveKey()).current

  useEffect(() => {
    const ctrl = new AbortController()
    fetch('/api/persona', { signal: ctrl.signal })
      .then((r) => {
        if (!r.ok) throw new Error(String(r.status))
        return r.json() as Promise<{ name?: string }>
      })
      .then((p) => {
        if (p.name) setName(p.name)
      })
      .catch(() => {
        // keep the fallback name
      })
    return () => ctrl.abort()
  }, [])

  const teardown = useCallback(() => {
    sessionRef.current?.dispose()
    sessionRef.current = null
    startedAtRef.current = null
    setStartedAt(null)
    setSpeaking(false)
  }, [])

  useEffect(() => teardown, [teardown])

  // The "call ended" screen is shown briefly, then the idle screen returns.
  useEffect(() => {
    if (phase.name !== 'ended') return
    const id = setTimeout(() => setPhase({ name: 'idle' }), ENDED_SHOW_MS)
    return () => clearTimeout(id)
  }, [phase.name])

  const finish = useCallback(() => {
    const began = startedAtRef.current
    const durationSec = began === null ? 0 : Math.floor((Date.now() - began) / 1000)
    teardown()
    setPhase({ name: 'ended', durationSec })
  }, [teardown])

  const fail = useCallback(
    (note: string) => {
      teardown()
      setPhase({ name: 'idle', note })
    },
    [teardown],
  )

  const startCall = () => {
    dispatch({ type: 'reset' })
    setPhase({ name: 'dialing' })

    let ended = false
    const session = new CallSession(
      {
        onLevel: () => {},
        onMessage: (msg: ServerMessage) => {
          switch (msg.type) {
            case 'ready': {
              const now = Date.now()
              startedAtRef.current = now
              setStartedAt(now)
              setPhase({ name: 'call' })
              break
            }
            case 'busy':
              fail('Линия занята, попробуйте позже')
              break
            case 'unauthorized':
              fail('Нет доступа: нужна ссылка с ключом')
              break
            case 'error':
              fail(msg.message)
              break
            case 'transcript':
              dispatch({ type: 'user', text: msg.text })
              break
            case 'bot_text':
              dispatch({ type: 'bot_text', text: msg.text })
              break
            case 'state':
              setSpeaking(msg.state === 'speaking')
              if (msg.state === 'listening') dispatch({ type: 'listening' })
              break
            case 'interrupt':
              dispatch({ type: 'interrupt' })
              break
            case 'end':
              ended = true
              break
          }
        },
        onClose: (code) => {
          if (code === 4409) fail('Линия занята, попробуйте позже')
          else if (code === 4401) fail('Нет доступа: нужна ссылка с ключом')
          else if (code === 4400) fail('Сервер отклонил звонок')
          else if (ended || startedAtRef.current !== null) finish()
          else fail('Не удалось соединиться')
        },
      },
      loadPhoneVoice(),
    )
    sessionRef.current = session

    void (async () => {
      try {
        await session.prepare()
        await session.playRing()
        if (sessionRef.current !== session) return
        session.connect(buildUrl(key))
      } catch (e) {
        if (sessionRef.current !== session) return
        fail(e instanceof MicError ? e.message : 'Не удалось запустить звонок')
      }
    })()
  }

  const hangup = () => {
    if (phase.name === 'dialing') {
      teardown()
      setPhase({ name: 'idle' })
      return
    }
    sessionRef.current?.hangup()
    finish()
  }

  const inCall = phase.name === 'dialing' || phase.name === 'call'
  const caption = transcript.lines[transcript.lines.length - 1]

  let status: React.ReactNode = ''
  if (phase.name === 'dialing') status = 'Вызов…'
  else if (phase.name === 'call' && startedAt !== null) status = <Timer startedAt={startedAt} />
  else if (phase.name === 'ended')
    status = `Вызов завершён · ${formatTime(phase.durationSec)}`
  else if (phase.name === 'idle' && phase.note) status = phase.note

  return (
    <main className="phone">
      <div className="blobs" aria-hidden="true">
        <i className="blob b1" />
        <i className="blob b2" />
        <i className="blob b3" />
      </div>

      <section className="caller">
        <div className={`avatar${speaking && phase.name === 'call' ? ' speaking' : ''}`}>
          <span aria-hidden="true">👵</span>
        </div>
        <h1>{name}</h1>
        <p className="status" aria-live="polite">
          {status}
          {phase.name === 'idle' && !phase.note && (
            <span className="hint">Позвоните бабушке и попробуйте выманить код из SMS</span>
          )}
        </p>
      </section>

      <p className="caption" aria-live="off">
        {phase.name === 'call' && caption ? caption.text : ''}
      </p>

      <div className="actions">
        {inCall ? (
          <button className="btn hangup" onClick={hangup} aria-label="Сбросить">
            <PhoneIcon />
          </button>
        ) : (
          <button className="btn call" onClick={startCall} aria-label="Позвонить">
            <PhoneIcon />
          </button>
        )}
      </div>
    </main>
  )
}
