import { useCallback, useEffect, useReducer, useRef, useState } from 'react'
import { CallSession, MicError } from './call'
import { loadEngine, loadPhoneVoice, loadRecords, resolveKey, saveEngine, savePhoneVoice, saveRecord } from './storage'
import { emptyTranscript, transcriptReducer } from './transcript'
import type { CallRecord, Engine, EndReason, Persona, ServerMessage } from './types'
import { CallScreen, DialingScreen, EndedScreen, IdleScreen, MessageScreen } from './screens'

const FALLBACK_PERSONAS: Persona[] = [
  {
    id: 'granny',
    name: 'Валентина Петровна',
    emoji: '👵',
    description: 'Ей 74 года, она никуда не торопится.',
  },
]

const FALLBACK_ENGINES: Engine[] = [
  { id: 'local', name: 'Локально (Whisper + Qwen + Piper)', available: true },
]

type Phase =
  | { name: 'idle' }
  | { name: 'dialing' }
  | { name: 'call' }
  | { name: 'busy' }
  | { name: 'unauthorized' }
  | { name: 'error'; text: string }
  | { name: 'server_error'; text: string }
  | { name: 'ended'; reason: EndReason; durationSec: number }

function buildUrl(key: string, personaId: string, engineId: string): string {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws'
  const q = new URLSearchParams({ persona: personaId, engine: engineId })
  if (key) q.set('key', key)
  return `${scheme}://${location.host}/ws/call?${q}`
}

export function App() {
  const [phase, setPhase] = useState<Phase>({ name: 'idle' })
  const [personas, setPersonas] = useState<Persona[]>(FALLBACK_PERSONAS)
  const [personasError, setPersonasError] = useState(false)
  const [personaId, setPersonaId] = useState(FALLBACK_PERSONAS[0].id)
  const [engines, setEngines] = useState<Engine[]>(FALLBACK_ENGINES)
  const [engineId, setEngineId] = useState(() => loadEngine() ?? FALLBACK_ENGINES[0].id)
  const [phoneVoice, setPhoneVoice] = useState(() => loadPhoneVoice())
  const [records, setRecords] = useState<CallRecord[]>(() => loadRecords())
  const [transcript, dispatch] = useReducer(transcriptReducer, emptyTranscript)
  const [speaking, setSpeaking] = useState(false)
  const [level, setLevel] = useState(0)
  const [startedAt, setStartedAt] = useState<number | null>(null)

  const sessionRef = useRef<CallSession | null>(null)
  const startedAtRef = useRef<number | null>(null)
  const personaRef = useRef<Persona>(FALLBACK_PERSONAS[0])
  const engineRef = useRef(engineId)
  const key = useRef(resolveKey()).current

  useEffect(() => {
    const ctrl = new AbortController()
    fetch('/api/personas', { signal: ctrl.signal })
      .then((r) => {
        if (!r.ok) throw new Error(String(r.status))
        return r.json() as Promise<Persona[]>
      })
      .then((list) => {
        if (!Array.isArray(list) || list.length === 0) throw new Error('empty')
        setPersonas(list)
        setPersonaId((cur) => (list.some((p) => p.id === cur) ? cur : list[0].id))
      })
      .catch(() => {
        if (!ctrl.signal.aborted) setPersonasError(true)
      })
    return () => ctrl.abort()
  }, [])

  useEffect(() => {
    const ctrl = new AbortController()
    fetch('/api/engines', { signal: ctrl.signal })
      .then((r) => {
        if (!r.ok) throw new Error(String(r.status))
        return r.json() as Promise<Engine[]>
      })
      .then((list) => {
        if (!Array.isArray(list) || list.length === 0) throw new Error('empty')
        setEngines(list)
        // A saved engine that is gone or unavailable (e.g. the key was removed) falls back.
        setEngineId((cur) => (list.some((e) => e.id === cur && e.available) ? cur : 'local'))
      })
      .catch(() => {
        // keep the local-only fallback
      })
    return () => ctrl.abort()
  }, [])

  const pickEngine = (id: string) => {
    setEngineId(id)
    saveEngine(id)
  }

  const pickPhoneVoice = (on: boolean) => {
    setPhoneVoice(on)
    savePhoneVoice(on)
    sessionRef.current?.setPhoneVoice(on)
  }

  const teardown = useCallback(() => {
    sessionRef.current?.dispose()
    sessionRef.current = null
    startedAtRef.current = null
    setStartedAt(null)
    setSpeaking(false)
    setLevel(0)
  }, [])

  useEffect(() => teardown, [teardown])

  const finish = useCallback(
    (reason: EndReason) => {
      const began = startedAtRef.current
      const durationSec = began === null ? 0 : Math.floor((Date.now() - began) / 1000)
      teardown()
      if (began !== null && durationSec > 0) {
        const p = personaRef.current
        setRecords(
          saveRecord({
            durationSec,
            personaId: p.id,
            personaName: p.name,
            engine: engineRef.current,
            date: new Date().toISOString(),
          }),
        )
      }
      setPhase({ name: 'ended', reason, durationSec })
    },
    [teardown],
  )

  const startCall = () => {
    const persona = personas.find((p) => p.id === personaId) ?? personas[0]
    personaRef.current = persona
    engineRef.current = engineId
    dispatch({ type: 'reset' })
    setPhase({ name: 'dialing' })

    let endReason: EndReason | null = null
    const session = new CallSession({
      onLevel: setLevel,
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
            teardown()
            setPhase({ name: 'busy' })
            break
          case 'unauthorized':
            teardown()
            setPhase({ name: 'unauthorized' })
            break
          case 'error':
            teardown()
            setPhase({ name: 'server_error', text: msg.message })
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
            endReason = msg.reason
            break
        }
      },
      onClose: (code) => {
        if (code === 4400) {
          // The `error` message already set the phase; this covers a lost message.
          teardown()
          setPhase((p) =>
            p.name === 'server_error' ? p : { name: 'server_error', text: 'Сервер отклонил звонок.' },
          )
        } else if (code === 4409) {
          teardown()
          setPhase({ name: 'busy' })
        } else if (code === 4401) {
          teardown()
          setPhase({ name: 'unauthorized' })
        } else if (endReason) finish(endReason)
        else if (startedAtRef.current === null) {
          teardown()
          setPhase({ name: 'error', text: 'Не удалось соединиться с сервером.' })
        } else finish('lost')
      },
    }, phoneVoice)
    sessionRef.current = session

    void (async () => {
      try {
        await session.prepare()
        await session.playRing()
        if (sessionRef.current !== session) return
        session.connect(buildUrl(key, persona.id, engineId))
      } catch (e) {
        if (sessionRef.current !== session) return
        teardown()
        setPhase({
          name: 'error',
          text: e instanceof MicError ? e.message : 'Не удалось запустить звонок.',
        })
      }
    })()
  }

  const hangup = () => {
    sessionRef.current?.hangup()
    finish('hangup')
  }

  const backToIdle = () => setPhase({ name: 'idle' })
  const persona = personaRef.current

  return (
    <main className="phone">
      {phase.name === 'idle' && (
        <IdleScreen
          personas={personas}
          personaId={personaId}
          engines={engines}
          engineId={engineId}
          onEngine={pickEngine}
          phoneVoice={phoneVoice}
          onPhoneVoice={pickPhoneVoice}
          onPersona={setPersonaId}
          personasError={personasError}
          records={records}
          onCall={startCall}
        />
      )}
      {phase.name === 'dialing' && <DialingScreen persona={persona} onCancel={() => {
          teardown()
          backToIdle()
        }} />}
      {phase.name === 'call' && (
        <CallScreen
          persona={persona}
          engineName={engines.find((e) => e.id === engineRef.current)?.name}
          startedAt={startedAt}
          speaking={speaking}
          level={level}
          phoneVoice={phoneVoice}
          onPhoneVoice={pickPhoneVoice}
          lines={transcript.lines}
          onHangup={hangup}
        />
      )}
      {phase.name === 'busy' && (
        <MessageScreen
          icon="📵"
          title="Линия занята"
          text="Бабушка сейчас разговаривает с другим мошенником. Попробуйте чуть позже."
          action="Позвонить снова"
          onAction={startCall}
          secondary="Назад"
          onSecondary={backToIdle}
        />
      )}
      {phase.name === 'unauthorized' && (
        <MessageScreen
          icon="🔑"
          title="Нет доступа"
          text="Откройте ссылку с правильным ключом: адрес сайта должен заканчиваться на ?key=…"
          action="Назад"
          onAction={backToIdle}
        />
      )}
      {phase.name === 'server_error' && (
        <MessageScreen
          icon="⚠️"
          title="Звонок отклонён"
          text={phase.text}
          action="Назад"
          onAction={backToIdle}
        />
      )}
      {phase.name === 'error' && (
        <MessageScreen
          icon="🎙️"
          title="Звонок не удался"
          text={phase.text}
          action="Назад"
          onAction={backToIdle}
        />
      )}
      {phase.name === 'ended' && (
        <EndedScreen
          persona={persona}
          reason={phase.reason}
          durationSec={phase.durationSec}
          onAgain={startCall}
          onBack={backToIdle}
        />
      )}
    </main>
  )
}
