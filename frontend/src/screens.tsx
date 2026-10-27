import { useEffect, useRef, useState } from 'react'
import type { CallRecord, Engine, EndReason, Persona } from './types'

const ENGINE_LABELS: Record<string, string> = {
  local: 'локально',
  gemini: 'Gemini',
  cloud: 'ElevenLabs',
  fish: 'Fish',
  higgs: 'Higgs',
  espeech: 'ESpeech',
}
import type { Line } from './transcript'

function mmss(totalSec: number): string {
  const m = Math.floor(totalSec / 60)
  const s = totalSec % 60
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

function plural(n: number, one: string, few: string, many: string): string {
  const m10 = n % 10
  const m100 = n % 100
  if (m10 === 1 && m100 !== 11) return one
  if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few
  return many
}

function spent(sec: number): string {
  const m = Math.floor(sec / 60)
  const s = sec % 60
  const parts: string[] = []
  if (m > 0) parts.push(`${m} ${plural(m, 'минуту', 'минуты', 'минут')}`)
  parts.push(`${s} ${plural(s, 'секунду', 'секунды', 'секунд')}`)
  return parts.join(' ')
}

function Timer({ startedAt }: { startedAt: number | null }) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 250)
    return () => clearInterval(id)
  }, [])
  const sec = startedAt === null ? 0 : Math.max(0, Math.floor((now - startedAt) / 1000))
  return (
    <div className="timer" role="timer" aria-label="Время разговора">
      {mmss(sec)}
    </div>
  )
}

interface IdleProps {
  engines: Engine[]
  engineId: string
  onEngine(id: string): void
  personas: Persona[]
  personaId: string
  onPersona(id: string): void
  personasError: boolean
  records: CallRecord[]
  onCall(): void
}

export function IdleScreen(p: IdleProps) {
  return (
    <section className="screen idle">
      <header>
        <h1>Бабуля против мошенников</h1>
        <p className="lead">
          Позвоните «бабушке» и представьтесь сотрудником Kaspi. Попробуйте выманить у неё
          SMS-код. Чем дольше она вас задерживает, тем лучше для настоящих бабушек.
        </p>
      </header>

      {p.personasError && (
        <p className="note">Список собеседников не загрузился, доступна только бабушка по умолчанию.</p>
      )}

      <div className="engines" role="radiogroup" aria-label="Голосовой движок">
        {p.engines.map((e) => (
          <button
            key={e.id}
            type="button"
            role="radio"
            aria-checked={e.id === p.engineId}
            disabled={!e.available}
            className="engine"
            onClick={() => p.onEngine(e.id)}
          >
            <strong>{e.name}</strong>
            {!e.available && <small>{e.note ?? 'недоступен'}</small>}
          </button>
        ))}
      </div>

      <div className="personas" role="radiogroup" aria-label="Кому звонить">
        {p.personas.map((x) => (
          <button
            key={x.id}
            type="button"
            role="radio"
            aria-checked={x.id === p.personaId}
            className="persona"
            onClick={() => p.onPersona(x.id)}
          >
            <span className="emoji">{x.emoji}</span>
            <span className="persona-text">
              <strong>{x.name}</strong>
              <small>{x.description}</small>
            </span>
          </button>
        ))}
      </div>

      <button type="button" className="btn call" onClick={p.onCall}>
        <span aria-hidden="true">📞</span> Позвонить
      </button>

      <Records records={p.records} />
    </section>
  )
}

function Records({ records }: { records: CallRecord[] }) {
  return (
    <div className="records">
      <h2>Рекорды</h2>
      {records.length === 0 ? (
        <p className="muted">Пока пусто. Первый звонок за вами.</p>
      ) : (
        <table>
          <tbody>
            {records.map((r, i) => (
              <tr key={`${r.date}-${i}`}>
                <td className="rank">{i + 1}</td>
                <td className="dur">{mmss(r.durationSec)}</td>
                <td>
                  {r.personaName}
                  {r.engine && <small className="muted"> · {ENGINE_LABELS[r.engine] ?? r.engine}</small>}
                </td>
                <td className="date">{new Date(r.date).toLocaleDateString('ru-RU')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

export function DialingScreen({ persona, onCancel }: { persona: Persona; onCancel(): void }) {
  return (
    <section className="screen center">
      <div className="avatar ringing">{persona.emoji}</div>
      <h2>{persona.name}</h2>
      <p className="muted">Вызов…</p>
      <button type="button" className="btn hang" onClick={onCancel}>
        Отмена
      </button>
    </section>
  )
}

interface CallProps {
  engineName?: string
  persona: Persona
  startedAt: number | null
  speaking: boolean
  level: number
  lines: Line[]
  onHangup(): void
}

export function CallScreen(p: CallProps) {
  const logRef = useRef<HTMLDivElement>(null)
  const last = p.lines[p.lines.length - 1]
  useEffect(() => {
    const el = logRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [p.lines.length, last?.text])

  return (
    <section className="screen in-call">
      <div className="who">
        <div className={`avatar small${p.speaking ? ' talking' : ''}`}>{p.persona.emoji}</div>
        <div>
          <strong>{p.persona.name}</strong>
          <div className={`status${p.speaking ? ' on' : ''}`}>
            {p.speaking ? 'говорит' : 'слушает'}
          </div>
          {p.engineName && <small className="engine-tag">{p.engineName}</small>}
        </div>
      </div>

      <Timer startedAt={p.startedAt} />
      <p className="timer-caption">столько времени потрачено у мошенника</p>

      <div className="meter" aria-hidden="true">
        <i style={{ transform: `scaleX(${Math.min(1, p.level * 2)})` }} />
      </div>

      <div className="log" ref={logRef} aria-live="polite">
        {p.lines.map((l) => (
          <div key={l.id} className={`line ${l.role}`}>
            <span className="who-label">{l.role === 'user' ? 'Вы' : 'Бабушка'}</span>
            {l.text}
            {l.interrupted && <em> (перебили)</em>}
          </div>
        ))}
      </div>

      <button type="button" className="btn hang" onClick={p.onHangup}>
        Положить трубку
      </button>
    </section>
  )
}

const REASONS: Record<EndReason, string> = {
  hangup: 'Вы положили трубку.',
  max_duration: 'Бабушка сама положила трубку: разговор слишком затянулся.',
  idle: 'Бабушка положила трубку: вы слишком долго молчали.',
  lost: 'Соединение потеряно.',
}

export function EndedScreen(p: {
  persona: Persona
  reason: EndReason
  durationSec: number
  onAgain(): void
  onBack(): void
}) {
  return (
    <section className="screen center">
      <div className="avatar">{p.persona.emoji}</div>
      <h2>Звонок завершён</h2>
      <p className="summary">Вы потратили {spent(p.durationSec)} на бабушку</p>
      <p className="muted">{REASONS[p.reason]}</p>
      <button type="button" className="btn call" onClick={p.onAgain}>
        Позвонить ещё
      </button>
      <button type="button" className="btn ghost" onClick={p.onBack}>
        На главную
      </button>
    </section>
  )
}

export function MessageScreen(p: {
  icon: string
  title: string
  text: string
  action: string
  onAction(): void
  secondary?: string
  onSecondary?(): void
}) {
  return (
    <section className="screen center">
      <div className="avatar">{p.icon}</div>
      <h2>{p.title}</h2>
      <p className="muted">{p.text}</p>
      <button type="button" className="btn call" onClick={p.onAction}>
        {p.action}
      </button>
      {p.secondary && (
        <button type="button" className="btn ghost" onClick={p.onSecondary}>
          {p.secondary}
        </button>
      )}
    </section>
  )
}
