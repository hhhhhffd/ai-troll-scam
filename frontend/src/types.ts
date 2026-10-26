export interface Persona {
  id: string
  name: string
  emoji: string
  description: string
}

export type ServerMessage =
  | { type: 'ready'; sampleRate: number; persona: string }
  | { type: 'busy' }
  | { type: 'unauthorized' }
  | { type: 'transcript'; role: 'user'; text: string }
  | { type: 'bot_text'; text: string }
  | { type: 'state'; state: 'speaking' | 'listening' }
  | { type: 'interrupt' }
  | { type: 'end'; reason: 'max_duration' | 'idle' }

export type EndReason = 'hangup' | 'max_duration' | 'idle' | 'lost'

export interface CallRecord {
  durationSec: number
  personaId: string
  personaName: string
  date: string
}
