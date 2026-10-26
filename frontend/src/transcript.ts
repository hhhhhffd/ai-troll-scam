export interface Line {
  id: number
  role: 'user' | 'bot'
  text: string
  interrupted?: boolean
}

export interface TranscriptState {
  lines: Line[]
  openBotId: number | null
}

export type TranscriptAction =
  | { type: 'user'; text: string }
  | { type: 'bot_text'; text: string }
  | { type: 'listening' }
  | { type: 'interrupt' }
  | { type: 'reset' }

export const emptyTranscript: TranscriptState = { lines: [], openBotId: null }

export function transcriptReducer(s: TranscriptState, a: TranscriptAction): TranscriptState {
  switch (a.type) {
    case 'reset':
      return emptyTranscript
    case 'user': {
      const line: Line = { id: s.lines.length, role: 'user', text: a.text }
      // Gemini transcribes the user's speech after the bot has started answering:
      // keep the question above the reply that is still being spoken.
      const at = s.lines.findIndex((l) => l.id === s.openBotId)
      if (at >= 0) return { ...s, lines: [...s.lines.slice(0, at), line, ...s.lines.slice(at)] }
      return { lines: [...s.lines, line], openBotId: null }
    }
    case 'bot_text': {
      if (s.openBotId !== null) {
        return {
          ...s,
          lines: s.lines.map((l) => (l.id === s.openBotId ? { ...l, text: l.text + a.text } : l)),
        }
      }
      const id = s.lines.length
      return { lines: [...s.lines, { id, role: 'bot', text: a.text }], openBotId: id }
    }
    case 'listening':
      return { ...s, openBotId: null }
    case 'interrupt':
      return {
        lines: s.lines.map((l) => (l.id === s.openBotId ? { ...l, interrupted: true } : l)),
        openBotId: null,
      }
  }
}
