export type ServerMessage =
  | { type: 'ready'; sampleRate: number }
  | { type: 'busy' }
  | { type: 'unauthorized' }
  | { type: 'error'; message: string }
  | { type: 'transcript'; role: 'user'; text: string }
  | { type: 'bot_text'; text: string }
  | { type: 'state'; state: 'speaking' | 'listening' }
  | { type: 'interrupt' }
  | { type: 'end'; reason: 'max_duration' | 'idle' }
