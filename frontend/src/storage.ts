import type { CallRecord } from './types'

const KEY_STORAGE = 'troll.key'
const RECORDS_STORAGE = 'troll.records'
const ENGINE_STORAGE = 'troll.engine'
const PHONE_VOICE_STORAGE = 'troll.phoneVoice'
const MAX_RECORDS = 10

export function resolveKey(): string {
  let fromUrl: string | null = null
  try {
    fromUrl = new URLSearchParams(location.search).get('key')
  } catch {
    // malformed URL: fall back to storage
  }
  try {
    if (fromUrl) {
      localStorage.setItem(KEY_STORAGE, fromUrl)
      return fromUrl
    }
    return localStorage.getItem(KEY_STORAGE) ?? ''
  } catch {
    return fromUrl ?? ''
  }
}

export function loadRecords(): CallRecord[] {
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(RECORDS_STORAGE) ?? '[]')
    if (!Array.isArray(parsed)) return []
    return parsed.filter(
      (r): r is CallRecord =>
        r && typeof r.durationSec === 'number' && typeof r.personaName === 'string',
    )
  } catch {
    return []
  }
}

export function saveRecord(record: CallRecord): CallRecord[] {
  const records = [...loadRecords(), record]
    .sort((a, b) => b.durationSec - a.durationSec)
    .slice(0, MAX_RECORDS)
  try {
    localStorage.setItem(RECORDS_STORAGE, JSON.stringify(records))
  } catch {
    // storage unavailable: records live only for this page session
  }
  return records
}

export function loadEngine(): string | null {
  try {
    return localStorage.getItem(ENGINE_STORAGE)
  } catch {
    return null
  }
}

export function saveEngine(id: string): void {
  try {
    localStorage.setItem(ENGINE_STORAGE, id)
  } catch {
    // storage unavailable: choice lives only for this page session
  }
}

/** Phone-line voice filter; on by default, only an explicit '0' turns it off. */
export function loadPhoneVoice(): boolean {
  try {
    return localStorage.getItem(PHONE_VOICE_STORAGE) !== '0'
  } catch {
    return true
  }
}

export function savePhoneVoice(on: boolean): void {
  try {
    localStorage.setItem(PHONE_VOICE_STORAGE, on ? '1' : '0')
  } catch {
    // storage unavailable: choice lives only for this page session
  }
}
