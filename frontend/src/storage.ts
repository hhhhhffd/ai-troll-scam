import type { CallRecord } from './types'

const KEY_STORAGE = 'troll.key'
const RECORDS_STORAGE = 'troll.records'
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
