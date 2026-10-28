const KEY_STORAGE = 'troll.key'
const PHONE_VOICE_STORAGE = 'troll.phoneVoice'

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

/** Phone-line voice filter; on by default, only an explicit '0' turns it off. */
export function loadPhoneVoice(): boolean {
  try {
    return localStorage.getItem(PHONE_VOICE_STORAGE) !== '0'
  } catch {
    return true
  }
}
