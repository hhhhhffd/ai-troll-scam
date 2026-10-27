# Протокол WebSocket `/ws/call`

URL: `wss://<host>/ws/call?key=<DEMO_KEY>&persona=<persona_id>&engine=local|gemini|cloud|fish|higgs|espeech`

- `key` проверяется на хэндшейке, если в `.env` задан `DEMO_KEY`.
- `persona` — id из `GET /api/personas` (по умолчанию `granny`).
- `engine` — `local` (Whisper + Qwen + Piper, по умолчанию), `gemini` (Gemini Live) или каскад STT → облачная LLM → TTS: `cloud` (ElevenLabs v4), `fish` (Fish Audio, облако), `higgs` (Higgs TTS 3 на своей GPU), `espeech` (ESpeech на своей GPU). Неизвестное значение трактуется как `local`. Доступность см. `GET /api/engines`. Протокол у всех движков один, отличается только `sampleRate` в `ready`: 22050 для `local`, 24000 для остальных.
- Если движку нужны другие модели на GPU, сервер перед `ready` выгружает лишние и загружает нужные (до нескольких минут для Higgs). Клиент всё это время ждёт `ready`.

## Клиент → сервер

- **Binary**: сырой PCM16 little-endian, mono, 16000 Hz (рекомендуемый размер кадра — 20 мс = 640 байт).
- **Text JSON**: `{"type": "hangup"}` — положить трубку (можно просто закрыть сокет).

## Сервер → клиент

- **Binary**: сырой PCM16 little-endian, mono, частота из `ready.sampleRate`.
- **Text JSON**:

| Сообщение | Когда |
|---|---|
| `{"type":"ready","sampleRate":22050\|24000,"persona":"granny"}` | Звонок принят, сразу после подключения |
| `{"type":"busy"}` | Линия занята; затем сервер закрывает сокет с кодом `4409` |
| `{"type":"unauthorized"}` | Неверный ключ; затем закрытие с кодом `4401` |
| `{"type":"error","message":"..."}` | Звонок отклонён (движок недоступен: нет ключа, не запущен TTS-контейнер, модель не загрузилась); затем закрытие с кодом `4400` |
| `{"type":"transcript","role":"user","text":"..."}` | Финальная фраза пользователя после STT. В режиме `gemini` приходит с задержкой, иногда уже после начала ответа бота |
| `{"type":"bot_text","text":"..."}` | Кусок текста ответа бота (по мере синтеза, конкатенировать). Теги эмоций (`[sighs]` и т.п.) сервер вырезает |
| `{"type":"state","state":"speaking"}` / `"listening"` | Бот начал/закончил говорить |
| `{"type":"interrupt"}` | Пользователь перебил: клиент обязан сбросить буфер воспроизведения |
| `{"type":"end","reason":"max_duration"\|"idle"}` | Сервер завершает звонок, затем закрывает сокет (код `1000`) |

## HTTP

- `GET /api/engines` → `[{"id":"local","name":"...","available":true},{"id":"cloud","name":"Облако (ElevenLabs v4)","available":false,"note":"нужны ELEVENLABS_API_KEY и ELEVENLABS_VOICE_ID"}, ...]`; `note` есть только у недоступных и объясняет причину
- `GET /api/personas` → `[{"id":"granny","name":"Валентина Петровна","emoji":"👵","description":"..."}]`
- `GET /api/health` → `{"status":"ok","busy":false,"gpu":["ollama","whisper"]}`; `gpu` — какие модели сейчас в VRAM
