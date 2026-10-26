# Протокол WebSocket `/ws/call`

URL: `wss://<host>/ws/call?key=<DEMO_KEY>&persona=<persona_id>`

- `key` проверяется на хэндшейке, если в `.env` задан `DEMO_KEY`.
- `persona` — id из `GET /api/personas` (по умолчанию `granny`).

## Клиент → сервер

- **Binary**: сырой PCM16 little-endian, mono, 16000 Hz (рекомендуемый размер кадра — 20 мс = 640 байт).
- **Text JSON**: `{"type": "hangup"}` — положить трубку (можно просто закрыть сокет).

## Сервер → клиент

- **Binary**: сырой PCM16 little-endian, mono, частота из `ready.sampleRate`.
- **Text JSON**:

| Сообщение | Когда |
|---|---|
| `{"type":"ready","sampleRate":22050,"persona":"granny"}` | Звонок принят, сразу после подключения |
| `{"type":"busy"}` | Линия занята; затем сервер закрывает сокет с кодом `4409` |
| `{"type":"unauthorized"}` | Неверный ключ; затем закрытие с кодом `4401` |
| `{"type":"transcript","role":"user","text":"..."}` | Финальная фраза пользователя после STT |
| `{"type":"bot_text","text":"..."}` | Кусок текста ответа бота (по мере синтеза, конкатенировать) |
| `{"type":"state","state":"speaking"}` / `"listening"` | Бот начал/закончил говорить |
| `{"type":"interrupt"}` | Пользователь перебил: клиент обязан сбросить буфер воспроизведения |
| `{"type":"end","reason":"max_duration"\|"idle"}` | Сервер завершает звонок, затем закрывает сокет (код `1000`) |

## HTTP

- `GET /api/personas` → `[{"id":"granny","name":"Валентина Петровна","emoji":"👵","description":"..."}]`
- `GET /api/health` → `{"status":"ok","busy":false}`
