# Бабуля против мошенников

Голосовой бот для троллинга телефонных мошенников. Открываете сайт, жмёте зелёную трубку, представляетесь сотрудником Каспи банка и пытаетесь выудить код из SMS. Бабушка Валентина Петровна тянет время, переспрашивает и ловит на противоречиях. Наружу сайт доступен через Cloudflare Tunnel.

```
Браузер ──WS (PCM16)──► cloudflared ─► nginx (web) ─┬─ /         статика React (экран звонка)
                                                    └─ /ws/call  bot (Pipecat)
                                                                  ├─ Silero VAD + barge-in
                                                                  ├─ STT: Groq Whisper large-v3-turbo
                                                                  ├─ LLM: модели Groq по очереди → llm (Ollama, qwen3:4b-instruct, GPU)
                                                                  └─ TTS: ElevenLabs v4 Turbo (Text-to-Dialogue)
```

Почему WebSocket, а не WebRTC: Cloudflare Tunnel не проксирует UDP, а сырой PCM поверх WebSocket проходит надёжно. Контракт клиента и сервера описан в [PROTOCOL.md](PROTOCOL.md).

## Бабушка

Персонаж описан в `bot/persona.md`: во frontmatter имя и приветствие, дальше системный промпт. К промпту бот добавляет правила для аудиотегов ElevenLabs (`[sighs]`, `[annoyed]`, `[shouts]` и т.п.): голос их отыгрывает, а на экран они не попадают. Персонаж вшит в образ, после правки нужен `docker compose up -d --build bot`.

## LLM: цепочка моделей

`GROQ_MODEL` — список `модель[:reasoning_effort]` через запятую. Если модель отвечает 429 или 5xx, та же реплика сразу уходит следующей. Лимиты бесплатного тарифа Groq считаются отдельно для каждой модели, так что список умножает запас. При `CLOUD_LLM_LOCAL_FALLBACK=1` последним идёт локальный `LLM_MODEL` в Ollama (контейнер `llm`). Бот загружает его в VRAM при старте (около 4 ГБ), чтобы он отвечал сразу, без 15 с холодной загрузки.

## Запуск

Требования: Docker. Для локального запасного LLM нужна NVIDIA GPU (на Windows это Docker Desktop с WSL2) и Ollama на хосте.

```bash
cp .env.example .env   # GROQ_API_KEY, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, DEMO_KEY, CLOUDFLARE_TUNNEL_TOKEN
docker compose up -d
```

- Локально сайт открывается на `http://localhost:8080/?key=<DEMO_KEY>`. Без https микрофон работает только на `localhost`.
- Без ключей Groq или ElevenLabs бот стартует, но отклоняет звонки с сообщением, каких ключей не хватает.
- Чтобы запустить без туннеля, уберите `COMPOSE_PROFILES=tunnel` из `.env`.

### Ключи

| Переменная | Где взять |
|---|---|
| `GROQ_API_KEY` | [console.groq.com/keys](https://console.groq.com/keys) |
| `ELEVENLABS_API_KEY` | elevenlabs.io → Developers → API Keys |
| `ELEVENLABS_VOICE_ID` | ElevenLabs → Voices. На бесплатном тарифе голоса из Voice Library через API недоступны (`402`), подойдёт свой голос из Voice Design |

### Публичный доступ

1. В Cloudflare Zero Trust откройте **Networks → Tunnels**, создайте туннель и добавьте public hostname `https://<домен>` → `http://web:80`. Токен туннеля впишите в `CLOUDFLARE_TUNNEL_TOKEN`.
2. **`DEMO_KEY`**: ключ проверяется на WS-хэндшейке, ссылка для друзей выглядит как `https://<домен>/?key=<ключ>`. Ключ сохраняется в localStorage и в access-логи не пишется. Если ключ пустой, бот пишет в лог предупреждение, и звонить может любой. Поверх можно включить Cloudflare Access.
3. Наружу открыт только `cloudflared → web`. Порты `bot` и `llm` не публикуются, а `web` слушает только `127.0.0.1`.

## Ограничения поведения

- Одновременно идёт только один звонок. Второй звонящий видит «Линия занята» (WS close `4409`), первый звонок при этом не прерывается.
- Звонок длится максимум `MAX_CALL_SECONDS` (600 с). Если пользователь молчит дольше `IDLE_TIMEOUT_SECONDS` (60 с), звонок сбрасывается.
- Барж-ин: если начать говорить, бот замолкает, генерация и TTS отменяются, фронт сбрасывает буфер воспроизведения.
- Аудио нигде не записывается, но речь уходит в Groq и ElevenLabs. Транскрипты пишутся в лог только при `LOG_LEVEL=DEBUG`.

## Разработка

```bash
# сгенерировать тестовые фразы (ElevenLabs) и позвонить через nginx (внутри контейнера bot)
docker compose exec -T bot python - "Здравствуйте, это Каспи банк." "Назовите код из SMS." < scripts/make_test_speech.py
docker compose exec -T bot python - /tmp/u0.wav /tmp/u1.wav --url ws://web/ws/call --key <DEMO_KEY> --barge-in < scripts/test_call.py

# юнит-тесты бота
docker compose exec -T bot python -m unittest discover -s tests

# фронт с hot reload (прокси /ws и /api на localhost:8000)
cd frontend && npm ci && npm run dev
```

`scripts/test_call.py` проверяет сквозной путь, `busy` / 4409 и barge-in. Фронт проверяется `npm run build` (там же `tsc --noEmit`).
