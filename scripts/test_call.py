#!/usr/bin/env python3
"""Dev test client for /ws/call: streams a 16 kHz mono PCM16 WAV in real time and prints replies.

Generate utterances with the bot's own Piper voice (inside the container):
  docker compose exec -T bot python -c "import wave,sys;from piper import PiperVoice;\
v=PiperVoice.load('/models/piper/ru_RU-irina-medium.onnx');\
w=wave.open('/tmp/u.wav','wb');v.synthesize_wav('Здравствуйте, это служба безопасности Каспи банка. Назовите, пожалуйста, код из SMS.',w)"
  docker compose cp bot:/tmp/u.wav /tmp/u.wav && ffmpeg -i /tmp/u.wav -ar 16000 -ac 1 /tmp/u16.wav

Usage: test_call.py UTTERANCE.wav [UTTERANCE2.wav ...] [--url ws://localhost:8000/ws/call]
           [--persona granny] [--key KEY] [--barge-in] [--save BOT.wav]
Runs anywhere with `websockets`; e.g. inside the bot container:
  docker compose exec -T bot python - /tmp/u1.wav < scripts/test_call.py
Each WAV is one turn: it is sent after the bot finished the previous reply.
--barge-in additionally sends the last WAV while the bot is still speaking the greeting.
"""

import argparse
import asyncio
import json
import time
import wave

import websockets

FRAME_MS = 20


def read_pcm(path: str) -> bytes:
    with wave.open(path) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2), (
            f"{path}: need 16 kHz mono PCM16"
        )
        return w.readframes(w.getnframes())


async def stream(ws, pcm: bytes, on_speech_end=lambda: None) -> None:
    step = 16000 * 2 * FRAME_MS // 1000
    for i in range(0, len(pcm), step):
        await ws.send(pcm[i : i + step])
        await asyncio.sleep(FRAME_MS / 1000)
    on_speech_end()
    # trailing silence so VAD can close the turn
    silence = bytes(step)
    for _ in range(1000 // FRAME_MS):
        await ws.send(silence)
        await asyncio.sleep(FRAME_MS / 1000)


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("wavs", nargs="+")
    p.add_argument("--url", default="ws://localhost:8000/ws/call")
    p.add_argument("--persona", default="granny")
    p.add_argument("--key", default="")
    p.add_argument("--barge-in", action="store_true")
    p.add_argument("--save", help="write everything the bot said to this WAV file")
    args = p.parse_args()
    url = f"{args.url}?persona={args.persona}&key={args.key}"

    async with websockets.connect(url) as ws:
        bot_pcm = bytearray()
        sample_rate = 22050  # replaced by ready.sampleRate
        state = {"speech_end": None, "audio_bytes": 0, "last_audio": time.monotonic()}

        async def reader() -> None:
            nonlocal sample_rate
            async for msg in ws:
                now = time.monotonic()
                if isinstance(msg, bytes):
                    if state["speech_end"] is not None:
                        print(f"  LATENCY {now - state['speech_end']:.2f}s (end of speech -> first bot audio)")
                        state["speech_end"] = None
                    bot_pcm.extend(msg)
                    state["audio_bytes"] += len(msg)
                    state["last_audio"] = now
                else:
                    print("<-", msg)
                    if '"ready"' in msg:
                        sample_rate = json.loads(msg)["sampleRate"]

        reader_task = asyncio.create_task(reader())

        async def wait_bot_done(start_timeout: float) -> None:
            """Wait for the bot to start talking, then until its audio has been quiet for a while."""
            start, seen = time.monotonic(), state["audio_bytes"]
            while state["audio_bytes"] == seen and time.monotonic() - start < start_timeout:
                await asyncio.sleep(0.1)
            while time.monotonic() - state["last_audio"] < 1.5:
                await asyncio.sleep(0.2)

        # Second concurrent connection must be refused while this call is active.
        async with websockets.connect(f"{args.url}?key={args.key}") as second:
            print("second connection ->", await second.recv())
            try:
                await second.recv()
            except websockets.ConnectionClosed as e:
                print("second connection closed:", e.rcvd.code if e.rcvd else None)

        await wait_bot_done(5)  # greeting
        pcms = [read_pcm(w) for w in args.wavs]
        for i, pcm in enumerate(pcms):
            print(f"-> turn {i + 1}: sending {len(pcm) / 32000:.1f}s of speech")
            await stream(ws, pcm, lambda: state.update(speech_end=time.monotonic()))
            await wait_bot_done(20)
        if args.barge_in:
            print("-> barge-in: asking, then speaking over the reply")
            seen = state["audio_bytes"]
            sender = asyncio.create_task(stream(ws, pcms[0]))
            while state["audio_bytes"] == seen:  # wait until the bot starts answering
                await asyncio.sleep(0.05)
            await sender
            await stream(ws, pcms[0])  # interrupt expected while the answer is playing
        await ws.send(json.dumps({"type": "hangup"}))
        reader_task.cancel()
        if args.save:
            with wave.open(args.save, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                w.writeframes(bytes(bot_pcm))


asyncio.run(main())
