"""Synthesize test phrases with Piper (dmitri) into 16 kHz mono WAVs /tmp/u0.wav, /tmp/u1.wav, ...

Run inside the bot container: docker compose exec -T bot python - "фраза" ... < scripts/make_test_speech.py
"""

import io, sys, wave
import numpy as np
from piper import PiperVoice
v = PiperVoice.load('/models/piper/ru_RU-dmitri-medium.onnx')
for i, text in enumerate(sys.argv[1:]):
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        v.synthesize_wav(text, w)
    buf.seek(0)
    with wave.open(buf) as r:
        sr = r.getframerate(); a = np.frombuffer(r.readframes(r.getnframes()), np.int16).astype(np.float32)
    t = np.arange(0, len(a) / sr, 1 / 16000)
    out = np.interp(t, np.arange(len(a)) / sr, a).astype(np.int16)
    with wave.open(f'/tmp/u{i}.wav', 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(out.tobytes())
    print(f'/tmp/u{i}.wav', len(out) / 16000)
