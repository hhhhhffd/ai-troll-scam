"""Wire format of /ws/call (see PROTOCOL.md): raw PCM serializer and a bridge for UI events."""

import json

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.base_serializer import FrameSerializer

INPUT_SAMPLE_RATE = 16000


class RawPcmSerializer(FrameSerializer):
    """Binary = raw PCM16 mono; text = JSON control messages."""

    def __init__(self, output_sample_rate: int, on_hangup):
        super().__init__()
        self._output_sample_rate = output_sample_rate
        self._on_hangup = on_hangup

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, OutputAudioRawFrame):
            return frame.audio
        if isinstance(frame, InterruptionFrame):
            return json.dumps({"type": "interrupt"})
        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            return json.dumps(frame.message, ensure_ascii=False)
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, bytes):
            return InputAudioRawFrame(audio=data, sample_rate=INPUT_SAMPLE_RATE, num_channels=1)
        try:
            message = json.loads(data)
        except json.JSONDecodeError:
            return None
        if isinstance(message, dict) and message.get("type") == "hangup":
            await self._on_hangup()
        return None


class UiEventsProcessor(FrameProcessor):
    """Turns TTS text and bot speaking state into JSON messages for the client.

    Sits between tts and transport.output(). Bot(Started|Stopped)Speaking frames travel
    upstream from the output transport and pass through here on their way up.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TTSTextFrame):
            await self._send({"type": "bot_text", "text": frame.text})
        elif isinstance(frame, BotStartedSpeakingFrame):
            await self._send({"type": "state", "state": "speaking"})
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._send({"type": "state", "state": "listening"})

    async def _send(self, message: dict):
        # State must bypass the audio queue (it would arrive after the phrase); text stays in
        # sync with the audio it belongs to.
        frame_cls = (
            OutputTransportMessageUrgentFrame if message["type"] == "state" else OutputTransportMessageFrame
        )
        await self.push_frame(frame_cls(message=message))
