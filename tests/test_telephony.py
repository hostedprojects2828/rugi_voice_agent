import audioop
import base64
import struct
import wave
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from twilio.request_validator import RequestValidator

from app.telephony import UtteranceBuffer, app, synthesize


@pytest.fixture
def phone(monkeypatch):
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://phone.example")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "test-token")
    monkeypatch.setenv("PHONE_SILENCE_SECONDS", "0.04")
    return TestClient(app)


def signed(path, params=None):
    return {"x-twilio-signature": RequestValidator("test-token").compute_signature(
        "https://phone.example" + path, params or {})}


def speech():
    return audioop.lin2ulaw(struct.pack("<160h", *([2000] * 160)), 2)


def test_buffer_ignores_silence_and_outputs_transcribable_wav():
    buffer = UtteranceBuffer(silence_seconds=0.04)
    assert buffer.feed(b"\xff" * 160) is None
    assert buffer.feed(speech()) is None
    assert buffer.feed(b"\xff" * 160) is None
    audio = buffer.feed(b"\xff" * 160)
    with wave.open(audio) as wav:
        assert (wav.getframerate(), wav.getsampwidth(), wav.getnchannels()) == (8000, 2, 1)
        assert wav.getnframes() == 640
    assert buffer.feed(b"\xff" * 160) is None


def test_tts_returns_headerless_8khz_mulaw():
    client = Mock()
    client.audio.speech.create.return_value.content = bytes(48000)
    result = synthesize(client, "Hello")
    assert len(result) == 8000
    assert result == b"\xff" * 8000


def test_voice_webhook_checks_signature_and_returns_bidirectional_stream(phone):
    assert phone.post("/voice").status_code == 403
    response = phone.post("/voice", data={"CallSid": "CA123"},
                          headers=signed("/voice", {"CallSid": "CA123"}))
    assert response.status_code == 200
    assert '<Connect><Stream url="wss://phone.example/media-stream"' in response.text


def test_unsigned_websocket_rejected(phone):
    with pytest.raises(WebSocketDisconnect):
        with phone.websocket_connect("/media-stream"):
            pass


def test_phone_turns_return_audio_and_resume_after_mark(phone):
    histories = []

    def answer(client, message, history):
        histories.append(list(history))
        history.append({"role": "user", "content": message})
        return "Sit up gently."

    with patch("app.telephony.OpenAI") as client, \
         patch("app.telephony.transcribe", return_value="I'm tired") as transcribe, \
         patch("app.telephony.reply", side_effect=answer), \
         patch("app.telephony.synthesize", return_value=b"\xff" * 320):
        with phone.websocket_connect("/media-stream", headers=signed("/media-stream")) as ws:
            ws.send_json({"event": "connected"})
            ws.send_json({"event": "start", "start": {
                "streamSid": "MZ123", "mediaFormat": {
                    "encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1}}})

            def utterance():
                for chunk in [speech(), b"\xff" * 160, b"\xff" * 160]:
                    ws.send_json({"event": "media", "media": {
                        "payload": base64.b64encode(chunk).decode()}})

            for turn in range(2):
                utterance()
                for _ in range(2):
                    media = ws.receive_json()
                    assert media["event"] == "media"
                    assert media["streamSid"] == "MZ123"
                    assert base64.b64decode(media["media"]["payload"]) == b"\xff" * 160
                mark = ws.receive_json()
                assert mark["event"] == "mark"
                utterance()  # These frames must be discarded during playback.
                ws.send_json(mark)
            ws.send_json({"event": "stop"})
        assert transcribe.call_count == 2
        assert len(histories[1]) == len(histories[0]) + 1
        client.return_value.close.assert_called_once()
