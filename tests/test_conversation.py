import io
import wave
from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.conversation import record_microphone, reply, run_conversation


def test_reply_keeps_prior_turns():
    client = Mock()
    output = {"role": "assistant", "content": "Try sitting up."}
    client.responses.create.return_value = SimpleNamespace(
        output_text="Try sitting up.", output=[output])
    history = []
    reply(client, "I'm tired", history)
    reply(client, "What did you suggest?", history)
    sent = client.responses.create.call_args.kwargs["input"]
    assert sent == [
        {"role": "user", "content": "I'm tired"}, output,
        {"role": "user", "content": "What did you suggest?"}]


def test_recording_ends_after_silence_and_produces_wav():
    stream = Mock()
    stream.__enter__ = Mock(return_value=stream)
    stream.__exit__ = Mock(return_value=False)
    loud = array("h", [1000] * 1600).tobytes()
    quiet = bytes(3200)
    stream.read.side_effect = [(quiet, False), (loud, False),
                               (quiet, False), (quiet, False)]
    sd = SimpleNamespace(RawInputStream=Mock(return_value=stream))
    with patch.dict("sys.modules", {"sounddevice": sd}):
        audio = record_microphone(silence_seconds=0.2)
    with wave.open(audio) as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getnframes() == 6400
    assert stream.read.call_count == 4


def test_loop_plays_reply_before_listening_again():
    events = []
    def record(*args):
        events.append("listen")
        return io.BytesIO(b"audio")
    def speak(text):
        events.append(text)
    with patch("app.conversation.record_microphone", side_effect=record), \
         patch("app.conversation.transcribe", side_effect=["Hello", "stop conversation"]), \
         patch("app.conversation.reply", return_value="Good morning!"):
        run_conversation(Mock(), speak)
    assert events == ["Good morning, wake up Mone", "listen", "Good morning!",
                      "listen", "Goodbye! Enjoy your morning."]


def test_silence_does_not_trigger_transcription():
    with patch("app.conversation.record_microphone", side_effect=[None, KeyboardInterrupt]), \
         patch("app.conversation.transcribe") as transcribe:
        try:
            run_conversation(Mock(), Mock())
        except KeyboardInterrupt:
            pass
    transcribe.assert_not_called()
