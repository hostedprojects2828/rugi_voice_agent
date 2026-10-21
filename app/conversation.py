import io
import os
import wave
from array import array
from collections import deque

from app.graph import wake_up_graph


def record_microphone(silence_seconds=1.5, threshold=500, max_seconds=30):
    """Record one utterance, ending after silence; return None if nobody speaks."""
    try:
        import sounddevice as sd
    except ImportError:
        raise RuntimeError(
            "Install microphone support: pip install -r requirements.txt") from None

    rate, block = 16000, 1600
    frames = []
    preroll = deque(maxlen=3)
    silent_blocks = 0
    print("Listening… speak now (Ctrl+C to stop).", flush=True)
    with sd.RawInputStream(samplerate=rate, channels=1, dtype="int16") as stream:
        for _ in range(int(max_seconds * rate / block)):
            data, overflowed = stream.read(block)
            if overflowed:
                raise RuntimeError(
                    "Microphone overflow. Try closing other audio apps.")
            chunk = bytes(data)
            samples = array("h", chunk)
            rms = (sum(sample * sample for sample in samples) /
                   len(samples)) ** 0.5
            if not frames:
                preroll.append(chunk)
                if rms < threshold:
                    continue
                frames.extend(preroll)
            else:
                frames.append(chunk)
            silent_blocks = silent_blocks + 1 if rms < threshold else 0
            if silent_blocks * block / rate >= silence_seconds:
                break
    if not frames:
        return None
    audio = io.BytesIO()
    with wave.open(audio, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"".join(frames))
    audio.seek(0)
    audio.name = "microphone.wav"
    return audio


def transcribe(client, audio):
    return client.audio.transcriptions.create(
        model=os.getenv("TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe"),
        file=audio,
    ).text.strip()


def reply(client, message, history):
    state = wake_up_graph.invoke(
        {"user_message": message, "action": "", "agent_response": ""})
    user_turn = {"role": "user", "content": message}
    response = client.responses.create(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        instructions=(
            "You are a friendly wake-up companion having a spoken conversation. "
            "Respond naturally to what the user says and remember the conversation. "
            "Use one to three short sentences, no markdown. Help them wake up gently. "
            f"Current wake-up intent: {state['action']}. "
            "If they are awake, congratulate them without forcing the conversation to end."
        ),
        input=[*history, user_turn],
        store=False,
    )
    text = response.output_text.strip()
    if not text:
        raise RuntimeError(
            "The LLM returned an empty reply. Please try again.")
    history.extend([user_turn, *response.output])
    return text


def run_conversation(client, speak, silence_seconds=1.5, threshold=500):
    greeting = "Good morning, wake up Mone"
    history = [{"role": "assistant", "content": greeting}]
    print("Voice conversation started. Say 'stop conversation' or press Ctrl+C to exit.")
    print("Agent response:", greeting)
    speak(greeting)
    while True:
        audio = record_microphone(silence_seconds, threshold)
        if audio is None:
            continue
        with audio:
            message = transcribe(client, audio)
        if not message:
            continue
        print("You said:", message)
        if message.lower().strip(" .!?,") in {"stop conversation", "end conversation", "goodbye"}:
            speak("Goodbye! Enjoy your morning.")
            return
        text = reply(client, message, history)
        print("Agent response:", text)
        speak(text)
