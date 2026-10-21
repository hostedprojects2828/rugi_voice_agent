"""Twilio Media Streams transport for the existing turn-based agent."""

import asyncio
import audioop
import base64
import binascii
import io
import logging
import os
import wave
from collections import deque
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from openai import OpenAI
from twilio.request_validator import RequestValidator
from twilio.twiml.voice_response import VoiceResponse

from app.conversation import reply, transcribe

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
app = FastAPI()
logger = logging.getLogger(__name__)


def public_url(path):
    base = os.environ["PUBLIC_BASE_URL"].rstrip("/")
    if not base.startswith("https://"):
        raise ValueError("PUBLIC_BASE_URL must start with https://")
    return base + path


def valid_signature(url, params, signature):
    return RequestValidator(os.environ["TWILIO_AUTH_TOKEN"]).validate(
        url, params, signature or "")


class UtteranceBuffer:
    """Detect a turn using PCM RMS, retaining a short speech pre-roll."""

    def __init__(self, threshold=500, silence_seconds=1.5, max_seconds=30):
        self.threshold = threshold
        self.silence_bytes = int(silence_seconds * 16000)
        self.max_bytes = int(max_seconds * 16000)
        self.reset()

    def reset(self):
        self.frames = bytearray()
        self.preroll = deque(maxlen=10)
        self.quiet_bytes = 0

    def feed(self, mulaw):
        pcm = audioop.ulaw2lin(mulaw, 2)
        if not pcm:
            return None
        quiet = audioop.rms(pcm, 2) < self.threshold
        if not self.frames:
            self.preroll.append(pcm)
            if quiet:
                return None
            self.frames.extend(b"".join(self.preroll))
            self.preroll.clear()
        else:
            self.frames.extend(pcm)
        self.quiet_bytes = self.quiet_bytes + len(pcm) if quiet else 0
        if self.quiet_bytes < self.silence_bytes and len(self.frames) < self.max_bytes:
            return None
        audio = io.BytesIO()
        with wave.open(audio, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(8000)
            wav.writeframes(self.frames)
        audio.seek(0)
        audio.name = "call.wav"
        self.reset()
        return audio


def synthesize(client, text):
    # Speech PCM is mono signed 16-bit little-endian at 24 kHz.
    response = client.audio.speech.create(
        model=os.getenv("TTS_MODEL", "gpt-4o-mini-tts"),
        voice=os.getenv("TTS_VOICE", "alloy"),
        input=text,
        response_format="pcm",
    )
    pcm, _ = audioop.ratecv(response.content, 2, 1, 24000, 8000, None)
    return audioop.lin2ulaw(pcm, 2)


@app.post("/voice")
async def voice(request: Request):
    params = await request.form()
    if not valid_signature(public_url("/voice"), params,
                           request.headers.get("x-twilio-signature")):
        raise HTTPException(status_code=403)
    twiml = VoiceResponse()
    twiml.say("You are speaking with an AI wake-up companion. Good morning! How are you feeling?")
    twiml.connect().stream(url=public_url("/media-stream").replace("https://", "wss://", 1))
    return Response(str(twiml), media_type="application/xml")


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    # Twilio signs the HTTPS handshake URL, even for a WSS connection.
    if not valid_signature(public_url("/media-stream"), {},
                           websocket.headers.get("x-twilio-signature")):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    client = OpenAI(timeout=60, max_retries=1)
    history = [{"role": "assistant", "content": "Good morning! How are you feeling?"}]
    buffer = UtteranceBuffer(
        threshold=float(os.getenv("PHONE_SPEECH_THRESHOLD", "500")),
        silence_seconds=float(os.getenv("PHONE_SILENCE_SECONDS", "1.5")),
    )
    stream_sid = None
    busy = False
    pending_mark = None
    worker = None
    turn = 0

    async def respond(audio):
        nonlocal busy, pending_mark, turn
        try:
            with audio:
                message = await asyncio.to_thread(transcribe, client, audio)
            if not message:
                busy = False
                return
            text = await asyncio.to_thread(reply, client, message, history)
            mulaw = await asyncio.to_thread(synthesize, client, text)
            turn += 1
            pending_mark = f"reply-{turn}"
            for offset in range(0, len(mulaw), 160):
                await websocket.send_json({
                    "event": "media", "streamSid": stream_sid,
                    "media": {"payload": base64.b64encode(mulaw[offset:offset + 160]).decode("ascii")},
                })
            # Stay busy until Twilio confirms the buffered reply has played.
            await websocket.send_json({"event": "mark", "streamSid": stream_sid,
                                       "mark": {"name": pending_mark}})
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Could not generate telephone reply")
            await websocket.close(code=1011)

    try:
        while True:
            data = await websocket.receive_json()
            event = data.get("event")
            if event == "start":
                if stream_sid is not None:
                    await websocket.close(code=1008)
                    return
                fmt = data["start"]["mediaFormat"]
                if fmt != {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1}:
                    await websocket.close(code=1003)
                    return
                stream_sid = data["start"]["streamSid"]
            elif event == "media" and stream_sid and not busy:
                audio = buffer.feed(base64.b64decode(data["media"]["payload"], validate=True))
                if audio is not None:
                    busy = True
                    worker = asyncio.create_task(respond(audio))
            elif event == "mark" and pending_mark and data["mark"]["name"] == pending_mark:
                pending_mark = None
                buffer.reset()
                busy = False
            elif event == "stop":
                break
    except WebSocketDisconnect:
        pass
    except (KeyError, ValueError, binascii.Error):
        await websocket.close(code=1008)
    finally:
        if worker is not None:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        # A cancelled to_thread operation may still be completing an API request.
        await asyncio.to_thread(client.close)
