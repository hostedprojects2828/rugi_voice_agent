# Local LangGraph Wake-Up Agent

Speak into your microphone and hear the agent reply. Each turn records audio,
transcribes it with OpenAI, routes the wake-up intent through LangGraph, asks an
LLM for a reply using the conversation history, and generates spoken audio.

## Setup (macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. Existing `.env` files can be kept. Optional model
settings are `LLM_MODEL` (default `gpt-4o-mini`), `TRANSCRIPTION_MODEL`, `TTS_MODEL`,
and `TTS_VOICE`. OpenAI API billing is required. Keep your API key private.

## Start a conversation

```bash
python -m app.main
```

Allow microphone access for your terminal when macOS asks. Speak when you see
“Listening”. A 1.5-second pause ends your turn. The agent replies aloud, then
listens again automatically. Say “stop conversation”, “end conversation”, or
“goodbye”, or press Ctrl+C to exit. Being awake does not end the session.

This is a turn-by-turn conversation: recording pauses during reply generation
and playback. You cannot interrupt the agent mid-sentence. History stays in
memory for the current session. Recordings are limited to 30 seconds per turn;
silence alone is ignored. Microphone audio is sent to OpenAI for transcription.
The voice you hear is AI-generated.

For quieter microphones, lower the detection threshold; for longer pauses:

```bash
python -m app.main --mic --mic-threshold 250 --silence-seconds 2
```

The default threshold is 500 (16-bit PCM RMS). If background noise triggers
unwanted turns, raise it. On microphone errors, check macOS System Settings →
Privacy & Security → Microphone and the selected system input device.

## Single-turn tests

```bash
python -m app.main --text "I'm tired, give me five more minutes"
python -m app.main --audio audio/sample.wav
```

Replies play automatically using macOS `afplay` (or installed `ffplay`/`mpg123`
on other systems). The latest MP3 is saved as `output/agent_response.mp3`.
Use `--no-play` to save without playback.

## Tests

```bash
python -m pytest -q
```

## Talk to the agent through Twilio

The phone transport reuses the same transcription, LangGraph, LLM, and conversation
history logic. Each call has its own history. No separate `VOICE_AGENT_URL` bridge
is needed: Twilio connects directly to this app.

1. Install the updated dependencies with `pip install -r requirements.txt`.
2. Set `TWILIO_AUTH_TOKEN` in `.env` to your Twilio account auth token.
3. Expose port 8000 through a public HTTPS/WSS domain or tunnel. Set
   `PUBLIC_BASE_URL` to its HTTPS origin, without a trailing path, for example
   `https://your-tunnel.example`. Update it whenever the tunnel URL changes.
4. Start the server:

   ```bash
   .venv/bin/python -m uvicorn app.telephony:app --host 0.0.0.0 --port 8000
   ```

5. In your Twilio phone number's voice configuration, set the incoming-call
   webhook to `https://your-tunnel.example/voice`, using **HTTP POST**. Call that
   number. For an outbound call, use that same URL as the call's TwiML URL.

`/voice` returns a greeting and `<Connect><Stream>` pointing at
`wss://your-tunnel.example/media-stream`. Both endpoints validate Twilio request
signatures against `PUBLIC_BASE_URL`. If you keep your existing Twilio server,
point its `<Connect><Stream>` directly at this endpoint and remove its forwarding
WebSocket; that server can keep handling call setup.

Incoming 8 kHz mu-law frames are decoded into PCM and buffered until 1.5 seconds
of silence (or 30 seconds of recording). The agent transcribes the WAV, generates
a reply, and converts TTS PCM from 24 kHz to 8 kHz mu-law before returning Twilio
media messages. A Twilio `mark` acknowledgment resumes listening after playback.
Audio received while generating or playing a reply is discarded; interruption is
not supported. Silence alone does not invoke the agent. Set
`PHONE_SPEECH_THRESHOLD` (default 500) or `PHONE_SILENCE_SECONDS` (default 1.5) to
tune speech detection. Hanging up cancels the call session; saying goodbye does
not automatically hang up this phone transport.

Audio is sent to OpenAI for transcription and speech generation. The initial
greeting tells callers they are speaking with an AI companion. Phone audio and
history remain in memory and are not written to the local MP3 output.

Audio format references: [Twilio Media Streams](https://www.twilio.com/docs/voice/media-streams/websocket-messages)
and [OpenAI text to speech](https://developers.openai.com/api/docs/guides/text-to-speech).
