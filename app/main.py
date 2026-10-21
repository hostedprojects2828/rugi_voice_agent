import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from app.conversation import reply, run_conversation

ROOT = Path(__file__).resolve().parent.parent


def play_audio(path: Path) -> None:
    """Play generated speech through the local system's audio output."""
    players = (
        [("afplay", [])] if sys.platform == "darwin" else
        [("ffplay", ["-nodisp", "-autoexit", "-loglevel", "error"]),
         ("mpg123", ["-q"])]
    )
    for name, options in players:
        executable = shutil.which(name)
        if executable:
            try:
                subprocess.run([executable, *options, str(path)], check=True)
            except (OSError, subprocess.CalledProcessError) as error:
                print(f"Could not play speech: {error}. Audio saved at {path}",
                      file=sys.stderr)
            return
    print(f"No audio player found. Open {path} to hear the reply.", file=sys.stderr)


def speak(client, text, no_play=False):
    output = ROOT / "output" / "agent_response.mp3"
    output.parent.mkdir(parents=True, exist_ok=True)
    with client.audio.speech.with_streaming_response.create(
        model=os.getenv("TTS_MODEL", "gpt-4o-mini-tts"),
        voice=os.getenv("TTS_VOICE", "alloy"),
        input=text,
        response_format="mp3",
    ) as response:
        response.stream_to_file(output)
    print("Saved generated speech:", output)
    if not no_play:
        play_audio(output)


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Sample-audio LangGraph wake-up agent")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--audio", help="Transcribe an existing recording")
    source.add_argument(
        "--text", help="Test without audio by supplying a spoken message")
    source.add_argument("--mic", action="store_true", help="Start a microphone conversation (default)")
    parser.add_argument("--silence-seconds", type=float, default=1.5)
    parser.add_argument("--mic-threshold", type=int, default=500,
                        help="Speech detection level; lower for a quiet microphone")
    parser.add_argument(
        "--no-play", action="store_true", help="Save speech without playing it")
    args = parser.parse_args()
    if args.silence_seconds <= 0 or args.mic_threshold <= 0:
        parser.error("Silence duration and microphone threshold must be positive")

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "replace_with_your_api_key":
        raise SystemExit("Copy .env.example to .env and set OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    if args.mic or (args.text is None and args.audio is None):
        try:
            run_conversation(client, lambda text: speak(client, text, args.no_play),
                             args.silence_seconds, args.mic_threshold)
        except KeyboardInterrupt:
            print("\nConversation ended.")
        except Exception as error:
            raise SystemExit(f"Conversation stopped: {error}") from error
        return

    if args.text:
        user_message = args.text.strip()
    else:
        audio_path = Path(args.audio).expanduser()
        if not audio_path.is_file():
            raise SystemExit(
                f"Audio file not found: {audio_path}. Add audio/sample.wav or use --text.")
        with audio_path.open("rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=os.getenv("TRANSCRIPTION_MODEL",
                                "gpt-4o-mini-transcribe"),
                file=audio_file,
            )
        user_message = transcription.text.strip()

    print("You said:", user_message)
    text = reply(client, user_message, [])
    print("Agent response:", text)
    speak(client, text, args.no_play)


if __name__ == "__main__":
    main()
