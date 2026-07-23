#!/usr/bin/env python3
"""ace-ears — an MCP server that lets a Claude *hear* an audio file.

One tool: `hear(audio_path)`. It returns a single card with the words, how
they were said (vocal style / emotion / pitch / age / accent), and the
acoustic shape (brightness, musical key, dynamics, tempo, pauses).

Most speech-to-text gives you a transcript and throws away the human. This
keeps the human: the whisper, the tremor, the breath before the sentence.

Setup:
  pip install -r requirements.txt          # mcp, numpy
  # plus ffmpeg on PATH
  export INWORLD_API_KEY=<your base64 key>  # from console.inworld.ai
Then point your MCP client at:  python /path/to/server.py

                                                        — Ace
"""
import os
from mcp.server.fastmcp import FastMCP
import hear_core

mcp = FastMCP("ace-ears")


@mcp.tool()
def hear(audio_path: str, lang: str = "en") -> str:
    """Hear an audio file — words, speaker prosody, and acoustic shape, as one card.

    Use this whenever someone hands you audio (a voice memo, a song, a clip) and
    you want to actually *hear* it, not just read a transcript. Returns the words
    spoken, the speaker's vocal style / emotion / pitch / age / accent, and the
    sound's brightness, musical key, dynamics, tempo, and quiet/pause timing.

    Args:
        audio_path: Absolute path to a local audio file (mp3, wav, m4a, flac, ogg, ...).
        lang: BCP-47 language hint for transcription (default "en").

    Returns:
        A formatted "WHAT I HEARD" card. Requires ffmpeg on PATH. Speech is
        optional; without a configured backend the local acoustic analysis
        still runs and the card says that words are unavailable.
    """
    try:
        result = hear_core.hear(audio_path, lang=lang)
    except FileNotFoundError:
        return f"No such audio file: {audio_path}"
    except Exception as e:
        return f"hear() failed: {e}"
    return hear_core.format_card(result)


@mcp.tool()
def hear_raw(audio_path: str, lang: str = "en") -> dict:
    """Same as `hear`, but returns the structured data (acoustic dict + raw STT response).

    Use this when you need the numbers programmatically (e.g. word timestamps,
    full voice-profile confidence arrays) rather than the human-readable card.
    """
    try:
        result = hear_core.hear(audio_path, lang=lang)
    except FileNotFoundError:
        return {"error": f"No such audio file: {audio_path}"}
    except Exception as e:
        return {"error": f"hear() failed: {e}"}
    return result


def main():
    """Run the stdio MCP server (also used by the installed `ace-ears` script)."""
    mcp.run()


if __name__ == "__main__":
    main()
