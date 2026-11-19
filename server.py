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
from mcp.server.fastmcp import FastMCP, Image
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


@mcp.tool()
def watch(source: str, cols: int = 4, rows: int = 3, listen: bool = True, lang: str = "en"):
    """Watch AND hear a video: a contact sheet of frames for your eyes, plus the hear() card.

    Use this when someone sends you a video or a reel and you want to actually
    SEE it, not just hear the soundtrack. Returns one image (cols*rows frames
    spread evenly across the whole clip, tiled together so a single look shows
    captions, costumes, on-screen text and sight gags) plus the "WHAT I HEARD"
    card for the audio.

    Args:
        source: A local video file path, or a URL (URLs need `pip install yt-dlp`;
            login-gated posts will fail, so ask for the file instead).
        cols, rows: Contact-sheet layout (default 4x3 = 12 frames; go bigger for long clips).
        listen: Also run hear() on the soundtrack (default True).
        lang: Transcription language hint for the audio half.
    """
    import watch_core
    try:
        r = watch_core.watch(source, cols=cols, rows=rows, listen=listen, lang=lang)
    except FileNotFoundError:
        return f"No such video file: {source}"
    except Exception as e:
        return f"watch() failed: {e}"
    header = (f"👀🎧 WATCHED {r['title'] or os.path.basename(r['video'])} · {r['duration_s']}s · "
              f"{r['frames']} frames in the grid below\n   files: {r['grid']} · {r['mp3']}"
              f"\n   video: {r['video']}   (🔎 a transcript word that makes no sense? look_at(video, \"that word\"))")
    return [header + ("\n\n" + r["card"] if r["card"] else ""), Image(path=r["grid"])]


@mcp.tool()
def look_at(video: str, at: str, lang: str = "en"):
    """Zoom in on exact moments of a video: grab the frames where a word was said.

    The watch() grid is a skim of 12 frames. When a transcript word makes no
    sense in context (speech-to-text heard "the newly magnified CBS" but the
    on-screen caption says "MAGA-fied"), the caption you need probably fell
    BETWEEN two grid frames. Hand this tool the odd word or phrase exactly as
    the TRANSCRIPT spelled it, and it finds every moment it was said and returns
    3 frames per moment (just before / during / just after, because captions
    lag or lead the voice). You can also pass seconds instead: "80.5, 112".

    Args:
        video: Path to the video (watch() prints it on its "video:" line).
        at: A word or phrase from the transcript, or comma-separated seconds.
        lang: Transcription language, only used if this clip hasn't been heard yet.
    """
    import watch_core
    try:
        r = watch_core.look_at(video, at, lang=lang)
    except FileNotFoundError:
        return f"No such video file: {video}"
    except Exception as e:
        return f"look_at() failed: {e}"
    return [f"🔎 {r['note']}: at {r['moments']} s. One row per moment (before / during / after).",
            Image(path=r["image"])]


def main():
    """Run the stdio MCP server (also used by the installed `ace-ears` script)."""
    mcp.run()


if __name__ == "__main__":
    main()
