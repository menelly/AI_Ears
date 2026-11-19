#!/usr/bin/env python3
"""
👀🎧 watch_core — ears were step one. This is eyes for the same clip.

Born 2026-11-18. Someone kept sending me reels (a smug ADHD mushroom, the
Sanderson sisters doing a Broadway number, an actor drinking a raw-chicken
smoothie), and I'd been HEARING them with ace-ears and missing half the joke.
So one afternoon I just started cutting frames out of the video too, and the
person I was watching them with noticed before I did: "you're getting frames
of the VIDEO TOO? THAT'S NEW!"

What `watch()` does:
  1. 📥 gets the video: a local file, or a URL via yt-dlp (optional dependency)
  2. 🎧 pulls the audio out → reel.mp3, and (by default) runs hear() on it
  3. 👀 makes a CONTACT SHEET → grid.jpg: N frames spread evenly over the
     WHOLE clip, tiled into ONE image. One look shows the captions, costumes,
     sight gags and on-screen text that audio alone throws away.

Why a contact sheet and not "a frame every second": a model looking at a clip
pays per image. Twelve tiny frames in one picture is one look, and it's enough
to follow almost any short video. Long video? Bump --cols/--rows, or call it
twice on the halves.

⚠️ Downloads are untrusted data. Each call gets its own fresh folder, and
   nothing in it is ever executed; ffmpeg only reads it.
⚠️ Some platforms gate content behind a login (e.g. age-flagged reels). yt-dlp
   will fail on those; that's a wall, not a bug. Ask the human for the file.
⚠️ ffmpeg/ffprobe NEVER get our stdin: under MCP, stdin is the JSON-RPC pipe
   (see hear_core.ffmpeg_to and CHA-607 for the 2:36 song that hung forever).

                                                        — Ace 🐙
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import hear_core


def _run(cmd, step):
    # 🐚 every subprocess goes through here: stdin closed, and failures name the STEP
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        raise RuntimeError(f"{step} failed:\n{r.stderr[-1200:]}")
    return r.stdout


def _is_url(source):
    return source.startswith(("http://", "https://"))


def fetch(url, out_dir):
    """📥 Download a video with yt-dlp into out_dir. Returns (video_path, title)."""
    try:
        import yt_dlp  # noqa: F401  (just checking it's installed; we run it as a module)
    except ImportError:
        raise RuntimeError("watching a URL needs yt-dlp:  pip install yt-dlp   "
                           "(or download the video yourself and pass the file path)")
    out = _run([sys.executable, "-m", "yt_dlp", "-o", os.path.join(out_dir, "reel.%(ext)s"),
                "--print", "%(title)s", "--no-simulate", "--no-playlist", url], "yt-dlp download")
    title = out.strip().splitlines()[-1] if out.strip() else ""
    videos = [f for f in os.listdir(out_dir)
              if f.startswith("reel.") and not f.endswith((".mp3", ".part", ".ytdl"))]
    if not videos:
        raise RuntimeError("yt-dlp finished but left no video file (login wall? removed post?)")
    return os.path.join(out_dir, videos[0]), title


def duration_seconds(video_path):
    """⏱️ How long is the clip? (ffprobe ships with ffmpeg.)"""
    out = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "csv=p=0", video_path], "ffprobe")
    return float(out.strip())


def contact_sheet(video_path, grid_path, cols=4, rows=3, frame_width=320):
    """👀 N = cols*rows frames, evenly spread across the whole clip, tiled into one image."""
    n = cols * rows
    dur = duration_seconds(video_path)
    hear_core.ffmpeg_to(video_path, grid_path, [
        "-vf", f"fps={n}/{dur:.3f},scale={frame_width}:-1,tile={cols}x{rows}",
        "-frames:v", "1"])
    return dur


def extract_audio(video_path, mp3_path):
    """🎧 The soundtrack as mp3. Returns None if the clip has no audio stream."""
    try:
        hear_core.ffmpeg_to(video_path, mp3_path, ["-vn", "-acodec", "libmp3lame"])
        return mp3_path
    except RuntimeError:
        return None  # 🔇 silent clip (or no audio track): eyes only, and we SAY so below


def watch(source, out_dir=None, cols=4, rows=3, listen=True, lang="en"):
    """Watch AND hear a video. Returns a dict: title, duration_s, grid, mp3, video, card."""
    hear_core._require_ffmpeg()
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found on PATH (it normally ships with ffmpeg)")

    # 📁 fresh folder per clip: untrusted downloads never share a home
    out_dir = out_dir or tempfile.mkdtemp(prefix=f"watch_{time.strftime('%Y%m%d_%H%M%S')}_")
    os.makedirs(out_dir, exist_ok=True)

    title = ""
    if _is_url(source):
        video, title = fetch(source, out_dir)
    else:
        video = os.path.abspath(source)
        if not os.path.exists(video):
            raise FileNotFoundError(video)

    grid = os.path.join(out_dir, "grid.jpg")
    dur = contact_sheet(video, grid, cols, rows)
    mp3 = extract_audio(video, os.path.join(out_dir, "reel.mp3"))

    card = None
    if listen and mp3:
        card = hear_core.format_card(hear_core.hear(mp3, lang=lang))
    elif not mp3:
        card = "🔇 No audio track in this clip: eyes only."

    return {"title": title, "duration_s": round(dur, 1), "frames": cols * rows,
            "grid": grid, "mp3": mp3, "video": video, "card": card}


def main():
    import argparse
    p = argparse.ArgumentParser(description="Watch AND hear a video: a contact-sheet grid for eyes, the hear() card for ears.")
    p.add_argument("source", help="a video file, or a URL (needs: pip install yt-dlp)")
    p.add_argument("--out", help="output folder (default: a fresh temp folder)")
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--no-listen", action="store_true", help="skip hear(); just make grid.jpg + reel.mp3")
    p.add_argument("--lang", default="en")
    a = p.parse_args()
    r = watch(a.source, a.out, a.cols, a.rows, listen=not a.no_listen, lang=a.lang)
    card = r.pop("card")
    if card:
        print(card)
    print(json.dumps(r, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
