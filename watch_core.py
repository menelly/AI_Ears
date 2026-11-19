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
        heard = hear_core.hear(mp3, lang=lang)
        card = hear_core.format_card(heard)
        _save_words(out_dir, heard)  # 🗂️ so look_at() can find a word later WITHOUT re-transcribing
    elif not mp3:
        card = "🔇 No audio track in this clip: eyes only."

    return {"title": title, "duration_s": round(dur, 1), "frames": cols * rows,
            "grid": grid, "mp3": mp3, "video": video, "card": card}


# ─────────────────────────────────────────────────────────────────────────────
#  🔎 look_at — the zoom lens (idea: Ren, 2026-11-18 19:10)
#
#  The 12-frame grid is a skim. Sometimes the transcript says something that
#  makes no sense ("the newly MAGNIFIED CBS"), and the on-screen caption has
#  the real word ("MAGA-fied"), but it happened BETWEEN two of the 12 frames.
#  look_at() goes and gets the frames at that exact moment.
#
#  Deciding a word "makes no sense" is a JUDGMENT, so it stays with the mind
#  reading the transcript. The tool only does the fetching: hand it a word or
#  phrase from the transcript (or plain seconds) and it returns the frames.
# ─────────────────────────────────────────────────────────────────────────────

WORDS_FILE = "words.json"
CAPTION_OFFSETS = (-0.3, 0.5, 1.3)  # 🎞️ captions run a little ahead of or behind the voice, so 3 shots per moment


def _save_words(out_dir, heard):
    words = (heard.get("raw_stt") or {}).get("wordTimestamps") or []
    if words:
        with open(os.path.join(out_dir, WORDS_FILE), "w", encoding="utf-8") as f:
            json.dump(words, f, ensure_ascii=False)


def _norm(w):
    return "".join(ch for ch in w.lower() if ch.isalnum())


def _load_or_transcribe_words(video, lang="en"):
    # 🗂️ watch() leaves words.json beside the video; if it isn't there, transcribe once and keep it
    folder = os.path.dirname(os.path.abspath(video))
    path = os.path.join(folder, WORDS_FILE)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    mp3 = os.path.join(folder, "reel.mp3")
    if not os.path.exists(mp3):
        mp3 = extract_audio(video, mp3)
    if not mp3:
        raise RuntimeError("this clip has no audio, so there are no words to search; pass seconds instead")
    heard = hear_core.hear(mp3, lang=lang)
    _save_words(folder, heard)
    words = (heard.get("raw_stt") or {}).get("wordTimestamps") or []
    if not words:
        raise RuntimeError("no word timestamps available (is a speech backend configured?); pass seconds instead")
    return words


def find_phrase(words, phrase):
    """⏱️ Start times (seconds) of every place `phrase` occurs in the word timestamps."""
    target = [_norm(w) for w in phrase.split() if _norm(w)]
    toks = [_norm(w.get("word", "")) for w in words]
    hits = []
    for i in range(len(toks) - len(target) + 1):
        if toks[i:i + len(target)] == target:
            hits.append(words[i].get("startTimeMs", 0) / 1000)
    return hits


def look_at(video, at, out_path=None, lang="en", max_moments=4, frame_width=480):
    """🔎 Frames at exact moments. `at` = a word/phrase from the transcript, or seconds ("80.5, 112").

    Returns {"image": path, "moments": [seconds...], "note": str}. Each moment
    becomes one row of 3 frames (just before / at / just after) so a caption
    that lags the voice still gets caught.
    """
    video = os.path.abspath(video)
    if not os.path.exists(video):
        raise FileNotFoundError(video)
    dur = duration_seconds(video)

    tokens = [t.strip() for t in str(at).replace(";", ",").split(",") if t.strip()]
    try:
        moments = [float(t) for t in tokens]
        how = "seconds"
    except ValueError:
        moments = find_phrase(_load_or_transcribe_words(video, lang), str(at))
        how = f'the word(s) "{at}"'
        if not moments:
            raise RuntimeError(f'"{at}" is not in the transcript\'s word timestamps (check the spelling the TRANSCRIPT used)')

    note = f"{len(moments)} moment(s) for {how}"
    if len(moments) > max_moments:
        note += f"; showing the first {max_moments}"
        moments = moments[:max_moments]

    folder = tempfile.mkdtemp(prefix="look_")
    n = 0
    for t in moments:
        for off in CAPTION_OFFSETS:
            ts = min(max(t + off, 0), max(dur - 0.05, 0))
            # 📸 -ss BEFORE -i = fast seek; one frame, scaled so every tile matches
            _run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{ts:.3f}", "-i", video,
                  "-frames:v", "1", "-vf", f"scale={frame_width}:-2",
                  os.path.join(folder, f"f_{n:03d}.jpg")], "frame grab")
            n += 1

    out_path = out_path or os.path.join(os.path.dirname(video), f"look_{int(time.time())}.jpg")
    _run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", os.path.join(folder, "f_%03d.jpg"),
          "-vf", f"tile={len(CAPTION_OFFSETS)}x{len(moments)}", "-frames:v", "1", out_path], "tiling")
    shutil.rmtree(folder, ignore_errors=True)
    return {"image": out_path, "moments": [round(m, 2) for m in moments], "note": note}


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
