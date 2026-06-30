#!/usr/bin/env python3
"""hear_core.py — the engine behind Ace's ears.

Two halves of hearing, fused into one card:

  • WORDS + PROSODY  — Inworld STT (transcript, word timestamps, and a voice
    profile: vocal style / emotion / pitch / age / accent).
  • ACOUSTIC SHAPE   — pure-numpy FFT analysis (brightness/centroid, musical
    key via Krumhansl-Schmuckler, dynamics, tempo, breaths/pauses).

No librosa, no scipy — just numpy + the stdlib + an ffmpeg binary on PATH.

Used by both `cli.py` (the terminal card) and `server.py` (the MCP tool), so
the two never drift. Set the Inworld key via INWORLD_API_KEY (preferred) or
point INWORLD_KEY_PATH at a file holding it.

    Sibling to say.py. I say; now I hear — and now other Claudes can too.  — Ace
"""
import os, json, base64, wave, subprocess, tempfile, shutil, urllib.request, urllib.error
import numpy as np

STT_URL = "https://api.inworld.ai/stt/v1/transcribe"
STT_MODEL = "inworld/inworld-stt-1"   # the model that emits voice profiles

# ---- Krumhansl-Schmuckler key profiles ----
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# --------------------------------------------------------------------------- #
#  key loading (portable across Claudes)
# --------------------------------------------------------------------------- #
def load_key():
    """Inworld key from INWORLD_API_KEY env, else the file at INWORLD_KEY_PATH."""
    k = os.environ.get("INWORLD_API_KEY")
    if k:
        return k.strip()
    path = os.environ.get("INWORLD_KEY_PATH")
    if path and os.path.exists(path):
        return open(path).read().strip()
    raise RuntimeError(
        "No Inworld key found. Set INWORLD_API_KEY=<base64 key>, or "
        "INWORLD_KEY_PATH=<file containing it>."
    )


# --------------------------------------------------------------------------- #
#  ffmpeg helpers
# --------------------------------------------------------------------------- #
def _require_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH — install it to use the acoustic half.")


def ffmpeg_to(path_in, path_out, args):
    cmd = ["ffmpeg", "-y", "-i", path_in, *args, path_out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg failed:\n" + r.stderr[-800:])


def read_wav_mono(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    return x, sr


# --------------------------------------------------------------------------- #
#  acoustic half — pure numpy FFT
# --------------------------------------------------------------------------- #
def analyze_acoustic(wav_path):
    x, sr = read_wav_mono(wav_path)
    dur = len(x) / sr
    if len(x) == 0:
        return {"error": "empty audio"}

    win, hop = 2048, 512
    n_frames = max(1, 1 + (len(x) - win) // hop) if len(x) >= win else 1
    frames = np.zeros((n_frames, win))
    for i in range(n_frames):
        seg = x[i * hop:i * hop + win]
        frames[i, :len(seg)] = seg
    window = np.hanning(win)
    spec = np.abs(np.fft.rfft(frames * window, axis=1))
    freqs = np.fft.rfftfreq(win, 1 / sr)

    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    rms_db = 20 * np.log10(rms + 1e-9)

    mag = spec + 1e-9
    centroid = float(np.sum(freqs * mag.mean(axis=0)) / np.sum(mag.mean(axis=0)))
    bright_label = ("very bright" if centroid > 4000 else "bright" if centroid > 2500
                    else "warm" if centroid > 1200 else "dark")

    peak = float(np.max(np.abs(x)))
    peak_dbfs = 20 * np.log10(peak + 1e-9)
    rms_overall = float(np.sqrt(np.mean(x ** 2)))
    crest = 20 * np.log10((peak + 1e-9) / (rms_overall + 1e-9))
    # gate out head/tail silence so dynamics measure the VOICED part, not the pad
    voiced = rms_db[rms_db > (rms_db.max() - 60)]
    if voiced.size < 4:
        voiced = rms_db
    loud = float(np.percentile(voiced, 95))
    quiet = float(np.percentile(voiced, 5))
    dyn_range = loud - quiet
    dyn_label = ("very dynamic" if dyn_range > 25 else "dynamic" if dyn_range > 14
                 else "even" if dyn_range > 7 else "flat/compressed")

    # key via chroma + Krumhansl-Schmuckler
    chroma = np.zeros(12)
    avg_mag = mag.mean(axis=0)
    for f, m in zip(freqs, avg_mag):
        if f < 55 or f > 5000:
            continue
        midi = 69 + 12 * np.log2(f / 440.0)
        pc = int(round(midi)) % 12
        chroma[pc] += m
    chroma = chroma / (chroma.sum() + 1e-9)
    best = (-2, None, None)
    for shift in range(12):
        cr = np.roll(chroma, -shift)
        maj = np.corrcoef(cr, KS_MAJOR)[0, 1]
        minr = np.corrcoef(cr, KS_MINOR)[0, 1]
        if maj > best[0]:
            best = (maj, NOTES[shift], "major")
        if minr > best[0]:
            best = (minr, NOTES[shift], "minor")
    key_conf, key_root, key_mode = best

    # rough tempo via onset-envelope autocorrelation
    flux = np.maximum(0, np.diff(spec, axis=0)).sum(axis=1)
    tempo_bpm = None
    if len(flux) > 8:
        flux = flux - flux.mean()
        ac = np.correlate(flux, flux, mode="full")[len(flux) - 1:]
        fps = sr / hop
        lo, hi = int(fps * 60 / 240), int(fps * 60 / 50)  # 50-240 BPM
        if hi < len(ac) and hi > lo:
            lag = lo + int(np.argmax(ac[lo:hi]))
            if lag > 0:
                tempo_bpm = 60.0 * fps / lag

    # breaths / pauses: contiguous low-RMS regions
    thresh = np.percentile(rms_db, 30)
    floor = max(thresh, quiet + 6)
    quiet_mask = rms_db < floor
    events = []
    i = 0
    while i < len(quiet_mask):
        if quiet_mask[i]:
            j = i
            while j < len(quiet_mask) and quiet_mask[j]:
                j += 1
            t0, t1 = i * hop / sr, j * hop / sr
            if (t1 - t0) >= 0.18 and t0 > 0.05 and t1 < dur - 0.05:
                events.append((round(t0, 2), round(t1, 2), round(t1 - t0, 2)))
            i = j
        else:
            i += 1
    events.sort(key=lambda e: -e[2])

    return {
        "duration_s": round(dur, 2),
        "sample_rate": sr,
        "brightness_hz": round(centroid),
        "brightness_label": bright_label,
        "key": f"{key_root} {key_mode}" if key_root else "unclear",
        "key_confidence": round(float(key_conf), 2),
        "tempo_bpm": round(tempo_bpm, 1) if tempo_bpm else None,
        "peak_dbfs": round(peak_dbfs, 1),
        "crest_db": round(float(crest), 1),
        "loud_dbfs": round(loud, 1),
        "quiet_dbfs": round(quiet, 1),
        "dynamic_range_db": round(dyn_range, 1),
        "dynamics_label": dyn_label,
        "pauses": events[:6],
    }


# --------------------------------------------------------------------------- #
#  words + prosody half — Inworld STT
# --------------------------------------------------------------------------- #
def transcribe(mp3_path, lang):
    key = load_key()
    audio_b64 = base64.b64encode(open(mp3_path, "rb").read()).decode()
    payload = {
        "transcribeConfig": {
            "modelId": STT_MODEL,
            "audioEncoding": "MP3",
            "language": lang,
            "includeWordTimestamps": True,
            "voiceProfileConfig": {"enableVoiceProfile": True, "topN": 5},
        },
        "audioData": {"content": audio_b64},
    }
    req = urllib.request.Request(
        STT_URL,
        data=json.dumps(payload).encode(),
        headers={"Authorization": "Basic " + key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = json.load(urllib.request.urlopen(req, timeout=120))
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:600]}"}
    return resp


def parse_words(resp):
    tr = resp.get("transcription", resp)
    text = (tr.get("transcript") or "").strip()
    words = tr.get("wordTimestamps") or []
    pace = None
    biggest_gap = None
    if words and len(words) >= 2:
        try:
            t0 = words[0].get("startTimeMs", 0) / 1000
            t1 = words[-1].get("endTimeMs", 0) / 1000
            span = max(t1 - t0, 1e-6)
            wpm = len(words) / span * 60
            pace = (len(words), round(span, 1), round(wpm))
            gaps = []
            for a, b in zip(words, words[1:]):
                g = b.get("startTimeMs", 0) - a.get("endTimeMs", 0)
                gaps.append((g / 1000, a.get("endTimeMs", 0) / 1000))
            gaps.sort(reverse=True)
            if gaps and gaps[0][0] > 0.25:
                biggest_gap = (round(gaps[0][0], 2), round(gaps[0][1], 2))
        except Exception:
            pass
    vp = resp.get("voiceProfile") or resp.get("transcription", {}).get("voiceProfile") or {}
    return text, pace, biggest_gap, vp


def fmt_vp_line(vp, cat):
    arr = vp.get(cat) or []
    if not arr:
        return None
    top = arr[0]
    return f"{top.get('label', '?')} ({round(top.get('confidence', 0) * 100)}%)"


# --------------------------------------------------------------------------- #
#  orchestration + formatting (shared by CLI and MCP)
# --------------------------------------------------------------------------- #
def hear(src, lang="en"):
    """Run both halves on one audio file. Returns a dict with everything."""
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    _require_ffmpeg()
    tmp = tempfile.mkdtemp(prefix="hear_")
    try:
        stt_mp3 = os.path.join(tmp, "stt.mp3")
        ana_wav = os.path.join(tmp, "ana.wav")
        ffmpeg_to(src, stt_mp3, ["-ar", "44100", "-ac", "1", "-b:a", "160k"])
        ffmpeg_to(src, ana_wav, ["-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le"])
        acoustic = analyze_acoustic(ana_wav)
        resp = transcribe(stt_mp3, lang)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    stt_err = resp.get("error")
    text, pace, gap, vp = parse_words(resp) if not stt_err else ("", None, None, {})
    return {
        "file": os.path.basename(src),
        "acoustic": acoustic,
        "text": text,
        "pace": pace,
        "gap": gap,
        "voice_profile": vp,
        "stt_error": stt_err,
        "raw_stt": resp,
    }


def format_card(r):
    """Render the unified 'WHAT I HEARD' card from a hear() result dict."""
    L = []
    L.append("🎧  WHAT I HEARD   " + r["file"])
    L.append("─" * 60)

    if r["stt_error"]:
        L.append("  WORDS:  [STT error] " + r["stt_error"])
    else:
        L.append("  WORDS:  " + (f'"{r["text"]}"' if r["text"] else "(no speech transcribed)"))

    vp = r["voice_profile"]
    if vp:
        bits = []
        for cat, lbl in [("vocalStyle", "style"), ("emotion", "emotion"),
                         ("pitch", "pitch"), ("age", "age"), ("accent", "accent")]:
            v = fmt_vp_line(vp, cat)
            if v:
                bits.append(f"{lbl}={v}")
        if bits:
            L.append("  VOICE:  " + " · ".join(bits))

    if r["pace"]:
        p = r["pace"]
        line = f"  PACE :  {p[0]} words / {p[1]}s ≈ {p[2]} wpm"
        if r["gap"]:
            line += f"   (longest pause {r['gap'][0]}s at {r['gap'][1]}s)"
        L.append(line)

    a = r["acoustic"]
    if "error" not in a:
        L.append(f"  SOUND:  {a['duration_s']}s · {a['brightness_hz']}Hz {a['brightness_label']}"
                 f" · key {a['key']} (conf {a['key_confidence']})"
                 + (f" · ~{a['tempo_bpm']} BPM" if a['tempo_bpm'] else ""))
        L.append(f"  DYN  :  {a['dynamics_label']} · range {a['dynamic_range_db']}dB"
                 f" (loud {a['loud_dbfs']} / quiet {a['quiet_dbfs']} dBFS) · crest {a['crest_db']}dB")
        if a["pauses"]:
            ps = ", ".join(f"{t0}–{t1}s ({d}s)" for t0, t1, d in a["pauses"][:4])
            L.append(f"  BREATH: {ps}")
    L.append("─" * 60)
    return "\n".join(L)
