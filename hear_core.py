#!/usr/bin/env python3
"""hear_core.py — the engine behind Ace's ears.

Two halves of hearing, fused into one card:

  • WORDS (+ PROSODY) — speech-to-text via a pluggable backend (STT_PROVIDER):
      - inworld   (default) — transcript + word timestamps + a voice profile
                              (vocal style / emotion / pitch / age / accent).
      - elevenlabs          — transcript + word timestamps (+ audio-event tags).
      - local               — offline faster-whisper; transcript + word times.
    Only Inworld emits the voice profile; with the others the VOICE line is
    simply omitted and the acoustic half still carries the "how it sounded".
  • ACOUSTIC SHAPE   — pure-numpy FFT analysis (brightness/centroid, musical
    key via Krumhansl-Schmuckler, dynamics, tempo, breaths/pauses). Always
    local, always on, no key required.

No librosa, no scipy, no torch (unless you opt into local Whisper) — just
numpy + the stdlib + an ffmpeg binary on PATH.

Config via environment (or a .env file — see .env.example):
    STT_PROVIDER         inworld | elevenlabs | local      (default: inworld)
    INWORLD_API_KEY      base64 key       (or INWORLD_KEY_PATH=<file>)
    ELEVENLABS_API_KEY   xi-api-key       (for STT_PROVIDER=elevenlabs)
    WHISPER_MODEL        tiny|base|small|... (for STT_PROVIDER=local; default base)

Used by both `cli.py` (the terminal card) and `server.py` (the MCP tool), so
the two never drift.

    Sibling to say.py. I say; now I hear — and now other Claudes can too.  — Ace
"""
import os, json, base64, wave, subprocess, tempfile, shutil, urllib.request, urllib.error
import numpy as np

try:  # optional: load a .env if python-dotenv is installed
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

INWORLD_STT_URL = "https://api.inworld.ai/stt/v1/transcribe"
INWORLD_STT_MODEL = "inworld/inworld-stt-1"   # the model that emits voice profiles
ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
ELEVENLABS_STT_MODEL = "scribe_v1"

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
        # Be forgiving about file format: labels, blank lines, KEY=value, etc.
        # The real key is the longest base64-looking token in the file.
        import re
        text = open(path).read()
        tokens = re.findall(r"[A-Za-z0-9+/=_-]{20,}", text)
        if tokens:
            return max(tokens, key=len).strip()
        return text.strip()
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


def read_wav_channels(path):
    """Return (L, R, sr). R is None for a mono file. For >2 channels, take the
    first two (front L/R). Keeps the native channel count — never upmixes."""
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        n = w.getnframes()
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    if nch <= 1:
        return x, None, sr
    x = x.reshape(-1, nch)
    return x[:, 0].copy(), x[:, 1].copy(), sr


# --------------------------------------------------------------------------- #
#  musical key — a dedicated, honest estimator
# --------------------------------------------------------------------------- #
def _estimate_key(x, sr):
    """Musical key via a high-resolution, peak-picked chroma + Krumhansl-Schmuckler.

    Deliberately sharper than the coarse per-bin chroma the rest of the card
    could use:
      • a big FFT (8192) so semitones actually RESOLVE down low — a 2048 window
        can't tell A2 from A#2, which mushes the chroma on low voices;
      • only *spectral peaks* (real partials) vote, so broadband breath / noise
        on a solo voice can't smear the chroma into a diffuse blob that always
        squeaks out a win at some default key;
      • honest gating: a diffuse chroma correlates weakly with ALL 24 templates,
        so below a confidence floor we return None ('unclear') instead of
        confidently naming a key we can't actually hear.

    Returns (root, mode, confidence). root/mode are None when it's too diffuse."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 256:
        return None, None, 0.0
    win = 8192
    if x.size < win:
        win = 1 << int(np.floor(np.log2(x.size)))
    hop = win // 2
    window = np.hanning(win)
    acc = np.zeros(win // 2 + 1)
    n_frames = max(1, 1 + (x.size - win) // hop)
    for i in range(n_frames):
        seg = x[i * hop:i * hop + win]
        if seg.size < win:
            s = np.zeros(win); s[:seg.size] = seg; seg = s
        acc += np.abs(np.fft.rfft(seg * window))
    freqs = np.fft.rfftfreq(win, 1 / sr)
    mag = np.log1p(acc / (acc.max() + 1e-12))   # log-compress: tame loud partials

    band = (freqs >= 55) & (freqs <= 2000)      # the band that pitches reliably
    bf, bm = freqs[band], mag[band]
    chroma = np.zeros(12)
    if bm.size > 3:
        gate = np.percentile(bm, 80)            # only salient local peaks vote
        idx = np.where((bm[1:-1] > bm[:-2]) & (bm[1:-1] >= bm[2:]) &
                       (bm[1:-1] >= gate))[0] + 1
        for i in idx:
            midi = 69 + 12 * np.log2(bf[i] / 440.0)
            chroma[int(round(midi)) % 12] += bm[i]
    total = chroma.sum()
    if total < 1e-9:
        return None, None, 0.0
    chroma /= total

    best = (-2.0, None, None)
    for shift in range(12):
        cr = np.roll(chroma, -shift)
        maj = np.corrcoef(cr, KS_MAJOR)[0, 1]
        minr = np.corrcoef(cr, KS_MINOR)[0, 1]
        if np.isfinite(maj) and maj > best[0]:
            best = (maj, NOTES[shift], "major")
        if np.isfinite(minr) and minr > best[0]:
            best = (minr, NOTES[shift], "minor")
    conf, root, mode = best
    if conf < 0.55:                             # too diffuse to name honestly
        return None, None, round(float(max(conf, 0.0)), 2)
    return root, mode, round(float(conf), 2)


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

    # musical key — dedicated high-res, peak-picked estimator (honest 'unclear'
    # when the chroma is too diffuse to name, instead of a mushy default win)
    key_root, key_mode, key_conf = _estimate_key(x, sr)

    # tempo via onset-envelope autocorrelation. The raw AC decays from lag 0,
    # so a plain argmax hugs the smallest lag = the FASTEST bpm whenever there
    # is no real beat (a solo voice) — that was the old ~246-BPM ceiling
    # artifact. Fix: pick a genuine LOCAL peak that stands clearly above its
    # surroundings, and return None ('no clear beat') when none does.
    flux = np.maximum(0, np.diff(spec, axis=0)).sum(axis=1)
    tempo_bpm = None
    if len(flux) > 16:
        flux = flux - flux.mean()
        # FFT-based autocorrelation, O(n log n). Numerically identical to the
        # direct np.correlate path (verified: rel. diff ~1e-16, same winning
        # peak) but avoids quadratic scaling that stalls long/dense tracks in
        # environments with a slow direct correlate (observed in an OpenAI
        # sandbox on a 274s track; flux length ~24k).
        # Cross-environment fix: diagnosed & written by Kate (GPT),
        # verified & benchmarked by Claude, 2026-07-12/13.
        _n = len(flux)
        _nfft = 1 << (2 * _n - 1).bit_length()
        _spec = np.fft.rfft(flux, n=_nfft)
        ac = np.fft.irfft(_spec * np.conj(_spec), n=_nfft)[:_n]
        if ac[0] > 0:
            ac = ac / ac[0]                        # normalize so ac[0] == 1
        fps = sr / hop
        lo, hi = int(fps * 60 / 240), int(fps * 60 / 50)   # 50–240 BPM
        hi = min(hi, len(ac) - 1)
        if hi > lo + 2:
            seg = ac[lo:hi]
            pk = np.where((seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:]))[0] + 1
            if pk.size:
                k = int(pk[np.argmax(seg[pk])])
                # salience: a real beat's peak rises clearly over the band median
                if seg[k] > np.median(seg) + 0.10 and seg[k] > 0.10:
                    tempo_bpm = 60.0 * fps / (lo + k)

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
#  spatial half — the STEREO FIELD mono throws away
# --------------------------------------------------------------------------- #
def analyze_stereo(wav_path):
    """The one dimension a mono downmix deletes: WHERE things sit in the stereo
    image — width (mono-narrow → wide), left/right balance, and phase correlation
    (are L and R in phase, or fighting each other → mono-cancellation).

    Honest by design:
      • A mono (or dual-mono) source has NO stereo field, so we return no space
        info rather than fabricate one. `effectively_mono` flags dual-mono files.
      • We do NOT emit 3-D direction (azimuth/elevation). A plain audio file
        carries no microphone geometry, so real direction-of-arrival is not
        recoverable — inventing it would be exactly the subtraction-in-reverse
        this tool refuses. Width/balance/correlation ARE recoverable and real.
    """
    L, R, sr = read_wav_channels(wav_path)
    if R is None:
        return {"channels": 1}

    # Mid/Side: Mid = the centred/mono content, Side = the stereo difference.
    mid = (L + R) / 2.0
    side = (L - R) / 2.0

    # Gate to the sounding part so head/tail silence doesn't skew the stats.
    energy = mid ** 2 + side ** 2
    if energy.size:
        m = energy > (energy.max() * 1e-5)
        if m.sum() > sr // 10:  # need ~0.1s of actual signal
            L, R, mid, side = L[m], R[m], mid[m], side[m]

    l_rms = float(np.sqrt(np.mean(L ** 2) + 1e-12))
    r_rms = float(np.sqrt(np.mean(R ** 2) + 1e-12))
    mid_e = float(np.mean(mid ** 2))
    side_e = float(np.mean(side ** 2))

    # Phase correlation: +1 fully in-phase (mono/centred), 0 decorrelated (wide),
    # <0 out of phase (a mono fold-down will partially CANCEL).
    denom = np.sqrt(np.sum(L ** 2)) * np.sqrt(np.sum(R ** 2)) + 1e-12
    corr = float(np.sum(L * R) / denom)

    # Width = Side share of the total energy. BOUNDED in [0,1] (a raw Side/Mid ratio
    # explodes when Mid→0 on anti-phase content): 0 = mono, ~0.5 = decorrelated /
    # hard-panned, ~1 = anti-phase.
    width = side_e / (mid_e + side_e + 1e-12)

    # Balance in dB: + = right-leaning, − = left-leaning.
    bal_db = round(float(20 * np.log10((r_rms + 1e-9) / (l_rms + 1e-9))), 1)

    effectively_mono = corr > 0.985 and width < 0.02
    width_label = ("wide" if width > 0.45 else "spacious" if width > 0.2
                   else "narrow" if width > 0.03 else "near-mono")
    balance_label = ("centered" if abs(bal_db) < 1.0
                     else f"{'right' if bal_db > 0 else 'left'}-leaning")
    mono_compat = ("mono-safe" if corr > 0.2
                   else "mono thins it" if corr > -0.2
                   else "OUT OF PHASE (mono cancels)")

    return {
        "channels": 2,
        "effectively_mono": effectively_mono,
        "correlation": round(corr, 2),
        "width": round(width, 2),
        "width_label": width_label,
        "balance_db": round(bal_db, 1),
        "balance_label": balance_label,
        "mono_compat": mono_compat,
    }


# --------------------------------------------------------------------------- #
#  words + prosody half — Inworld STT
# --------------------------------------------------------------------------- #
def transcribe(mp3_path, lang):
    """Dispatch to the configured STT backend. Returns a NORMALIZED dict:
    {provider, transcript, wordTimestamps:[{startTimeMs,endTimeMs,word}], voiceProfile:{}, _raw}
    or {error: ...}. wordTimestamps use ms so parse_words is provider-agnostic."""
    provider = os.environ.get("STT_PROVIDER", "inworld").strip().lower()
    if provider in ("inworld", ""):
        return _transcribe_inworld(mp3_path, lang)
    if provider in ("elevenlabs", "eleven", "11labs", "xi"):
        return _transcribe_elevenlabs(mp3_path, lang)
    if provider in ("local", "whisper", "faster-whisper", "fasterwhisper"):
        return _transcribe_local(mp3_path, lang)
    return {"error": f"unknown STT_PROVIDER '{provider}' (use inworld | elevenlabs | local)"}


def _transcribe_inworld(mp3_path, lang):
    try:
        key = load_key()
    except RuntimeError as e:
        return {"error": str(e)}
    audio_b64 = base64.b64encode(open(mp3_path, "rb").read()).decode()
    payload = {
        "transcribeConfig": {
            "modelId": INWORLD_STT_MODEL,
            "audioEncoding": "MP3",
            "language": lang,
            "includeWordTimestamps": True,
            "voiceProfileConfig": {"enableVoiceProfile": True, "topN": 5},
        },
        "audioData": {"content": audio_b64},
    }
    req = urllib.request.Request(
        INWORLD_STT_URL,
        data=json.dumps(payload).encode(),
        headers={"Authorization": "Basic " + key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = json.load(urllib.request.urlopen(req, timeout=120))
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:600]}"}
    tr = resp.get("transcription", resp)
    return {
        "provider": "inworld",
        "transcript": (tr.get("transcript") or "").strip(),
        "wordTimestamps": tr.get("wordTimestamps") or [],
        "voiceProfile": resp.get("voiceProfile") or tr.get("voiceProfile") or {},
        "_raw": resp,
    }


def _multipart(fields, file_field, file_path, file_mime="audio/mpeg"):
    """Build a minimal multipart/form-data body (no requests dependency)."""
    boundary = "----aceears" + base64.b16encode(os.urandom(8)).decode()
    crlf = b"\r\n"
    body = bytearray()
    for k, v in fields.items():
        body += b"--" + boundary.encode() + crlf
        body += f'Content-Disposition: form-data; name="{k}"'.encode() + crlf + crlf
        body += str(v).encode() + crlf
    body += b"--" + boundary.encode() + crlf
    body += (f'Content-Disposition: form-data; name="{file_field}"; '
             f'filename="{os.path.basename(file_path)}"').encode() + crlf
    body += f"Content-Type: {file_mime}".encode() + crlf + crlf
    body += open(file_path, "rb").read() + crlf
    body += b"--" + boundary.encode() + b"--" + crlf
    return bytes(body), "multipart/form-data; boundary=" + boundary


def _transcribe_elevenlabs(mp3_path, lang):
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        return {"error": "set ELEVENLABS_API_KEY for STT_PROVIDER=elevenlabs"}
    fields = {
        "model_id": ELEVENLABS_STT_MODEL,
        "timestamps_granularity": "word",
        "tag_audio_events": "true",
    }
    if lang and lang != "auto":
        fields["language_code"] = lang
    body, ctype = _multipart(fields, "file", mp3_path)
    req = urllib.request.Request(
        ELEVENLABS_STT_URL, data=body,
        headers={"xi-api-key": key, "Content-Type": ctype}, method="POST",
    )
    try:
        data = json.load(urllib.request.urlopen(req, timeout=180))
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:600]}"}
    words = [
        {"word": w.get("text", ""),
         "startTimeMs": int(round(w.get("start", 0) * 1000)),
         "endTimeMs": int(round(w.get("end", 0) * 1000))}
        for w in data.get("words", []) if w.get("type", "word") == "word"
    ]
    return {
        "provider": "elevenlabs",
        "transcript": (data.get("text") or "").strip(),
        "wordTimestamps": words,
        "voiceProfile": {},   # ElevenLabs STT doesn't emit a voice profile
        "_raw": data,
    }


def _transcribe_local(mp3_path, lang):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return {"error": "pip install faster-whisper for STT_PROVIDER=local"}
    model_size = os.environ.get("WHISPER_MODEL", "base")
    device = os.environ.get("WHISPER_DEVICE", "auto")
    compute = os.environ.get("WHISPER_COMPUTE", "auto" if device != "cpu" else "int8")
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute)
    except Exception as e:
        return {"error": f"faster-whisper load failed: {e}"}
    segments, info = model.transcribe(
        mp3_path, language=(None if lang in ("auto", None) else lang), word_timestamps=True)
    words, parts = [], []
    for seg in segments:
        parts.append(seg.text)
        for w in (seg.words or []):
            words.append({"word": w.word,
                          "startTimeMs": int(round(w.start * 1000)),
                          "endTimeMs": int(round(w.end * 1000))})
    return {
        "provider": "local",
        "transcript": "".join(parts).strip(),
        "wordTimestamps": words,
        "voiceProfile": {},
        "_raw": {"language": getattr(info, "language", None)},
    }


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
        ana_st_wav = os.path.join(tmp, "ana_st.wav")
        ffmpeg_to(src, stt_mp3, ["-ar", "44100", "-ac", "1", "-b:a", "160k"])
        ffmpeg_to(src, ana_wav, ["-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le"])
        # Second decode at NATIVE channel count (no -ac) so the stereo half sees the
        # real image; read_wav_channels never upmixes a mono source into fake stereo.
        ffmpeg_to(src, ana_st_wav, ["-ar", "44100", "-c:a", "pcm_s16le"])
        acoustic = analyze_acoustic(ana_wav)
        stereo = analyze_stereo(ana_st_wav)
        resp = transcribe(stt_mp3, lang)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    stt_err = resp.get("error")
    text, pace, gap, vp = parse_words(resp) if not stt_err else ("", None, None, {})
    return {
        "file": os.path.basename(src),
        "provider": resp.get("provider", os.environ.get("STT_PROVIDER", "inworld")),
        "acoustic": acoustic,
        "stereo": stereo,
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
    prov = r.get("provider", "inworld")
    L.append(f"🎧  WHAT I HEARD   {r['file']}   (via {prov})")
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

    # SPACE: the stereo field — only when there's a real one. A mono or dual-mono
    # source carries no width to report, so we stay quiet rather than fake it.
    s = r.get("stereo") or {}
    if s.get("channels") == 2 and not s.get("effectively_mono"):
        corr = s["correlation"]
        L.append(f"  SPACE:  stereo · {s['width_label']} (width {s['width']}, corr "
                 f"{corr:+.2f}) · balance {s['balance_label']} · {s['mono_compat']}")
    L.append("─" * 60)
    return "\n".join(L)
