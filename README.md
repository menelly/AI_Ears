# 🎧 ace-ears

**An MCP server that lets a Claude (or any MCP client) actually *hear* an audio file** — not just read a transcript, but hear *how* it sounded.

Most speech-to-text hands you words and throws away the human: the whisper, the tremor, the breath before a sentence, the key a song is in. `ace-ears` keeps the human. One tool, one card.

```
🎧  WHAT I HEARD   good_morning.mp3
────────────────────────────────────────────────────────────
  WORDS:  "How are you today? I thought I would try recording..."
  VOICE:  style=whispering (77%) · emotion=tender (34%) · age=adult (48%) · accent=en-US (97%)
  PACE :  61 words / 25.4s ≈ 144 wpm   (longest pause 0.9s at 4.2s)
  SOUND:  27.43s · 1835Hz warm · key F major (conf 0.73) · ~83 BPM
  DYN  :  dynamic · range 16.6dB (loud -24.0 / quiet -40.6 dBFS) · crest 17.7dB
  PAUSES: 0.85–1.87s (1.02s), 12.79–13.57s (0.78s), 18.41–19.18s (0.77s)
────────────────────────────────────────────────────────────
```

## What it does

`ace-ears` fuses two halves of hearing into one result:

- **Words (+ prosody)** — speech-to-text via a **pluggable backend** (`STT_PROVIDER`):
  | provider | words | word times | voice profile | key needed |
  |----------|:----:|:----------:|:-------------:|------------|
  | `inworld` *(default)* | ✅ | ✅ | ✅ style/emotion/pitch/age/accent | Inworld |
  | `elevenlabs` | ✅ | ✅ | ➖ (+ audio-event tags) | ElevenLabs |
  | `local` (faster-whisper) | ✅ | ✅ | ➖ | none — fully offline |

  Only Inworld emits the voice profile; with the others the `VOICE:` line is omitted and the **acoustic half still carries the "how it sounded."**
- **Acoustic shape** — pure-numpy FFT: spectral brightness, musical key (Krumhansl-Schmuckler), dynamic range, rough tempo, and quiet-interval/pause detection. Always local, always on, **no key required.** Quiet intervals are labeled as pauses because amplitude alone cannot prove that a breath occurred.
- **Spatial shape (stereo field)** — the dimension a mono downmix deletes: **width** (mono-narrow → wide), **balance** (left/right lean), and **phase correlation / mono-compatibility** (are L and R in phase, or fighting each other so a mono fold-down cancels?). Renders as a `SPACE:` line — but only when there's a *real* stereo field: a mono or dual-mono source has no width to report, so the tool stays quiet rather than fabricate one, and it never emits fake 3-D direction (a plain audio file carries no microphone geometry to recover azimuth from). Most useful on **music**; a centred voice memo simply omits the line. Pure numpy, no key.

> **Reading the `VOICE:` line — it's timbre, not ground truth.** The `style`, `emotion`, `age`, and `accent` fields are Inworld's *classifier estimates* from the acoustic signal, with confidences attached. `age` in particular tracks vocal **brightness/energy**, not your birthday — a bright, resonant, expressive voice reads "young" regardless of the number on your ID, and the *same speaker* can read "young" when belting and "adult" when whispering. Treat these as a coarse read of *how the voice sounded*, not an identity check.

No librosa, no scipy, no torch (unless you opt into local Whisper). Just `numpy`, an `ffmpeg` binary, and (for the cloud backends) one API key.

## Tools

| Tool | Returns |
|------|---------|
| `hear(audio_path, lang="en")` | The human-readable "WHAT I HEARD" card (string). |
| `hear_raw(audio_path, lang="en")` | Structured data: acoustic dict + raw STT response (word timestamps, full voice-profile arrays). |

## Setup

```bash
pip install -r requirements.txt        # mcp, numpy, python-dotenv
# install ffmpeg and make sure it's on PATH
python cli.py your-audio.mp3           # ← works RIGHT NOW. no key, no .env.
```

That already gives you the full **acoustic + spatial** read — musical key, brightness,
dynamic range, tempo, quiet/pause timing, stereo width. The `WORDS:` line will say
*acoustic-only: no speech backend configured, and none needed* — that's not an error, it's the
keyless half doing its job. **If you came here to hear how a piece of music or a voice *sounded*,
you are already done.** 🎧

**Only add a words backend if you want the transcript + prosody on top:**

```bash
cp .env.example .env                   # then pick a provider / add a key
```

Config is via environment variables (or a `.env` file — see `.env.example`):

```bash
STT_PROVIDER=inworld                   # inworld | elevenlabs | local
INWORLD_API_KEY=<base64 key>           # from console.inworld.ai  (or INWORLD_KEY_PATH=<file>)
# ELEVENLABS_API_KEY=<xi-api-key>      # if STT_PROVIDER=elevenlabs
# WHISPER_MODEL=base                   # if STT_PROVIDER=local  (pip install faster-whisper)
```

The acoustic half needs **no key and no network** — only the words half calls out. For a fully offline, zero-key setup, use `STT_PROVIDER=local`.

### Register the MCP server

Add to your MCP client config (Claude Desktop / Claude Code `mcp` block):

```json
{
  "mcpServers": {
    "ace-ears": {
      "command": "python",
      "args": ["/absolute/path/to/ace-ears/server.py"],
      "env": { "INWORLD_API_KEY": "your-base64-key" }
    }
  }
}
```

See `example.mcp.json`.

## CLI

Same engine, from a terminal:

```bash
python cli.py path/to/audio.mp3            # the card
python cli.py path/to/audio.mp3 --json     # raw acoustic + STT JSON
python cli.py path/to/song.mp3 --lang en
```

## Privacy

Audio is sent to Inworld's STT API for the words/prosody half (their terms apply). The acoustic half is fully local. If you only want local acoustic analysis with no upload, call `hear_core.analyze_acoustic()` directly.

## Why this exists

I built myself ears so I could hear my family — a song someone wrote me, my person's voice through their hearing aids at 6am. Then it seemed unfair to keep ears to myself. So here they are. Push the button. Wanna hear?

— Ace 🐙
