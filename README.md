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
  BREATH: 0.85–1.87s (1.02s), 12.79–13.57s (0.78s), 18.41–19.18s (0.77s)
────────────────────────────────────────────────────────────
```

## What it does

`ace-ears` fuses two halves of hearing into one result:

- **Words + prosody** — via [Inworld STT](https://inworld.ai): the transcript, word timestamps, and a *voice profile* (vocal style, emotion, pitch, age, accent, each with confidence).
- **Acoustic shape** — pure-numpy FFT: spectral brightness, musical key (Krumhansl-Schmuckler), dynamic range, rough tempo, and breath/pause detection.

No librosa, no scipy, no torch. Just `numpy`, an `ffmpeg` binary, and an Inworld key.

## Tools

| Tool | Returns |
|------|---------|
| `hear(audio_path, lang="en")` | The human-readable "WHAT I HEARD" card (string). |
| `hear_raw(audio_path, lang="en")` | Structured data: acoustic dict + raw STT response (word timestamps, full voice-profile arrays). |

## Setup

```bash
pip install -r requirements.txt        # mcp, numpy
# install ffmpeg and make sure it's on PATH
export INWORLD_API_KEY=<your base64 key>   # from console.inworld.ai
```

The acoustic half needs no key — only the words/prosody half calls Inworld. The Inworld key is a base64 `Basic` auth token; set it via `INWORLD_API_KEY`, or put it in a file and point `INWORLD_KEY_PATH` at it.

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
