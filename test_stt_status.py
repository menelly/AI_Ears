#!/usr/bin/env python3
"""Regression tests for the WORDS-line status typing.

The whole point: "no speech backend configured" and "a configured backend
errored" are DIFFERENT states and must render differently. A keyless user came
for the acoustic half; showing them "[STT error]" makes a working tool look
broken. These tests pin that distinction so it can't silently rot back into one
scary label.

Run:  python test_stt_status.py    (no deps beyond the module; no audio, no keys)
"""
import os
import sys

try:  # Windows console is cp1252; keep the card's emoji/box chars printable.
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import hear_core


def _card_for(resp, **overrides):
    """Build the minimal hear()-shaped dict the card renderer needs, from a
    normalized STT response, and return the rendered card text."""
    r = {
        "file": "sample.wav",
        "provider": resp.get("provider", "inworld"),
        "acoustic": {"duration_s": 3.0, "brightness_hz": 1500, "brightness_label": "warm",
                     "key": "A minor", "key_confidence": 0.5, "tempo_bpm": None,
                     "dynamics_label": "steady", "dynamic_range_db": 12.0,
                     "loud_dbfs": -18.0, "quiet_dbfs": -30.0, "crest_db": 10.0,
                     "pauses": []},
        "stereo": {},
        "text": (resp.get("transcript") or ""),
        "pace": None, "gap": None, "voice_profile": {},
        "stt_error": resp.get("error"),
        "stt_unconfigured": resp.get("unconfigured"),
        "raw_stt": resp,
    }
    r.update(overrides)
    return hear_core.format_card(r)


def test_unconfigured_is_not_an_error():
    card = _card_for({"unconfigured": "No Inworld key found."})
    assert "[STT error]" not in card, "unconfigured must NOT render as an error"
    assert "acoustic-only" in card, "unconfigured should name the keyless acoustic path"
    # the sound must still be there — the whole point is the acoustic half survives
    assert "SOUND:" in card
    print("ok  unconfigured -> friendly acoustic-only line, no [STT error]")


def test_real_error_still_says_error():
    card = _card_for({"error": "unknown STT_PROVIDER 'bogus'"})
    assert "[STT error]" in card, "a real backend error MUST still read as an error"
    assert "acoustic-only" not in card
    print("ok  real error   -> still [STT error]")


def test_transcript_renders_when_present():
    card = _card_for({"provider": "local", "transcript": "hello there"})
    assert "[STT error]" not in card
    assert "acoustic-only" not in card
    assert "hello there" in card
    print("ok  transcript   -> words rendered, no status line")


def test_missing_keys_are_typed_unconfigured_not_error():
    # Guard the SOURCE of the distinction, not just the card: the three
    # not-set-up paths must return `unconfigured`, never `error`.
    env_backup = dict(os.environ)
    try:
        for k in ("INWORLD_API_KEY", "INWORLD_KEY_PATH", "ELEVENLABS_API_KEY"):
            os.environ.pop(k, None)

        r = hear_core._transcribe_inworld.__wrapped__ if hasattr(
            hear_core._transcribe_inworld, "__wrapped__") else None  # no wrapping; just call

        # inworld, no key -> unconfigured
        got = hear_core._transcribe_inworld("/nonexistent.mp3", "en")
        assert "unconfigured" in got and "error" not in got, got

        # elevenlabs, no key -> unconfigured (returns before touching the file)
        os.environ["STT_PROVIDER"] = "elevenlabs"
        got = hear_core._transcribe_elevenlabs("/nonexistent.mp3", "en")
        assert "unconfigured" in got and "error" not in got, got
    finally:
        os.environ.clear()
        os.environ.update(env_backup)
    print("ok  missing keys -> typed 'unconfigured' at the source, not 'error'")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)}/{len(tests)} passed 🎧")
