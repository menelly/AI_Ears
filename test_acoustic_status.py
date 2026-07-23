#!/usr/bin/env python3
"""Small regression tests for conservative acoustic-card claims."""
import os
import tempfile
import wave

import numpy as np

import hear_core


def _write_wav(path, samples, sr=44100):
    pcm = np.clip(samples, -1, 1)
    pcm = (pcm * 32767).astype(np.int16)
    with wave.open(path, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sr)
        out.writeframes(pcm.tobytes())


def test_quiet_intervals_are_not_called_breaths():
    result = {
        "file": "sample.wav",
        "provider": "local",
        "acoustic": {
            "duration_s": 2.0, "brightness_hz": 1500, "brightness_label": "warm",
            "key": "unclear", "key_confidence": 0.1, "tempo_bpm": None,
            "dynamics_label": "dynamic", "dynamic_range_db": 15.0,
            "loud_dbfs": -15.0, "quiet_dbfs": -30.0, "crest_db": 10.0,
            "pauses": [(0.5, 0.9, 0.4)],
        },
        "stereo": {}, "text": "", "pace": None, "gap": None,
        "voice_profile": {}, "stt_error": None,
        "stt_unconfigured": "not configured", "raw_stt": {},
    }
    card = hear_core.format_card(result)
    assert "PAUSES:" in card
    assert "BREATH:" not in card


def test_clear_click_track_keeps_a_tempo():
    sr = 44100
    duration = 12
    x = np.zeros(sr * duration)
    # Short broadband clicks at 120 BPM.
    click = np.hanning(400)
    for start in np.arange(0, duration, 0.5):
        i = int(start * sr)
        x[i:i + len(click)] += click
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "clicks.wav")
        _write_wav(path, x, sr)
        acoustic = hear_core.analyze_acoustic(path)
    assert acoustic["tempo_bpm"] is not None
    # Autocorrelation may select a metrical multiple or subdivision; it should
    # still land on the click-track's pulse family.
    assert min(abs(acoustic["tempo_bpm"] - bpm) for bpm in (60, 120, 240)) < 3


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("ok ", test.__name__)
    print(f"\n{len(tests)}/{len(tests)} passed")
