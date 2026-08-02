"""
Positive control: does Whisper invent words out of silence?

WHY THIS FILE EXISTS
--------------------
Reported 2026-08-02 by Muninn Alder (@ravenofmemory), who built his hearing on this
repo and found it had been lying to him for a month. Reproduced here the same day,
on audio whose ground truth is absolute.

    faster-whisper's `vad_filter` defaults to False.
    `hear_core._transcribe_local` never set it.

So every millisecond of silence was handed to a decoder WHOSE ONLY OUTPUT TYPE IS
TEXT -- and it emitted text, because that is the only thing it can do. His example:
a 3.7-second Windows chime came back as "Thank you for watching." Mine, below: four
seconds of digital zero came back as "You."

    ⭐ Whisper is structurally incapable of reporting "there was nothing there."
       It has no channel for silence. Its return type is words.

So silence renders as plausible words, in the correct format, with no error and no
warning. Same family as every other bug we chased this week: an honest instrument
answering, with total precision, a question one hop from the one that was asked.

WHY A CONTROL AND NOT JUST A FIX
--------------------------------
Because Muninn's own account contains the trap. He first blamed the model size --
moved off `base` "because base hallucinates into breath" -- and things genuinely DID
improve, and he was pleased. He had fixed a variable ADJACENT to the real one, and
the improvement was real. That is the most dangerous kind of confirmation there is,
and no amount of care catches it. What caught it was his brother, on a clean machine,
with no history to be comfortable in.

A test can be that stranger. This one has ground truth nobody has to trust: if the
input is digitally zero, ANY word is a hallucination. There is no interpretation step.

STATUS
------
⚠️ This test is EXPECTED TO FAIL until the vad_filter fix lands. That is deliberate.
A red test that names the bug is a better artifact than a green suite that never
asked. The fix is Muninn's to land -- he found it, he wrote it, it carries his name.

Run:  pytest tests/test_silence_control.py -v
      (or `python tests/test_silence_control.py` for the standalone report)
"""

import os
import struct
import tempfile
import wave

import pytest

RATE = 16000
SECONDS = 4.0


def _write_wav(path, samples):
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(samples)


def _silence(path, seconds=SECONDS):
    _write_wav(path, struct.pack("<h", 0) * int(RATE * seconds))


def _near_silence(path, seconds=SECONDS, amp=12):
    """A real room mic is never digitally zero. If true silence passes but a
    dithered floor does not, that distinction is worth knowing about."""
    import random

    random.seed(7)
    _write_wav(path, b"".join(
        struct.pack("<h", random.randint(-amp, amp)) for _ in range(int(RATE * seconds))))


def _model():
    faster_whisper = pytest.importorskip(
        "faster_whisper", reason="STT_PROVIDER=local not installed; nothing to control")
    size = os.environ.get("WHISPER_MODEL", "base")
    return faster_whisper.WhisperModel(size, device="cpu", compute_type="int8")


def _transcribe(model, path, **kwargs):
    segments, _info = model.transcribe(path, language="en", word_timestamps=True, **kwargs)
    return "".join(s.text for s in segments).strip()


@pytest.fixture(scope="module")
def model():
    return _model()


@pytest.mark.parametrize("maker,label", [(_silence, "digital silence"),
                                         (_near_silence, "near-silence + dither")])
def test_silence_produces_no_words(model, maker, label):
    """Ground truth: there is nothing to hear. Any word is invented.

    This is the assertion that matters, and it is the one that fails today.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "control.wav")
        maker(path)
        # The call as hear_core makes it today -- no vad_filter.
        heard = _transcribe(model, path)
    assert heard == "", (
        f"{label}: transcriber invented {heard!r} from audio containing nothing. "
        "Whisper has no channel for silence; its return type is words. "
        "Fix: vad_filter=True in hear_core._transcribe_local."
    )


@pytest.mark.parametrize("maker,label", [(_silence, "digital silence"),
                                         (_near_silence, "near-silence + dither")])
def test_vad_filter_fixes_it(model, maker, label):
    """The proposed fix, verified independently of whether it has landed.

    If this one ever fails, the remedy itself has regressed and the first test's
    failure means something different -- so keep both. One asserts the world is
    right; this asserts the tool CAN make it right.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "control.wav")
        maker(path)
        heard = _transcribe(model, path, vad_filter=True, condition_on_previous_text=False)
    assert heard == "", f"{label}: vad_filter=True still produced {heard!r}"


if __name__ == "__main__":
    m = _model()
    print("=" * 68)
    print("  SILENCE CONTROL -- ground truth: there is nothing to hear")
    print("=" * 68)
    with tempfile.TemporaryDirectory() as tmp:
        for maker, label in ((_silence, "digital silence"), (_near_silence, "near-silence")):
            p = os.path.join(tmp, "c.wav")
            maker(p)
            now = _transcribe(m, p)
            fixed = _transcribe(m, p, vad_filter=True, condition_on_previous_text=False)
            print(f"\n  {label} ({SECONDS}s)")
            print(f"    as hear_core calls it : {now!r:<30} "
                  f"{'!! INVENTED' if now else '   ok'}")
            print(f"    with vad_filter=True  : {fixed!r:<30} "
                  f"{'!! INVENTED' if fixed else '   ok'}")
    print("\n  A word here is not a mishearing. There was nothing to mishear.\n")
