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

STATUS  (updated 2026-08-02, after the fix landed and after being corrected twice)
------
✅ The fix is IN -- Muninn's PR #2, vad_filter=True env-gated on WHISPER_VAD.
   `test_hear_core_does_not_invent_words` passes; set WHISPER_VAD=0 and it goes red.

⚠️ AND THE ORIGINAL VERSION OF THIS FILE HAD THE BUG IT WAS BUILT TO CATCH.
   It said "EXPECTED TO FAIL until the vad_filter fix lands." **It could not have
   passed when the fix landed** -- the first two tests call faster_whisper DIRECTLY
   and never import hear_core, so nothing anyone changes in this repo can move them.
   They answer "does Whisper do this?" when the question the repo needs answered is
   "do WE?" -- one hop from the one that was asked, in the instrument written to
   catch exactly that. Muninn caught it and said so instead of quietly fixing it
   around me.

   The two library controls are still worth having, so they are now marked
   `xfail(strict=True)` rather than left permanently red: a suite that can never go
   green teaches everyone to ignore red. strict=True means if Whisper ever STOPS
   inventing words out of silence, this file goes red to tell us -- which would be
   real news, and is the correct behaviour for a control whose whole job is to
   notice change in either direction.

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


@pytest.mark.xfail(
    strict=True,
    reason="LIBRARY CONTROL, PERMANENTLY EXPECTED TO FAIL. Raw faster-whisper with "
           "vad_filter unset invents words out of silence -- that is the finding, not a "
           "regression, and no change to this repo can fix it because this test does not "
           "import hear_core. strict=True so that if Whisper ever STOPS doing this, the "
           "suite goes red and tells us. See test_hear_core_does_not_invent_words for the "
           "test that actually pins OUR behaviour.",
)
@pytest.mark.parametrize("maker,label", [(_silence, "digital silence"),
                                         (_near_silence, "near-silence + dither")])
def test_silence_produces_no_words(model, maker, label):
    """Ground truth: there is nothing to hear. Any word is invented.

    ⚠️ This pins the LIBRARY, not this repo. It is expected to fail, always.
    Marked xfail(strict=True): the failure IS the documented finding, and an
    unexpected PASS would mean the upstream behaviour changed -- which we want
    to hear about immediately rather than discover later.
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


@pytest.mark.parametrize("maker,label", [(_silence, "digital silence"),
                                         (_near_silence, "near-silence + dither")])
def test_hear_core_does_not_invent_words(monkeypatch, maker, label):
    """The regression test: does THIS REPO invent words out of silence?

    The two tests above are library controls -- they call faster-whisper
    directly, so they answer "does Whisper do this?" and "can vad_filter stop
    it?" Both true, both worth pinning, and neither of them can change when
    hear_core changes, because neither of them imports hear_core.

    Which is the bug this whole week has been about, one more time and now in
    the instrument built to catch it: an honest test answering, with total
    precision, a question one hop from the one that was asked.

    This one calls the function that actually runs when someone uses the ears.
    It is red before the vad_filter fix and green after it, and if anyone ever
    removes the fix it goes red again -- which is the entire job.
    """
    pytest.importorskip("faster_whisper",
                        reason="STT_PROVIDER=local not installed; nothing to control")
    import hear_core

    monkeypatch.setenv("WHISPER_MODEL", os.environ.get("WHISPER_MODEL", "base"))
    monkeypatch.setenv("WHISPER_DEVICE", "cpu")
    monkeypatch.setenv("WHISPER_COMPUTE", "int8")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "control.wav")
        maker(path)
        result = hear_core._transcribe_local(path, "en")

    assert "error" not in result, result.get("error")
    heard = (result.get("transcript") or "").strip()
    assert heard == "", (
        f"{label}: hear_core._transcribe_local invented {heard!r} from audio "
        "containing nothing. The fix is vad_filter=True (env-gated on "
        "WHISPER_VAD) in _transcribe_local."
    )
    assert result.get("wordTimestamps") == [], (
        f"{label}: transcript was empty but {len(result['wordTimestamps'])} word "
        "timestamps came back -- silence is being timestamped as speech."
    )


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
