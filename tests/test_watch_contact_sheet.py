"""
👀 Positive control for watch(): does the contact sheet actually show the WHOLE clip?

The failure this guards against is a quiet one: a grid that's all frame #1
(or all from the first second) looks exactly like a real contact sheet. So we
build a synthetic video whose ground truth we KNOW: a solid colour that changes
every quarter of the clip (red → green → blue → white), with a tone on the
soundtrack, and check that the grid's four quarters come out in those colours.

If the frames were not spread across the clip, the right-hand tiles would
still be red, and this test fails. A test that can't fail tells you nothing.

Run:  python tests/test_watch_contact_sheet.py
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import watch_core  # noqa: E402

COLORS = [("red", (255, 0, 0)), ("green", (0, 128, 0)), ("blue", (0, 0, 255)), ("white", (255, 255, 255))]


def make_clip(path, seg=2.0):
    # 🎨 four solid-colour segments of `seg` seconds each, concatenated, with a 440 Hz tone
    inputs, n = [], len(COLORS)
    for name, _ in COLORS:
        inputs += ["-f", "lavfi", "-i", f"color=c={name}:s=160x120:d={seg}:r=10"]
    inputs += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seg * n}"]
    filt = "".join(f"[{i}:v]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v]"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *inputs,
                    "-filter_complex", filt, "-map", "[v]", "-map", f"{n}:a",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", path],
                   check=True, stdin=subprocess.DEVNULL)


def tile_color(grid_path, col, cols=4, rows=1):
    # 🔍 ask ffmpeg for the mean colour of one tile (no PIL needed)
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", grid_path, "-vf",
         f"crop=iw/{cols}:ih/{rows}:{col}*iw/{cols}:0,scale=1:1", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"], capture_output=True, check=True, stdin=subprocess.DEVNULL).stdout
    return tuple(out[:3])


def close(a, b, tol=60):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def main():
    d = tempfile.mkdtemp(prefix="watch_test_")
    clip = os.path.join(d, "clip.mp4")
    make_clip(clip)
    r = watch_core.watch(clip, out_dir=d, cols=4, rows=1, listen=False)
    assert os.path.exists(r["grid"]), "no grid.jpg"
    assert r["mp3"] and os.path.getsize(r["mp3"]) > 0, "no audio extracted from a clip that HAS a tone"
    got = [tile_color(r["grid"], i) for i in range(4)]
    for (name, want), have in zip(COLORS, got):
        assert close(have, want), f"tile for {name} came out {have}, expected ~{want}: frames not spread across the clip?"
    print(f"✅ contact sheet spans the clip: {[n for n, _ in COLORS]} → {got}")
    print(f"✅ audio extracted: {os.path.getsize(r['mp3'])} bytes")

    # 🔎 look_at by SECONDS: the middle frame of the row for t=5.0 must be BLUE (segment 3 = 4–6 s).
    # (A grabber that ignored the timestamp and took frame 0 would come back red.)
    lk = watch_core.look_at(clip, "5.0", out_path=os.path.join(d, "look.jpg"))
    mid = tile_color(lk["image"], 1, cols=3, rows=1)
    assert close(mid, (0, 0, 255)), f"look_at(5.0) came back {mid}, expected blue: seeking is broken?"
    print(f"✅ look_at(5.0 s) lands in the blue segment: {mid}")

    # 🔤 find_phrase: case, punctuation, multi-word, and every occurrence
    words = [{"word": w, "startTimeMs": i * 1000} for i, w in
             enumerate(["The", "newly", "Magnified,", "CBS", "and", "the", "magnified", "CBS!"])]
    assert watch_core.find_phrase(words, "magnified") == [2.0, 6.0]
    assert watch_core.find_phrase(words, "magnified cbs") == [2.0, 6.0]
    assert watch_core.find_phrase(words, "newly magnified") == [1.0]
    assert watch_core.find_phrase(words, "maga-fied") == []
    print("✅ find_phrase: case/punctuation-blind, multi-word, finds every occurrence, and says [] when absent")


if __name__ == "__main__":
    main()
