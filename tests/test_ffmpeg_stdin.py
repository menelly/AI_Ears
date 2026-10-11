"""
🎧🔌 Regression control for CHA-607: ffmpeg must never read the MCP server's stdin.

WHAT HAPPENED
-------------
2026-09-09: `hear` over MCP on a 2:36 song sent nothing for 1800 s and Claude Code
aborted it. The same file from the CLI returned a full card.
2026-10-10 (heartbeat arm), reproduced with STT switched OFF, so no network in it:
    over MCP stdio:   30 s clip 0.4 s   ·   2:36 song  HANG > 420 s
    direct call:      2:36 song 2.4 s
Same code, same file, only the transport differs. Under an stdio MCP server, stdin IS
the JSON-RPC pipe from the client, and ffmpeg listens on stdin for keypresses. A child
that inherits the pipe sits on the protocol stream.

THE FIX: `-nostdin` + `stdin=subprocess.DEVNULL` in hear_core.ffmpeg_to.

TWO TESTS, because they fail differently:
  1. CONTRACT: ffmpeg_to passes both guards (cheap, catches a refactor that drops one).
  2. BEHAVIOUR (opt-in): a real stdio MCP session calls hear() on a real file that hung,
     and has to answer within 60 s. Synthetic tones do NOT reproduce it; see that test.
"""
import os, shutil, subprocess, sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import hear_core  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")


def test_contract_ffmpeg_to_never_hands_stdin_to_ffmpeg(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"], seen["kw"] = cmd, kw
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(hear_core.subprocess, "run", fake_run)
    hear_core.ffmpeg_to("in.mp3", "out.wav", ["-ac", "1"])
    assert "-nostdin" in seen["cmd"], "ffmpeg will listen for keypresses on the MCP pipe"
    assert seen["kw"].get("stdin") is subprocess.DEVNULL, "child inherits the MCP stdin"


REPRO = os.environ.get("ACE_EARS_REPRO_AUDIO")


@needs_ffmpeg
@pytest.mark.skipif(not REPRO, reason="set ACE_EARS_REPRO_AUDIO to a file that hung (see docstring)")
def test_behaviour_reproducer_returns_over_real_mcp_stdio():
    """Real stdio MCP session, real reproducer file, STT off (no network).

    Controlled 2026-10-10 on claude_pop.mp3 (2:36, single mp3 stream; lives at
    receipts/claude-pop_2026-09-09/ on Ace's box, not in this public repo because it is
    someone else's song): OLD code HANG >120 s 3/3 · FIXED code ~2.3 s 3/3.

    ⚠️ OPEN QUESTION, NOT EXPLAINED AWAY: a synthetic 160 s sine mp3 did NOT hang the
    old code (both an idle-pipe test and an MCP test on the tone passed on broken code,
    so both were deleted rather than kept as decoration). Whatever about real music
    triggers it, I don't know yet. Hence a real file, opt-in.
    """
    pytest.importorskip("mcp")
    import asyncio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def go():
        env = dict(os.environ, STT_PROVIDER="none", PYTHONIOENCODING="utf-8")
        params = StdioServerParameters(command=sys.executable,
                                       args=[os.path.join(HERE, "server.py")], env=env)
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                res = await asyncio.wait_for(s.call_tool("hear", {"audio_path": REPRO}), 60)
                return res.content[0].text

    try:
        card = asyncio.run(go())
    except (asyncio.TimeoutError, TimeoutError):
        pytest.fail("hear() hung over MCP stdio on the reproducer: the CHA-607 shape is back")
    assert "WHAT I HEARD" in card
