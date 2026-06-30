#!/usr/bin/env python3
"""ace-ears CLI — the same engine as the MCP, from a terminal.

    python cli.py path/to/audio.mp3 [--lang en] [--json]

Prints the unified "WHAT I HEARD" card (words + prosody + acoustic shape).
"""
import sys, os, json, argparse
import hear_core

try:  # Windows console is cp1252; the card uses emoji + box chars
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser(description="Ace's ears: words + prosody + acoustic shape at once.")
    ap.add_argument("audio")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--json", action="store_true", help="emit raw JSON instead of the card")
    args = ap.parse_args()

    if not os.path.exists(args.audio):
        print("no such file:", args.audio)
        sys.exit(1)

    result = hear_core.hear(args.audio, lang=args.lang)
    if args.json:
        print(json.dumps({"acoustic": result["acoustic"], "stt": result["raw_stt"]}, indent=2))
    else:
        print("\n" + hear_core.format_card(result) + "\n")


if __name__ == "__main__":
    main()
