#!/usr/bin/env bash
# Launcher for the ears MCP server with NVIDIA GPU acceleration.
#
# faster-whisper/ctranslate2 need libcublas + libcudnn at RUNTIME and do not
# pull them in themselves. On a box with no system CUDA runtime and no sudo,
# the pip wheels (nvidia-cublas-cu12, nvidia-cudnn-cu12) live inside the venv
# where ctranslate2 can't see them — so we find them and put them on the path.
#
# Without this you get a bare "Library libcublas.so.12 is not found" at the
# moment of first transcription, which reads like a broken install rather than
# a missing LD_LIBRARY_PATH.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$DIR/venv/bin/python"

NVIDIA_LIB_DIRS="$("$VENV_PY" -c "
import nvidia.cublas, nvidia.cudnn, os
print(os.path.join(list(nvidia.cublas.__path__)[0], 'lib'))
print(os.path.join(list(nvidia.cudnn.__path__)[0], 'lib'))
" 2>/dev/null || true)"

if [ -n "$NVIDIA_LIB_DIRS" ]; then
  export LD_LIBRARY_PATH="$(echo "$NVIDIA_LIB_DIRS" | tr '\n' ':')${LD_LIBRARY_PATH:-}"
fi

exec "$VENV_PY" "$DIR/server.py" "$@"
