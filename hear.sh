#!/usr/bin/env bash
# CLI launcher with NVIDIA GPU acceleration.
# Same LD_LIBRARY_PATH trick as run_server.sh — see that file for why.
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

exec "$VENV_PY" "$DIR/cli.py" "$@"
