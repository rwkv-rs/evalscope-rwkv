#!/usr/bin/env bash
set -euo pipefail

# Enable the per-benchmark Scoreboard callback, then run the requested evaluation.
export SCOREBOARD_PUBLICATION_METADATA="$1"
shift
exec evalscope eval "$@"
