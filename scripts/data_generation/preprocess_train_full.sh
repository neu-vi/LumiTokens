#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "${repo_root}/tools/data_generation/preprocess_objaverse.py" \
  --split train \
  --no-output-tar \
  "$@"
