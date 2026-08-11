#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python -m lumitokens.infer \
  --config configs/relight_512_dpt.yaml \
  --checkpoint checkpoints/dpt/lumitokens_relight_512_dpt.pt \
  --data-root examples/polyhaven_lvsm_test \
  --scene-index 0 \
  --all-frames \
  --video \
  --num-input-views 10 \
  --view-chunk-size 1 \
  --output outputs/example_video \
  "$@"
