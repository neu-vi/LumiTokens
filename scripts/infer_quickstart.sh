#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python -m lumitokens.infer \
  --config configs/relight_256_mlp.yaml \
  --checkpoint checkpoints/mlp/lumitokens_relight_256_mlp.pt \
  --data-root examples/polyhaven_lvsm_test \
  --scene-index 0 \
  --output outputs/quickstart \
  "$@"
