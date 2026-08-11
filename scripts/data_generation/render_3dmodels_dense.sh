#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tool_dir="${repo_root}/tools/data_generation"
render_python="${LUMITOKENS_RENDER_PYTHON:-python}"

cd "${tool_dir}"
exec "${render_python}" render_3dmodels_dense_enhance.py "$@"
