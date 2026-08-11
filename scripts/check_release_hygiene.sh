#!/usr/bin/env bash

set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
cd "$repo_root"

failed=0

report_failure() {
    printf 'ERROR: %s\n' "$1" >&2
    failed=1
}

if tracked_ignored=$(git ls-files -ci --exclude-standard 2>/dev/null) && [ -n "$tracked_ignored" ]; then
    printf '%s\n' "$tracked_ignored" >&2
    report_failure "tracked files match .gitignore"
fi

credential_paths=$(find . -path ./.git -prune -o -type f \
    \( -name '.env' -o -name '.env.*' -o -iname 'api_keys.yaml' \
       -o -iname '*credentials*.json' -o -iname '*credentials*.yaml' \
       -o -name '*.pem' -o -name '*.key' -o -name '*.p12' -o -name '*.pfx' \) \
    ! -name '.env.example' ! -name 'api_keys_example.yaml' -print)
if [ -n "$credential_paths" ]; then
    printf '%s\n' "$credential_paths" >&2
    report_failure "credential-like files are present"
fi

large_files=$(find . -path ./.git -prune \
    -o -path ./checkpoints -prune \
    -o -path ./outputs -prune \
    -o -type f -size +10M -print)
if [ -n "$large_files" ]; then
    printf '%s\n' "$large_files" >&2
    report_failure "files larger than 10 MiB are present"
fi

if command -v rg >/dev/null 2>&1; then
    secret_matches=$(rg -n --hidden \
        --glob '!.git/**' \
        --glob '!checkpoints/**' \
        --glob '!scripts/check_release_hygiene.sh' \
        --glob '!docs/**' \
        --glob '!*.md' \
        '(WANDB_API_KEY|HF_TOKEN|HUGGING_FACE_HUB_TOKEN|API_KEY|SECRET_KEY|PASSWORD)[[:space:]]*[:=][[:space:]]*[^[:space:]#]{8,}' \
        . || true)
    if [ -n "$secret_matches" ]; then
        printf '%s\n' "$secret_matches" >&2
        report_failure "possible hard-coded secrets found"
    fi

    private_path_matches=$(rg -n --hidden \
        --glob '!.git/**' \
        --glob '!checkpoints/**' \
        --glob '!scripts/check_release_hygiene.sh' \
        --glob '!docs/**' \
        --glob '!*.md' \
        '/(projects|scratch|scratch2|music-shared-disk|group2)/' \
        . || true)
    if [ -n "$private_path_matches" ]; then
        printf '%s\n' "$private_path_matches" >&2
        report_failure "private infrastructure paths found"
    fi
else
    printf 'WARNING: rg is unavailable; content-pattern checks were skipped.\n' >&2
fi

if [ "$failed" -ne 0 ]; then
    exit 1
fi

printf 'Release hygiene checks passed.\n'
