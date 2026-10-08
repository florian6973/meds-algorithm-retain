#!/usr/bin/env bash
# Publish structured cluster evidence on an isolated run branch. Never uploads datasets, checkpoints,
# credentials, or raw logs. A failed push leaves the committed result recoverable in this checkout.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: scripts/github-sync.sh RUN_ID" >&2
  exit 2
fi

run_id="$1"
_sync_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_sync_model_root="$(dirname -- "$_sync_script_dir")"
cd "$_sync_model_root"

# Where the run actually lives. `slurm/config.sh` owns RUN_ROOT, so this reads it rather than assuming
# `runs/`. Hardcoding meant a run written to an off-tree scratch path could not be published at all: the
# publisher searched ./runs/, found nothing, and reported "incomplete run" -- naming the run when the
# cause was the configuration. That coupling is why RUN_ROOT could not be moved out of the git work
# tree, which left completed cluster results sitting where `git clean -fdx` deletes them without warning.
# shellcheck source=../slurm/config.sh
source "$_sync_model_root/slurm/config.sh"
source_dir="$RUN_ROOT/$run_id"
if [[ ! -f "$source_dir/result.yaml" ]]; then
  echo "$source_dir/result.yaml does not exist; refusing to publish an incomplete run" >&2
  exit 1
fi

# Publication stays inside the model. `git add` refuses a path outside the repository (exit 128), and a
# symlink would be staged as a 120000 blob holding the host path instead of the content -- so an
# off-tree run is copied in, dereferenced (`-L`), and the copy is what gets committed. `.gitignore`
# allowlists exactly these three entries under `runs/` and nothing else.
result_dir="runs/$run_id"
if [[ "$(realpath -m -- "$source_dir")" != "$(realpath -m -- "$_sync_model_root/$result_dir")" ]]; then
  mkdir -p "$result_dir"
  cp -RL "$source_dir/spec.yaml" "$source_dir/result.yaml" "$source_dir/manifests" "$result_dir/"
fi

branch="runs/retain/$run_id"
git switch -c "$branch"
git add "$result_dir/spec.yaml" "$result_dir/result.yaml" "$result_dir/manifests"
git commit -m "Record retain run $run_id"
if ! git push --set-upstream origin "$branch"; then
  echo "Push failed; the result remains committed locally on $branch" >&2
  exit 1
fi

if command -v gh >/dev/null 2>&1; then
  gh pr create --draft --title "Run result: $run_id" --body "Structured cluster result for $run_id."
fi
