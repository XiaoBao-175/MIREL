#!/usr/bin/env bash
set -euo pipefail

# Public MIREL evaluation entry point.
# The audited evaluator stays in the surrounding project checkout; this
# wrapper resolves the two released model names to the packaged model dirs.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RELEASE_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
EXPERIMENT_ROOT=${MIREL_EXPERIMENT_ROOT:-$(cd "$RELEASE_ROOT/.." && pwd)}
UPSTREAM="$EXPERIMENT_ROOT/scripts/eval/run_opd_full_eval.sh"

MODEL_NAME=
MODEL_FLAG_SEEN=0
TAG_FLAG_SEEN=0
ARGS=()

usage() {
  cat <<'EOF'
Usage:
  run_mirel_eval.sh --model-name MIREL-4B|MIREL-9B [evaluator options]
  run_mirel_eval.sh --model MODEL_DIR [evaluator options]

  The first form resolves the model under ../../models. The second form accepts
any complete HuggingFace model directory. All other options are forwarded to
the audited project evaluator, including --suite, --workers, --max-tokens,
--max-model-len, and --include-mmiu.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-name)
      [[ $# -ge 2 ]] || { echo "--model-name requires MIREL-4B or MIREL-9B" >&2; exit 2; }
      MODEL_NAME=$2
      shift 2
      ;;
    --model|--ckpt)
      MODEL_FLAG_SEEN=1
      ARGS+=("$1")
      [[ $# -ge 2 ]] || { echo "$1 requires a path" >&2; exit 2; }
      ARGS+=("$2")
      shift 2
      ;;
    --tag)
      TAG_FLAG_SEEN=1
      ARGS+=("$1")
      [[ $# -ge 2 ]] || { echo "--tag requires a name" >&2; exit 2; }
      ARGS+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -n "$MODEL_NAME" ]]; then
  [[ "$MODEL_FLAG_SEEN" -eq 0 ]] || {
    echo "--model-name cannot be combined with --model or --ckpt" >&2
    exit 2
  }
  case "$MODEL_NAME" in
    MIREL-4B|MIREL-9B) ;;
    *) echo "unsupported --model-name: $MODEL_NAME (use MIREL-4B or MIREL-9B)" >&2; exit 2 ;;
  esac
  MODEL_PATH="$RELEASE_ROOT/models/$MODEL_NAME"
  [[ -f "$MODEL_PATH/config.json" && -f "$MODEL_PATH/model.safetensors" ]] || {
    echo "released model is incomplete: $MODEL_PATH" >&2
    exit 1
  }
  ARGS+=(--model "$MODEL_PATH")
  [[ "$TAG_FLAG_SEEN" -eq 1 ]] || ARGS+=(--tag "$MODEL_NAME")
fi

[[ -x "$UPSTREAM" ]] || {
  echo "project evaluator not found or not executable: $UPSTREAM" >&2
  echo "Set MIREL_EXPERIMENT_ROOT to the experiment root if the checkout layout differs." >&2
  exit 1
}

exec bash "$UPSTREAM" "${ARGS[@]}"
