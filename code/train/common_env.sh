#!/usr/bin/env bash
set -euo pipefail

TRAIN_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FINAL_ROOT=$(cd "$TRAIN_SCRIPT_DIR/../.." && pwd)
PG_OPD_ROOT=$FINAL_ROOT/code/PG-OPD

# Set ENV_ROOT to the environment containing the released dependencies. If the
# script is called from an activated environment, VIRTUAL_ENV is used.
ENV_ROOT=${ENV_ROOT:-${VIRTUAL_ENV:-}}
if [[ -z "$ENV_ROOT" ]]; then
  echo "Set ENV_ROOT or activate the training environment before launching." >&2
  exit 2
fi
PYTHON=${PYTHON:-$ENV_ROOT/bin/python}

if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable not found: $PYTHON" >&2
  echo "Set ENV_ROOT or PYTHON before launching." >&2
  exit 2
fi

export PATH="$ENV_ROOT/bin:/usr/local/bin:/usr/bin:$PATH"
export PYTHONPATH="$PG_OPD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT:-120}
export WANDB_HTTP_TIMEOUT=${WANDB_HTTP_TIMEOUT:-120}

# Optional proxy settings for the original VPN environment. Leave empty on a
# machine with direct network access.
if [[ -n "${MIREL_PROXY:-}" ]]; then
  export http_proxy="$MIREL_PROXY"
  export https_proxy="$MIREL_PROXY"
  export HTTP_PROXY="$MIREL_PROXY"
  export HTTPS_PROXY="$MIREL_PROXY"
fi

cd "$PG_OPD_ROOT"
