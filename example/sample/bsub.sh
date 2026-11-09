#!/usr/bin/env bash
set -euo pipefail

# Scheduler-aware launcher for the test server configuration.
# Override with: DCBF_CONFIG=dcbf.init_dataset.vasp.qiming.json bash bsub.sh
: "${DCBF_HOME:?Set DCBF_HOME to the installed DCBF-SUS2SH package root}"
CONFIG="${DCBF_CONFIG:-dcbf.init_dataset.vasp.test.json}"

source "$DCBF_HOME/activate.sh"
dcbf run "$CONFIG"
