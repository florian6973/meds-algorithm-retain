#!/usr/bin/env bash
# Download the frozen text encoder on the LOGIN node, before submitting anything.
#
# `pretrain` encodes every code description once with FremyCompany/BioLORD-2023 (~0.4 GB). Compute nodes
# on most clusters have no outbound network, and the failure mode is a job that burns its allocation
# waiting on a connection that will never open. Run this once per $HF_HOME.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
HF_HUB_OFFLINE=0 source slurm/env.sh

ENCODER="${1:-FremyCompany/BioLORD-2023}"
echo "Fetching $ENCODER into $HF_HOME ..."
HF_HUB_OFFLINE=0 $PORTER_RUN python -c "
import sys
from sentence_transformers import SentenceTransformer
m = SentenceTransformer(sys.argv[1])
print('ok:', sys.argv[1], '->', m.encode(['Measurement: Hemoglobin']).shape)
" "$ENCODER"
echo "Done. Jobs can now run with HF_HUB_OFFLINE=1."
