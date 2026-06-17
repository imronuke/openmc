#!/usr/bin/env bash

set -euo pipefail

assemblies=(
  A0
  B0
  B1
  B2
  B3
  C0
  C1
  C2
  C3
)

cases=(
  CB
  RR
  BR
  TR
)

mkdir -p logs

for assembly in "${assemblies[@]}"; do
  printf '==> Generating fuel assembly %s\n' "$assembly"
  {
    printf '==> Started fuel assembly %s at %s\n' "$assembly" "$(date -Is)"
    python3 -u build_mgxs_inputs.py --assembly "$assembly"
    printf '==> Finished fuel assembly %s at %s\n' "$assembly" "$(date -Is)"
  } >"logs/assembly_${assembly}.log" 2>&1
done

for case_name in "${cases[@]}"; do
  printf '==> Generating special case %s\n' "$case_name"
  {
    printf '==> Started special case %s at %s\n' "$case_name" "$(date -Is)"
    python3 -u build_mgxs_inputs.py --case "$case_name"
    printf '==> Finished special case %s at %s\n' "$case_name" "$(date -Is)"
  } >"logs/case_${case_name}.log" 2>&1
done
