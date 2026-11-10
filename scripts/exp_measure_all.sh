#!/usr/bin/env bash
cd "$(dirname "$0")/.."
STEPS=last scripts/exp_curves.sh
scripts/exp_costs.sh
STEPS=rest TASKS=1 scripts/exp_curves.sh
echo ALLDONE
