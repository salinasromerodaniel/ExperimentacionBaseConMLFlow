#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" train.py --run-name baseline_C1_l2_none --C 1 --penalty l2 --class-weight none --seed 42
"${PYTHON_BIN}" train.py --run-name C001_l2_none --C 0.01 --penalty l2 --class-weight none --seed 42
"${PYTHON_BIN}" train.py --run-name C01_l2_none --C 0.1 --penalty l2 --class-weight none --seed 42
"${PYTHON_BIN}" train.py --run-name C10_l2_none --C 10 --penalty l2 --class-weight none --seed 42
"${PYTHON_BIN}" train.py --run-name C1_l1_none --C 1 --penalty l1 --class-weight none --seed 42
"${PYTHON_BIN}" train.py --run-name C1_l2_balanced --C 1 --penalty l2 --class-weight balanced --seed 42
