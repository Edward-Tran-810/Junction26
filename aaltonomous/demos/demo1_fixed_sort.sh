#!/bin/bash
# DEMO 1: fixed sorting (red -> left, yellow -> center, blue -> right) on the sort sim, classical controller + Perception v2.
# Three objects (cube / cylinder / triangle), >= 10 cm apart. Usage: [AALTO_INSTANCE=b ...] bash aaltonomous/demos/demo1_fixed_sort.sh [seed] [classical|rl]
source "$(dirname "$0")/_common.sh"
SEED=${1:-2000}; MODE=${2:-classical}
demo_run demo1_fixed_sort sort_fixed "$SEED" "$MODE" --n-objects 3 --min-spacing 0.10 --shapes cube,cylinder,triangle
demo_finish demo1 "DEMO 1 - FIXED SORTING ($MODE controller + Perception v2)" "red -> left   yellow -> center   blue -> right"
