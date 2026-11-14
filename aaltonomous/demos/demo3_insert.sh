#!/bin/bash
# DEMO 3 (stretch goal): ONE lidded box (left), cube -> square hole, cylinder -> circular hole, triangle -> triangular hole.
# Needs sim instance d (start_sim_insert.sh). Lid mesh: aaltonomous/scene/assets/lid_shapes.stl (see README.md for lid_shape_holes_200mm.stl).
#   bash aaltonomous/demos/demo3_insert.sh [seed] [classical|rl]
source "$(dirname "$0")/_common.sh"
source "$ROOT/aaltonomous/demos/instance_d.env"
SEED=${1:-2000}; MODE=${2:-classical}
export SERVER_ENV="AALTO_LID_ROT_DEG=180"
demo_run demo3_insert insert_shapes "$SEED" "$MODE" --n-objects 3 --min-spacing 0.10 --shapes cube,cylinder,triangle
demo_finish demo3 "DEMO 3 - SHAPE INSERTION ($MODE + Perception v2)" "cube->square hole  cylinder->circle hole  triangle->triangle hole  (lid turned 180 deg)"
