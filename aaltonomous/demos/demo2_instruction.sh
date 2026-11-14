#!/bin/bash
# DEMO 2: sort by natural-language instruction. The Language module parses the text; the intended mapping given with --mapping
# is used ONLY for scoring. Usage: bash aaltonomous/demos/demo2_instruction.sh ["instruction text" "red=right,yellow=left,blue=center"] [seed] [mode]
source "$(dirname "$0")/_common.sh"
INSTR=${1:-"Please send the blue pieces to the left box, drop the red ones in the middle box and put whatever is yellow into the right box."}
MAP=${2:-"blue=left,red=center,yellow=right"}
SEED=${3:-2001}; MODE=${4:-classical}
PARSED=$(python - "$INSTR" <<'PY' 2>/dev/null
import sys
from aaltonomous.baselines.classical.instruction import parse_instruction, expand
m = parse_instruction(sys.argv[1])
print("parsed: " + ", ".join(f"{k}->{v}" for k, v in {k[0]: v for k, v in expand(m).items() if v and k[1] == 'cube'}.items()))
PY
)
echo "[demo2] instruction: $INSTR"; echo "[demo2] intended   : $MAP"; echo "[demo2] $PARSED"
demo_run demo2_instruction sort_language "$SEED" "$MODE" --n-objects 3 --min-spacing 0.10 --shapes cube,cylinder,triangle --instruction "$INSTR" --mapping "$MAP"
demo_finish demo2 "DEMO 2 - SORT BY INSTRUCTION ($MODE + Perception v2)" "\"$INSTR\"" "$PARSED   (intended: $MAP)"
