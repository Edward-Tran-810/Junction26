"""Writes heldout_v1.jsonl: instructions written BEFORE the parser was run on them (2026-11-14, integration branch).
Never tune the parser on this file; if it is used for tuning, write heldout_v2 instead."""
import json
R, Y, B = "red", "yellow", "blue"
P = {1: {R: "left", Y: "center", B: "right"}, 2: {R: "left", Y: "right", B: "center"}, 3: {R: "center", Y: "left", B: "right"},
     4: {R: "center", Y: "right", B: "left"}, 5: {R: "right", Y: "left", B: "center"}, 6: {R: "right", Y: "center", B: "left"}}
rows = []
def add(text, gt, cat, adv=False):
    rows.append({"id": f"h{len(rows):03d}", "text": text, "gt": gt, "category": cat, "adversarial": adv})
# ---- permutations x paraphrase styles (6 perms x 6 styles = 36)
S = [
 "Red things end up in the {R} box, blue ones in the {B} box, and yellow ones in the {Y} box.",
 "Take everything blue to the {B} container; the yellow stuff goes {Y}; reds head to the {R} one.",
 "{Rc} bin: red. {Yc} bin: yellow. {Bc} bin: blue. Please sort accordingly.",
 "I'd like the yellow pieces dropped into the {Y} bin, the red pieces into the {R} bin, and finally the blue pieces into the {B} bin.",
 "Drop each blue item in the box on the {B}, each red item in the box on the {R}, and each yellow item in the {Y} one.",
 "Sort the objects by colour: {Y} for yellow, {B} for blue, {R} for red.",
]
for p, m in P.items():
    for i, tpl in enumerate(S):
        text = tpl.format(R=m[R], Y=m[Y], B=m[B], Rc=m[R].capitalize(), Yc=m[Y].capitalize(), Bc=m[B].capitalize())
        add(text, dict(m), f"perm{p}_style{i}")
# ---- irrelevant extra wording (8)
extra = [
 ("Hey robot, good morning! When you get a moment, could you please put the blue objects into the left bin, the red ones in the middle one, and yellow ones in the right bin? Thanks a lot, and be careful.", {B: "left", R: "center", Y: "right"}),
 ("The weather is nice today. Anyway: yellow objects belong in the right box, blue objects belong in the center box, red objects belong in the left box. Take your time.", {Y: "right", B: "center", R: "left"}),
 ("Before you start, check that your hands are free. Then move red to the center, blue to the right, and yellow to the left, one object at a time.", {R: "center", B: "right", Y: "left"}),
 ("This is a test of the sorting system for the demo. Blue goes in the right-hand box, red goes in the left-hand box and yellow goes in the middle box, as quickly as is safely possible.", {B: "right", R: "left", Y: "center"}),
 ("Please be gentle with the objects. The red ones go to the right bin and the blue ones go to the left bin. Yellow ones go to the center bin. Thank you!", {R: "right", B: "left", Y: "center"}),
 ("Quick job for you: red -> center bin, yellow -> left bin, blue -> right bin. Don't worry about the time.", {R: "center", Y: "left", B: "right"}),
 ("Folks at the booth are watching, so let's show them: put every yellow object in the right bin, every red object in the left bin, and every blue object in the center bin.", {Y: "right", R: "left", B: "center"}),
 ("Using whichever hand is closer, place the blue objects in the center box, the yellow objects in the left box and the red objects in the right box.", {B: "center", Y: "left", R: "right"}),
]
for t, m in extra: add(t, m, "distractor")
# ---- ambiguous / contradictory / incomplete: the correct behaviour is NOT to invent a bin for the unclear colour
amb = [
 ("Put the red objects in the left bin and the blue objects in the right bin.", {R: "left", B: "right"}, "missing_yellow"),
 ("Red goes left. Yellow goes center.", {R: "left", Y: "center"}, "missing_blue"),
 ("Put the red objects in the left bin. Actually no, put the red objects in the right bin. Yellow in the center, blue in the left.", {Y: "center", B: "left"}, "contradiction_red"),
 ("Blue in the left bin and blue in the right bin, red in the center bin, yellow in the right bin.", {R: "center", Y: "right"}, "contradiction_blue"),
 ("Sort everything into the bins.", {}, "no_mapping"),
 ("Put the red objects in the green bin, the blue objects in the right bin, the yellow objects in the left bin.", {B: "right", Y: "left"}, "unsupported_destination"),
 ("Put the purple objects in the left bin and the red ones in the center bin and the blue ones in the right bin and the yellow ones in the left bin.", {R: "center", B: "right", Y: "left"}, "unsupported_colour_ignored"),
 ("Red and blue go in the left bin, and yellow in the right bin.", {R: "left", B: "left", Y: "right"}, "two_to_one_ok"),
 ("Put the objects somewhere sensible.", {}, "vague"),
 ("Red to the left bin, yellow to the center bin, blue to the middle bin and also the right bin.", {R: "left", Y: "center"}, "contradiction_blue_two_bins"),
]
for t, m, c in amb: add(t, m, c, True)
open("heldout_v1.jsonl", "w").write("\n".join(json.dumps(r) for r in rows) + "\n")
print(len(rows), "instructions")
