"""Rule-based instruction parser: English sorting instruction -> (colour[, shape]) -> bin mapping.

    from aaltonomous.baselines.classical.instruction import parse_instruction, target_bin
    m = parse_instruction("Blue goes right, the rest in the middle; leave the red cubes alone.")
    # -> {'blue': 'right', 'yellow': 'center', ('red', 'cylinder'): 'center', ('red', 'triangle'): 'center'}
    target_bin(m, "red", "cylinder")  # -> 'center'
    target_bin(m, "red")              # -> None: depends on a shape we don't know -> leave it on the counter

Output keys follow scene.resolve_bin: a colour, or (colour, shape). The mapping is fully expanded over the 9
object types (3 colours x 3 shapes): a colour key when all its shapes go to one bin, else (colour, shape) keys.
An object type the instruction leaves unspecified, says to leave, or that the parser is unsure about is ABSENT
(-> left on the counter, 0 points) - an uncertain parse never guesses (a wrong bin costs -1).

Pipeline: normalise (contractions, hyphens, typos via edit distance against a small vocabulary) -> phrase markers
(rest / all / except / leave / remaining bin / left-to-right lists ...) -> sentences -> clause chunks (commas, "and",
"but", ...) -> per chunk: subject selectors (colour, shape, colour+shape, rest) and target bins (with negation)
-> chunk-merging state machine (shared targets, pronouns, "too", lists) -> rules with a specificity
((colour, shape) > colour | shape > "other <colour>" > everything/rest) -> per-type resolution; two rules of the
same specificity that disagree make that type unmapped.
No dependencies beyond the standard library.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

COLORS = ("red", "yellow", "blue")
SHAPES = ("cube", "cylinder", "triangle")
BINS = ("left", "center", "right")

# ---------------------------------------------------------------------------------------------------- vocabulary
COLOR_WORDS = {
    "red": "red", "reds": "red", "reddish": "red", "crimson": "red", "scarlet": "red", "ruby": "red",
    "cherry": "red", "vermilion": "red", "vermillion": "red", "maroon": "red", "burgundy": "red", "carmine": "red",
    "yellow": "yellow", "yellows": "yellow", "yellowish": "yellow", "gold": "yellow", "golden": "yellow",
    "lemon": "yellow", "amber": "yellow", "mustard": "yellow", "canary": "yellow", "saffron": "yellow",
    "blue": "blue", "blues": "blue", "bluish": "blue", "blueish": "blue", "navy": "blue", "azure": "blue",
    "cobalt": "blue", "sapphire": "blue", "indigo": "blue", "cerulean": "blue", "ultramarine": "blue",
}
# colours that exist in English but not on the counter: they still consume a clause (so it can't leak elsewhere)
OTHER_COLOR_WORDS = {"green", "purple", "orange", "black", "white", "pink", "brown", "grey", "gray", "violet",
                     "magenta", "silver", "beige", "cyan", "turquoise", "teal"}
SHAPE_WORDS = {
    "cube": "cube", "cubes": "cube", "cubic": "cube", "cuboid": "cube", "cuboids": "cube", "square": "cube",
    "squares": "cube", "squared": "cube",
    "cylinder": "cylinder", "cylinders": "cylinder", "cylindrical": "cylinder", "round": "cylinder",
    "rounded": "cylinder", "circular": "cylinder", "circle": "cylinder", "circles": "cylinder", "disc": "cylinder",
    "discs": "cylinder", "disk": "cylinder", "disks": "cylinder", "puck": "cylinder", "pucks": "cylinder",
    "tube": "cylinder", "tubes": "cylinder", "can": None, "cans": "cylinder", "drum": "cylinder", "drums": "cylinder",
    "triangle": "triangle", "triangles": "triangle", "triangular": "triangle", "prism": "triangle",
    "prisms": "triangle", "wedge": "triangle", "wedges": "triangle", "pyramid": "triangle", "pyramids": "triangle",
}
SHAPE_WORDS = {k: v for k, v in SHAPE_WORDS.items() if v}
SHAPE_ADJ = {"cubic", "square", "squared", "cylindrical", "round", "rounded", "circular", "triangular"}
POS_WORDS = {"left": "left", "leftmost": "left", "lefthand": "left",
             "right": "right", "rightmost": "right", "righthand": "right",
             "center": "center", "centre": "center", "middle": "center", "central": "center", "centermost": "center",
             "centremost": "center", "mid": "center", "midmost": "center"}
BIN_NOUNS = {"bin", "bins", "box", "boxes", "basket", "baskets", "container", "containers", "tray", "trays",
             "bucket", "buckets", "tub", "tubs", "crate", "crates", "receptacle", "receptacles", "one"}
OBJ_NOUNS = {"object", "objects", "item", "items", "thing", "things", "piece", "pieces", "ones", "block", "blocks",
             "toy", "toys", "shape", "shapes", "stuff", "bits", "parts", "part"}
PRONOUNS = {"it", "they", "them", "those", "these", "which", "each"}
COUNTER_WORDS = {"counter", "table", "mat", "surface", "countertop", "worktop", "desk", "bench"}
LEAVE_WORDS = {"leave", "leaving", "keep", "kept", "keeping", "stay", "stays", "remain", "untouched", "alone", "ignore", "ignored",
               "skip", "skipped", "disregard", "unsorted", "unmoved", "__leave__"}
NEG_WORDS = {"not", "never", "no", "nor", "neither", "__neg__"}
TOO_WORDS = {"too", "also", "__too__", "likewise", "similarly", "ditto"}
_BARE = (set(COLOR_WORDS) | set(SHAPE_WORDS) | OBJ_NOUNS | OTHER_COLOR_WORDS |
         {"the", "all", "both", "any", "every", "each", "of", "those", "these", "also", "colour", "coloured", "and", "or",
          "only", "just", "__rest__", "__all__", "put", "place", "move", "drop", "take", "sort", "grab", "pick", "get",
          "send", "toss", "throw", "chuck", "carry", "bring", "stick", "pop", "load", "please", "then", "now", "first",
          "other", "remaining", "leftover", "rest"})
def _is_bare(toks):
    """A pending noun phrase that may continue into the next clause: ends in a colour/shape/object word and names no
    bin, direction, pronoun, negation or counter ("I'd like the red", not "red in the lft one")."""
    bad = DIR_PREPS | (BIN_NOUNS - {"one"}) | PRONOUNS | NEG_WORDS | COUNTER_WORDS | LEAVE_WORDS
    return bool(toks) and not any(w in bad for w in toks) and (
        toks[-1] in COLOR_WORDS or toks[-1] in SHAPE_WORDS or toks[-1] in OBJ_NOUNS or toks[-1] in ("__rest__", "__all__"))


_NEG_GLUE = {"a", "an", "the", "any", "single", "one", "of"}
_REL_GLUE = {"that", "which", "who", "is", "are", "be", "s", "whose", "colour"}
NONBIN_NOUNS = {"wall", "walls", "corner", "door", "window", "arm", "hand", "hands", "gripper", "camera", "foot", "leg",
                "shoulder", "edge", "turn", "lane", "column", "row", "shelf", "screen", "monitor", "panel", "drawer",
                "pocket", "eye", "ear", "wing", "turns"}
DISTRACTOR_NOUNS = {"stop", "button", "tray", "cable", "wire", "light", "lamp", "lights", "tape", "line", "lines", "sign",
                    "marker", "floor", "door", "sticker", "label", "note", "pen", "cup", "mug", "shirt", "jacket", "cap",
                    "hat", "car", "team", "sky", "sea", "cloth", "towel", "wall", "handle", "switch", "screen", "flag",
                    "arrow", "dot", "dots", "mark", "folder", "binder", "notebook", "file", "bag", "chair", "emergency",
                    "alarm", "warning", "clip", "stripe", "tag", "card", "paper", "lid", "glove", "gloves", "helmet",
                    "vest", "badge", "lanyard", "ink", "paint", "wire", "hose", "valve", "lever", "pedal", "knob"}
DIR_PREPS = {"in", "into", "to", "inside", "onto", "toward", "towards", "at", "within"}
SPLIT_WORDS = {"and", "while", "whilst", "whereas", "but", "then", "plus", "however", "though", "although",
               "meanwhile", "afterwards", "next", "finally", "lastly", "after"}

# words the typo fixer may produce, and real words it must never "correct"
FUZZY_VOCAB = sorted(set(COLOR_WORDS) | set(POS_WORDS) | set(SHAPE_WORDS) |
                     {"bin", "bins", "objects", "object", "everything", "remaining", "except", "other", "others",
                      "rest", "leave", "items", "basket", "container", "anything", "respectively",
                      # confusable words that are NOT bins/colours: a typo of these must not become one
                      "counter", "counters", "table", "stay", "stays", "ones", "else", "except", "touch", "ignore",
                      "please", "place", "put", "move", "sort", "should", "goes", "belong", "remain", "pieces",
                      "things", "where", "leftover", "everything", "receive"})
NOT_TYPOS = {
    "led", "lead", "bled", "bred", "read", "reed", "rod", "rid", "rad", "bed", "fed", "wed", "ted", "redo", "reds",
    "glue", "clue", "blur", "blues", "true", "flue", "lift", "loft", "lest", "lent", "let", "lefts", "light",
    "might", "night", "fight", "sight", "tight", "eight", "bright", "fright", "wright", "rights", "righty", "write",
    "mellow", "fellow", "bellow", "yell", "bun", "ban", "been", "bit", "big", "pin", "tin", "win", "kin", "din",
    "fin", "gin", "sin", "bind", "counter", "counters", "contour", "cure", "cute", "tube", "cub", "cubs", "best", "test", "nest", "west", "vest",
    "zest", "jest", "pest", "rust", "reset", "center", "centre", "cent", "enter", "renter", "mildew", "fiddle",
    "riddle", "middy", "box", "fox", "lox", "bix", "items", "item", "stems", "these", "those", "there", "their",
    "then", "than", "that", "this", "the", "them", "they", "other", "others", "mother", "bother", "brother",
    "either", "leaves", "leaver", "lever", "least", "lift", "rift", "gift", "loft", "except", "accept", "expect",
    "object", "objects", "subject", "rested", "resting", "rests", "crest", "chest", "guest", "quest", "result",
    "cube", "cubes", "curb", "cubic", "tubes", "lube", "rube", "cylinder", "triangle", "angle", "tangle",
    "single", "bins", "bias", "binge", "ruby", "rub", "roy", "bed", "wedge", "edge", "ledge", "hedge", "sedge",
    "prim", "prison", "prise", "prize", "pride", "brim", "blue", "blew", "blow", "slue", "flew", "glum", "plum",
    "plue", "yelp", "yellow", "hello", "fallow", "sallow", "hallow", "follow", "hollow", "wallow", "rest",
    "remain", "remains", "reaming", "dreaming", "rending", "reading", "remaking", "anything", "nothing", "something",
    "everything", "every", "everyone", "anyone", "left", "right", "drop", "gold", "bold", "cold", "fold", "hold",
    "told", "sold", "mold", "golf", "golden", "olden", "holden", "goal", "navy", "nave", "wavy", "gravy", "davy",
    "bins", "pins", "sins", "wins", "fins", "midst", "mild", "mind", "mint", "mod", "squares", "square",
    "squire", "round", "around", "ground", "sound", "found", "bound", "mound", "pound", "wound", "hound",
    "rounds", "disc", "disk", "dish", "desk", "dust", "puck", "duck", "luck", "muck", "pick", "pack", "peck",
    "tub", "tab", "tube", "tune", "rule", "role", "pole", "hole", "whole", "wedges", "prism", "drum", "drug",
    "can", "cans", "circle", "circles", "uncle", "lefty", "box", "boxes", "foxes", "baskets", "red", "rad",
    "bread", "head", "dead", "lead", "mead", "redder", "reader", "ledger", "middle", "riddle", "griddle",
    "piddle", "diddle", "twiddle", "mid", "rid", "kid", "lid", "did", "hid", "bid", "aid",
}

COMMON_WORDS = set("""
a an the and or but if of in on at to by for from with into onto out up down off over under is are was were be
been am do does did done go goes went gone get got put puts set let lets may can will would should must could all
any each every some no not one two three this that these those there their them they it its he she we you i me my
our your his her who what which when where why how yes ok okay please thanks thank kindly just also too very so
then than now new old big way day use used see saw say said make made take took place places move moves sort
sorted lay lie sit end top bit lot few far near next last first second third bin bins box sky dark light pale
deep bright toy toys row now own per via led let lot low bed rod red rid sum fun run ran far few bad sad mad had
has her him ask yet nor odd see far act add ago aid aim air art bag bar bit buy car cut dry due ear eat egg eye
fit fix fly gap gas hit hot ice job key kid lab law lay leg lid lip log man map mix mud net nut oil pan pen pet
pie pin pit pop pot raw ray rid rob row rub rug sea set sew shy sin sip six ski son spy tag tan tap tar tax tea
ten tie tin tip toe ton top toy try tub two use van war wax web wet win yes yet zip
counter counters contour centred centered entering central centre center remain remaining remainder remained
cylinders triangles cubes rest rests lefty righty blues reds yellows greens
whatever whenever however wherever whichever whoever whether weather either neither other others another
rather instead ahead inside outside beside besides between behind below above there here these those
""".split())


CONFUSABLE = {"counter", "counters", "table", "stay", "stays", "touch", "ignore", "please", "place", "put", "move",
              "sort", "should", "goes", "belong", "remain", "where", "receive"}
_DIST_CACHE = {}


def _dl(a, b):
    """Damerau-Levenshtein (optimal string alignment) distance."""
    key = (a, b)
    if key in _DIST_CACHE:
        return _DIST_CACHE[key]
    la, lb = len(a), len(b)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            c = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    _DIST_CACHE[key] = d[la][lb]
    return d[la][lb]


def _fix_typo(w):
    """A word that is not a known word but is 1 edit (2 for long words) from exactly one vocabulary word."""
    if w in NOT_TYPOS or w in FUZZY_VOCAB or len(w) < 3 or not w.isalpha():
        return w
    if any(w in v for v in (COLOR_WORDS, POS_WORDS, SHAPE_WORDS, BIN_NOUNS, OBJ_NOUNS, COUNTER_WORDS, LEAVE_WORDS,
                            NEG_WORDS, TOO_WORDS, SPLIT_WORDS, PRONOUNS, OTHER_COLOR_WORDS, COMMON_WORDS)):
        return w
    maxd = 1 if len(w) < 8 else 2
    cands, bestd = [], 99
    for v in FUZZY_VOCAB:
        if abs(len(v) - len(w)) > maxd:
            continue
        if len(w) == 3 and len(v) > 4:
            continue  # 3-letter non-words ("lft", "blu", "rde") only map to short vocabulary words
        dd = _dl(w, v)
        if dd < bestd:
            cands, bestd = [v], dd
        elif dd == bestd:
            cands.append(v)
    # several equally close words are fine when they mean the same ("centr" -> center | centre, "redd" -> red | reds)
    if any(v in CONFUSABLE for v in cands):
        return w   # closest to a real non-target word ("countre", "cable", "says"): leave it alone
    if cands and bestd <= maxd and len({_canon(v) for v in cands}) == 1:
        return sorted(cands, key=lambda v: (v != _canon(v), v))[0]
    return w


def _canon(v):
    return COLOR_WORDS.get(v) or POS_WORDS.get(v) or SHAPE_WORDS.get(v) or v


# ---------------------------------------------------------------------------------------------------- phrases
_ORD = {"first": 0, "1st": 0, "one": 0, "second": 1, "2nd": 1, "two": 1, "third": 2, "3rd": 2, "three": 2,
        "last": 2, "final": 2}


def _ordinal_sub(m):
    i = _ORD[m.group(1)]
    side = m.group(4)
    if m.group(2) == "from":                       # "second from the right" counts from that side
        return " " + (BINS[i] if side == "left" else BINS[2 - i]) + " bin "
    if m.group(1) in ("first", "1st", "last", "final"):
        return " " + side + " bin "                # "the last bin on the right" = the right bin
    return " the bin "                             # "second bin on the left": unclear -> unreadable bin


def _with_sub(m):
    """'joins it / shares a bin with red / goes with the blue ones' = the same bin as ..."""
    w = (m.group(1) or "").strip()
    c = _canon(w)
    if c in COLORS:
        return f" __sameas_{c}__ "
    if c in BINS:
        return f" in the {c} "                    # "go with the left bin" = choose it
    if w in ("", "it", "them", "those", "that", "this", "these", "there"):
        return " __rel_same__ "
    return " the bin "


def _next_sub(m):
    """'next to the right bin' = center; 'next to blue' = the bin beside blue's; 'next to the middle' = unclear."""
    w = (m.group(1) or "").strip()
    c = _canon(w)
    if c in ("left", "right"):
        return " center bin "
    if c in COLORS:
        return f" __adj_{c}__ "
    if w in ("it", "them", "those", "that", "this", "these"):
        return " __rel_adj__ "
    return " the bin "


def _far_sub(m):
    """'furthest/closest TO the left' = left; 'furthest FROM the left' = right; 'closest/nearest FROM' = that side."""
    word, prep, side = m.group(1), m.group(2), m.group(3)
    away = word in ("furthest", "farthest") and prep == "from"
    return " " + ({"left": "right", "right": "left"}[side] if away else side) + " bin "


def _whose_sub(m):
    """'any cylinder whose colour is yellow' / 'objects that are red' -> 'yellow cylinder' / 'red objects'."""
    c = COLOR_WORDS.get(_fix_typo(m.group(2)))
    return f" {c} {m.group(1)} " if c else m.group(0)


def _swap_sub(m):
    a, b = _canon(_fix_typo(m.group(1))), _canon(_fix_typo(m.group(2)))
    if a in COLORS and b in COLORS and a != b or a in BINS and b in BINS and a != b:
        return f" , __swap_{a}_{b}__ , "
    return " __swapunk__ "


def _relative_sub(m):
    d = -1 if m.group(1) == "left" else 1
    i = BINS.index(POS_WORDS[m.group(2)]) + d
    return f" {BINS[i]} bin " if 0 <= i < 3 else " the bin "


PHRASES = [
    # colour names built from things: "sky-coloured", "lemon yellow", "fire engine red"
    (r"\b(?:sky|ocean|sea|denim|royal|baby|midnight|ice|navy|cornflower|steel)(?: colou?r(?:ed)?| blue)\b", " blue "),
    (r"\b(?:banana|sun|sunflower|butter|mustard|canary|lemon|gold|golden|corn|daffodil|egg yolk)(?: colou?r(?:ed)?| yellow)\b",
     " yellow "),
    (r"\b(?:tomato|blood|fire engine|fire truck|firetruck|strawberry|cherry|ruby|brick|rose|wine|post box|postbox)"
     r"(?: colou?r(?:ed)?| red)\b", " red "),
    # "any cylinder whose colour is yellow", "objects that are red"
    (r"\b(cubes?|cylinders?|triangles?|prisms?|wedges?|blocks?|pieces?|objects?|items?|ones|things?)(?: that| which)? "
     r"(?:is|are|whose colou?r is|whose colou?rs? are|coloured|painted|of colou?r) (\w+)\b", _whose_sub),
    # "those two colours" = a reference to the colours just named
    (r"\b(?:those|these|both)(?: two| 2)? colou?rs?\b|\bthe two colou?rs?\b", " those "),
    # "the two outer bins" = left and right, one each
    (r"\b(?:the )?(?:two )?(?:outer|outside|end|outermost) (?:bins|boxes|ones|containers)\b|\bboth ends\b", " __outer__ "),
    # colour groups a careful human knows
    (r"\b(?:the )?warm(?:er)? colou?r(?:s|ed)?(?: ones| objects| pieces| items)?\b", " red and yellow "),
    (r"\b(?:the )?(?:cool|cooler|cold) colou?r(?:s|ed)?(?: ones| objects| pieces| items)?\b", " blue "),
    (r"\b(?:the )?primary colou?rs\b", " __all__ "),
    # comparatives: "red goes further left than blue" (an order constraint, not a bin)
    (r"\b(?:further|farther|more|closer to the|nearer to the|more to the) (left|right)(?: than| of| side than)\b", r" __cmp_\1__ "),
    # colour idioms are not colour mentions
    (r"\b(?:feel|feels|feeling|felt)(?: a bit| a little| so| very| rather| kind of)? blue\b|\bout of the blue\b|"
     r"\b(?:once in a )?blue moon\b|\bred (?:tape|herring|flag|carpet|handed|eye)\b|"
     r"\byellow (?:bellied|belly|pages)\b|\bblue (?:collar|chip|sky thinking|tooth|print)\b|\bwhite lie\b", " "),
    # shapes described by what they look like
    (r"\b(?:shaped like|in the shape of|that looks? like|which looks? like|looking like|resembling|like)(?: a| an)?"
     r"(?: soda| soup| tin| drink| little| small| tiny| hockey| coke| beer)? (?:can|cans|tin|tins|tube|coin|pipe|log|drum|"
     r"barrel|puck|roll|battery|column|wheel|disc|disk|cylinder)s?\b", " cylinder "),
    (r"\b(?:shaped like|in the shape of|that looks? like|which looks? like|looking like|resembling|like)(?: a| an)?"
     r"(?: little| small| tiny| sugar)? (?:box|dice|die|block|brick|cube|square)s?\b", " cube "),
    (r"\b(?:shaped like|in the shape of|that looks? like|which looks? like|looking like|resembling|like)(?: a| an)?"
     r"(?: little| small| tiny)? (?:tent|wedge|roof|rooftop|pyramid|ramp|doorstop|slice of cheese|toblerone|triangle|"
     r"prism|sail|yield sign)s?\b", " triangle "),
    (r"\b(?:shaped like|in the shape of|that looks? like|which looks? like|looking like|resembling)\b", " __qualunk__ "),
    (r"\b(?:with|having) (?:three|3) (?:sides|corners|edges)\b|\bthree sided\b|\bthree cornered\b", " triangle "),
    (r"\b(?:with|having) (?:six|6) (?:square )?(?:faces|sides)\b", " cube "),
    # negation idioms that negate nothing here
    (r"\bno (?:time|worries|worry|problem|problems|rush|hurry|pressure|stress|idea|doubt|offence|offense|joke|kidding)\b"
     r"|\bnot (?:a problem|to worry|too fast|in a hurry|sure why|now)\b", " "),
    # "keep the reds away from the left bin" = not in
    (r"\b(?:away from|clear of|out of reach of)\b", " not in "),
    # "the bin furthest from the right-hand end" / "closest to the left side"
    (r"\b(?:the )?(?:bin |one |box )?(furthest|farthest|closest|nearest)(?: away)? (from|to) (?:the )?(left|right)(?: ?hand)?"
     r"(?: end| side| edge| wall)?\b", _far_sub),
    # "nothing but blue" = only blue
    (r"\bnothing (?:but|except|other than)\b", " only "),
    (r"\bnone of\b", " not "),
    # objects taken OUT of a bin: a source, not a destination
    (r"\b(?:out of|out from|back out of) (?:the |that |this |its |their )?(?:\w+ )?(?:bin|box|container|basket)\b", " __src__ "),
    # lists ordered by position
    (r"\b(?:going |counting |reading |ordered )?from (?:the )?left (?:to|towards|through) (?:the )?right\b", " __ltr__ "),
    (r"\b(?:going |counting |reading |ordered )?from (?:the )?right (?:to|towards|through) (?:the )?left\b", " __rtl__ "),
    (r"\bleft to right\b", " __ltr__ "),
    (r"\bright to left\b", " __rtl__ "),
    # ordinal bins counted from a side
    (r"\b(first|1st|second|2nd|third|3rd|last|final)(?: bin| box| one| container| basket)? "
     r"(?:(?:counting |starting |as seen |numbered )?(from|on|at)) (the )?(left|right)(?: side| end)?\b", _ordinal_sub),
    # position phrases -> one bin word
    (r"\b(?:on |at )?the (?:far |very )?(left|right)(?: ?hand)?(?: side| end)?(?= ?(?:bin|box|one|container|basket|tray|bucket))",
     r" the \1 "),
    (r"\b(?:on |to |at |along |towards? )?(?:the )?(?:far )?(?:left|right)(?: ?hand)?(?: side| end| edge| corner| part| wall| half)? of"
     r" (?:the |your |my |our |this )?(?:cell|room|counter|table|mat|robot|workspace|lab|area|desk|surface|bench|worktop|"
     r"countertop|wall|building|screen|floor|door|station)\b", " __dir__ "),   # a direction, not a bin
    (r"\b(left|right) ?hand side\b", r" \1 "),
    (r"\b(left|right) ?hand (?=bin|box|one|container|basket|tray|bucket)", r" \1 "),
    (r"\b(left|right) side\b", r" \1 "),
    (r"\b(?:far|furthest|farthest|very) (left|right)\b", r" \1 "),
    (r"\b(left|right) ?most\b", r" \1 "),
    (r"\b(?:in between|between the (?:other )?two(?: others| other ones| of them| bins| boxes)?|in the middle|"
     r"between them|between those(?: two)?|between these(?: two)?)\b", " center "),
    (r"\bdead centre\b|\bdead center\b", " center "),
    # relative bins: "the bin (just) to the left of the center one" / "... of that one"
    (r"\b(?:the )?(?:bin |one |box )?(?:just |immediately |directly |right |one )?(?:to the |on the )?(left|right) of "
     r"(?:the )?(left|right|center|centre|middle)(?: bin| one| box)?\b", _relative_sub),
    (r"\b(?:the )?(?:bin |one |box )?(?:just |immediately |directly |right |one )?(?:to the |on the )?(left|right) of "
     r"(?:that|this|it|them|those|these|where)(?: one| bin| box)?\b", r" __rel_\1__ "),
    (r"\b(?:to the |on the )?(?:left|right) of\b", " the bin "),        # relative to something we can't resolve
    (r"\b(?:next to|beside|adjacent to|neighbou?ring|alongside)(?: the)?(?: (\w+))?(?: bin| one| ones| objects| pieces)?\b",
     _next_sub),
    (r"\b(?:joins?|joining|accompan(?:y|ies)|shares? (?:a|the|its|their) (?:bin|box|one) with|goes? (?:along )?with|"
     r"keeps? company with|(?:is|are) grouped with|sits? with)(?: the)?(?: (\w+))?\b", _with_sub),
    # "normally X, but today Y": the later statement overrides
    (r"\bbut (?:today|tonight|this time|for now|right now|now|for this (?:run|round|episode|task|one)|on this occasion|"
     r"instead|this round|in this case)\b", " , __correct__ , "),
    # swaps relative to what was said (or to the usual mapping)
    (r"\b(?:swap|switch|exchange|trade|interchange|flip|reverse)(?: around| over| round)?(?: the)? (\w+)"
     r"(?: ones| objects| items| pieces| bins?| sides?)? (?:and|with|for) (?:the )?(\w+)(?: ones| objects| items| pieces| bins?| sides?)?\b",
     _swap_sub),
    (r"\b(\w+) and (\w+) (?:trade|swap|switch|exchange) (?:places|positions|bins|spots|sides)\b", _swap_sub),
    (r"\b(?:mirror|mirrored|mirroring|reverse|reversed|reversing|flip|flipped|flipping)(?: it| them| that| this| the order|"
     r" the arrangement| the mapping| the setup| everything| left to right| the sides)?\b", " , __swap_left_right__ , "),
    (r"\b(?:swap|switch|exchange|interchange|invert|rotate|shuffle|permute|cycle|change|mix)\w*(?: them| it| things|"
     r" everything| the bins| the colou?rs| the order)?(?: up| around| round)?\s*(?=:)", " "),   # explained after ':'
    (r"\b(?:swap|switch|exchange|interchange|invert|rotate|shuffle|permute|cycle)\w*\b", " __swapunk__ "),
    (r"\b(?:the )?(?:usual|standard|default|normal|regular|typical|original|ordinary|familiar|rung 1|fixed)"
     r" (?:setup|set up|sorting|mapping|sort|arrangement|way|rule|rules|order|configuration|plan|assignment|scheme|layout)\b",
     " __usual__ "),
    # "the bin at the opposite end": relative to the bin named before
    (r"\b(?:the )?(?:bin |one |box )?(?:at |on |in )?the (?:opposite|other|far) (?:end|side)(?: bin| one)?\b", " __rel_opp__ "),
    # "nothing red" = not red; "nothing else" = the rest, negated
    (r"\bnothing else\b", " __rest__ not "),
    (r"\bnothing\b", " not "),
    # self-corrections: what follows overrides what came before
    (r"\b(?:no,? wait|wait,? no|no no|sorry,? i mean|i meant to say|i meant|i mean|scratch that|make that|"
     r"correction|or rather|let me correct that|oops|on second thought|actually,? no|actually)\b", " , __correct__ , "),
    # "yellow goes where blue would normally go": the default stated before, not the final mapping
    (r"\b(?:wherever|where) (?:the )?(\w+)(?: ones| objects| items| pieces| things| blocks)? (?:would |should |will )?"
     r"(?:normally|usually|typically|ordinarily|otherwise) (?:go|goes|be|end up|land)\b", r" __sameasdflt_\1__ "),
    # "the center bin is where blue goes" defines the bin
    (r"\b(is|are) where (?:the )?(\w+)(?: ones| objects| items| pieces| things)? (?:goes|go|belongs?|should go|lives?|end up|ends up)\b",
     r" \1 for \2 "),
    # "wherever the blue ones go" = the same bin as blue
    (r"\b(?:wherever|where) (?:the )?(\w+)(?: ones| objects| items| pieces| things| blocks)? (?:go|goes|are going|is going|"
     r"end up|ends up|went|are put|get put|are headed|are being put)\b", r" __sameas_\1__ "),
    # "whichever bin is left over" = the remaining bin (before "what is left" = the rest)
    (r"\b(?:whichever|whatever|the|any|that) (?:bin|box|one|container|basket) (?:is |that is |which is |that remains |"
     r"remains |is still |that is still )?(?:left over|left|remaining|free|unused|empty|spare|available|still free)\b",
     " __rembin__ "),
    # priority statements: "the shape rule wins over the colour rule", "cylinders take priority"
    (r"\b(?:take|takes|taking|have|has|get|gets|got|given) (?:top |first |the |absolute )?(?:priority|precedence)\b|"
     r"\b(?:wins?|comes? first|overrides?|trumps?|outranks?|beats?|is more important than|matters? more)\b", " __prio__ "),
    # what is left / the rest
    (r"\b(?:what|whatever|anything|everything|all that|whichever ones?|any ones?|those that|the ones that|ones that)"
     r"(?: still)?(?:'s| is| are| remains?| stays?)? left(?: over)?\b", " __rest__ "),
    (r"\b(?:what|whatever|anything that|everything that|all that) remains?\b", " __rest__ "),
    (r"\bleft ?overs?\b(?! bin| box| one\b)", " __rest__ "),
    (r"\b(?:everything|anything|all|whatever|any ?thing|every ?thing) else\b", " __rest__ "),
    (r"\b(?:all |any |each )?(?:the )?others\b", " __rest__ "),
    (r"\bthe remainder\b", " __rest__ "),
    (r"\b(?:the )?rest(?: of (?:the |them|it|those)(?:objects?|items?|things?|pieces?|ones)?)?\b", " __rest__ "),
    (r"\b(?:the )?(?:two|both) other (?:colou?rs?|ones|objects|kinds)\b", " __rest__ "),
    (r"\b(?:all |any |every |each )?(?:the )?(?:other|remaining|leftover|unmentioned|outstanding)(?: two| 2| three)?"
     r" (?:objects?|items?|things?|pieces?|ones|colou?rs?|blocks?|toys?|shapes?|stuff)\b", " __rest__ "),
    # remaining bin ("the other bin", "the last one")
    (r"\b(?:the )?(?:remaining|other|leftover|free|empty|unused|spare|last remaining|remaining empty|one remaining|"
     r"unclaimed|available|open)(?: bin| box| one\b| container| basket)\b", " __rembin__ "),
    (r"\b(?:the )?(?:bin|box|one|container|basket) (?:that is |that's )?(?:left over|remaining|unused|still free|still empty|nobody uses|not yet used)\b",
     " __rembin__ "),
    # regardless of colour / shape
    (r"\b(?:regardless of|irrespective of|no matter(?: what| which)?|whatever|independent of|independently of)(?: the| their| its)?"
     r" (colou?r|shape)s?(?: (?:it|they) (?:is|are|has|have))?\b", r" __regardless_\1__ "),
    (r"\b(?:of )?(?:any|every|all) (colou?r|shape)s?\b", r" __regardless_\1__ "),
    # all
    (r"\b(?:everything|anything|every ?thing|any ?thing)\b", " __all__ "),
    (r"\b(?:all|every|each|any)(?: single| one of| of)?(?: the)? (?:objects?|items?|things?|pieces?|blocks?|toys?|shapes?)\b",
     " __all__ "),
    (r"\ball of them\b|\ball of these\b|\ball of those\b", " __all__ "),
    # exceptions
    (r"\b(?:with the exception of|except for|except|excepting|excluding|other than|apart from|aside from|save for|"
     r"barring|not counting|not including|but not|minus|unless)\b", " __except__ "),
    (r"\ball (?=but\b)", " __all__ "),                       # bare "all but X" = everything except X
    (r"(__all__|\ball\b)( \w+)?( \w+)? but\b", r"\1\2\3 __except__ "),
    # same bin as <colour>
    (r"\b(?:in |into |to )?(?:the )?same (?:bin|box|one|place|container|basket|spot) as(?: the)? (\w+)\b",
     r" __sameas_\1__ "),
    (r"\b(?:together|along) with(?: the)? (\w+)\b", r" __sameas_\1__ "),
    # leave on the counter
    (r"\bdo not (?:touch|move|pick up|pick|sort|handle|grab|bother with|bother|lift|take|place|put|transfer|relocate|"
     r"shift|disturb|mess with|drop|throw|toss|send)\b", " __dontmove__ "),   # leave; a bin after it -> negation
    (r"\b(?:in place|where (?:they|it) (?:are|is|were|was|lie|lies|sit|sits|stand|stands)|as (?:they|it) (?:are|is)|"
     r"as is|on its own|on their own|where they lie)\b", " __leave__ "),
    (r"\b(?:is|are|be|been|being|get|gets|got|stay|stays|remain|remains|kept|keep|keeping) left\b"
     r"(?= (?:on|alone|untouched|where|in place|as|behind|there|out|be|unsorted|lying|sitting|__leave__)|\s*$|\s*[,.;!?])",
     " __leave__ "),
    (r"\b(?:should|must|can|to|will|may|could|would) be left\b(?! (?:bin|box|one|container))", " __leave__ "),
    (r"\bleft (?:alone|untouched|behind|out|unsorted|be)\b", " __leave__ "),
    # also / too
    (r"\b(?:as well|the same goes for|same goes for|the same for|same for|same with|the same with|as does|as do|"
     r"so does|so do|same thing for|same thing with|ditto for|in the same way)\b", " __too__ "),
    # negated targets
    (r"\b(?:instead of|rather than|in place of)\b", " __neg__ "),
    # "... in the right bin with everything else": a companion, not a new subject
    (r"\b(?:together |along )?with (?:__rest__|__all__ else|__all__|the others)\b", " "),
    # "the cubes and triangles go by colour: red ..."
    (r"\b(?:go|goes|are sorted|is sorted|get sorted|gets sorted|sorted|split|divided|grouped|sort them|sort those|"
     r"sort these|are split|are divided) by colou?r\b", " __otherscope__ "),
    # ordered lists
    (r"\b(?:respectively|in that order|in the same order|correspondingly|in order|in this order)\b", " __resp__ "),
    # "everything else (sorted) by colour: red ...", "otherwise, red ...": the colour rules that follow cover the rest
    (r"\b(?:sort |sorted |split |divide |divided |group |grouped |organi[sz]e |organi[sz]ed |do |handle |and )?"
     r"(?:__rest__|__all__)(?: (?:is|are|should be|gets?|get|goes|go|can be|will be))?(?: sorted| split| divided| grouped|"
     r" organi[sz]ed| handled| goes| go)? by colou?r\b", " __otherscope__ "),
    (r"\botherwise\b|\b(?:as )?for (?:__rest__|__all__ else)\s*(?=[,:])|\b(?:among|of) __rest__\s*(?=[,:])", " __otherscope__ "),
    # one bin per colour
    (r"\b(?:each|every) colou?r (?:gets|has|goes in|goes into|needs|takes|uses) (?:its|their|a) (?:own|separate|different)"
     r" (?:bin|box)\b|\b(?:one|a) (?:bin|box) (?:per|for each) colou?r\b|\b(?:all |the )?(?:three )?colou?rs (?:go|are) "
     r"(?:in|into|to) (?:different|separate) bins\b|\bno two colou?rs (?:share|go in) (?:a|the same) bin\b", " __unique__ "),
    (r"\b(?:normally|usually|typically|ordinarily|by default|as a rule|on a normal day|most days)\b", " __normally__ "),
]
PHRASES = [(re.compile(p), r) for p, r in PHRASES]


_BARE_ORD = re.compile(r"\b(?:the )?(first|1st|second|2nd|third|3rd)(?: bin| box| one| container| basket)\b")
_DIR_ORD = re.compile(r"\b(?:first|1st|second|2nd|third|3rd|last|final)(?: bin| box| one| container| basket)? "
                      r"(?:counting |starting )?from (?:the )?(left|right)\b")


def normalise(text):
    t = text.lower()
    # sentence-initial interjection "Right, let's start" / "OK right, ..." is not a bin
    t = re.sub(r"(^|[.!?;:]\s*)(?:ok(?:ay)?,? |alright,? )?right\s*[,!]\s*(?=(?:let|so|now|then|first|here|ok|okay|we|i|"
               r"listen|team|robot|time|good|this|please|today|the task|here)\b)", r"\1okay , ", t)
    t = t.replace("’", "'").replace("‘", "'").replace("`", "'")
    t = re.sub(r"[–—]|\s-+\s|--+", " , ", t)
    t = re.sub(r"can't|cannot", "can not", t)
    t = re.sub(r"won't", "will not", t)
    t = re.sub(r"n't\b", " not", t)
    t = re.sub(r"\b(what|whatever|that|it|there|everything|anything|who)'s\b", r"\1 is", t)
    t = re.sub(r"\blet's\b", "let us", t)
    t = re.sub(r"'(re|ll|ve|d|m)\b", " ", t)
    t = re.sub(r"'s\b", "", t)
    t = re.sub(r"\b(?:ignore|disregard|forget(?: about)?|scrap|cancel|discard|never mind|throw out|delete)\b"
               r"(?=[^.;!?]*\b(?:old|previous|earlier|outdated|obsolete|original|prior|last|yesterday)\b)"
               r"[^.;!?'\"]*(?:['\"][^'\"]*['\"][^.;!?]*)?", " ", t)     # "ignore the old note that says '...'"
    t = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", t)
    t = re.sub(r"[()\[\]]", " , ", t)
    t = re.sub(r"[\"']", " ", t)
    t = re.sub(r"[/&+]", " and ", t)
    t = re.sub(r"\bcolou?red\b|\bcolou?rs?\b(?! (?:it|they) )", lambda m: " colour " if "colour" in m.group(0) or "color" in m.group(0) else m.group(0), t)
    # typo repair word by word
    t = re.sub(r"[a-z]+", lambda m: _fix_typo(m.group(0)), t)
    t = re.sub(r"\s+", " ", t)
    dm = _DIR_ORD.search(t)
    for rx, rep in PHRASES:
        t = re.sub(r"\s+", " ", rx.sub(rep, t))
    if dm:  # numbering direction was set by an earlier "... from the left": bare "the second one" follows it
        side = dm.group(1)
        t = _BARE_ORD.sub(lambda m: " " + (BINS[_ORD[m.group(1)]] if side == "left" else BINS[2 - _ORD[m.group(1)]])
                          + " bin ", t)
        t = re.sub(r"\s+", " ", t)
    return t.strip()


def tokenize(t):
    return re.findall(r"__[a-z_]+__|[a-z0-9]+|[.,;:!?\n]", t)


# ---------------------------------------------------------------------------------------------------- mentions
@dataclass
class Sel:
    color: str | None = None
    shape: str | None = None
    rest: bool = False          # everything / everything else
    other: bool = False         # "other red things": lower priority than other rules on the same colour
    unknown: bool = False       # a colour that doesn't exist here (green ...)
    pos: int = 0
    else_: bool = False         # "everything else / the rest": excludes every type mentioned anywhere else
    boost: float = 0.0          # "... except that cylinders go center": an exception clause outranks what it modifies
    ref: str | None = None      # "former" / "latter": resolved against the mention history

    def spec(self):
        if self.rest:
            return 0.0
        s = (1.0 if self.color else 0.0) + (1.0 if self.shape else 0.0)
        s = 2.0 if s == 1.0 else 3.0
        return s - (0.5 if self.other else 0.0)

    def matches(self, c, s):
        if self.unknown:
            return False
        return (self.color in (None, c)) and (self.shape in (None, s))


@dataclass
class Tgt:
    bin: str | None             # left / center / right, "rem" (remaining bin), "same:<colour>", None = unknown
    neg: bool = False
    pos: int = 0
    hint: str | None = None


@dataclass
class Chunk:
    toks: list
    sels: list = field(default_factory=list)       # positive subject selectors
    excl: list = field(default_factory=list)       # excluded selectors ("except blue", "not the red ones")
    tgts: list = field(default_factory=list)       # targets (neg flag)
    leave: bool = False
    counter: bool = False
    pron: bool = False
    too: bool = False
    either: bool = False                           # "left or right": ambiguous
    regardless: set = field(default_factory=set)   # {"colour", "shape"}
    dangling: bool = False                         # a bin was named but not understood ("the lft one")
    neg_leave: bool = False
    prio: bool = False
    prio_kind: set = field(default_factory=set)
    otherscope: bool = False
    lead: str = ""                                 # the boundary word before this chunk ("and", ",", "while" ...)
    cmp: tuple | None = None                       # ("left"|"right", token index) of "further left than"
    unique: bool = False
    normally: bool = False
    soft_dangling: bool = False                    # "... to that exception": a preposition, but no bin word
    after_colon: bool = False
    correct: bool = False                          # "no wait, I meant ...": overrides earlier rules
    negsels: list = field(default_factory=list)    # "no red in the center bin": constraints, not exclusions
    swaps: list = field(default_factory=list)
    usual: bool = False
    swapunk: bool = False
    first: str = ""                                # "S" or "T": which comes first

    @property
    def pos_tgts(self):
        return [t for t in self.tgts if not t.neg]

    @property
    def neg_tgts(self):
        return [t for t in self.tgts if t.neg]


def _is_bin_context(toks, i):
    """Is the position word toks[i] ('left'/'right'/...) a bin reference?"""
    w = toks[i]
    nxt = toks[i + 1] if i + 1 < len(toks) else ""
    nxt2 = toks[i + 2] if i + 2 < len(toks) else ""
    prv = toks[i - 1] if i > 0 else ""
    if w == "right":
        if nxt in {"away", "now", "after", "before", "then", "here", "there", "back", "off", "up", "down", "over",
                   "under", "next", "behind", "beside", "near"}:
            return False
        if prv in {"all", "that", "is"} and nxt in {"", ".", ",", "!", "?"} and prv != "is":
            return False
        if nxt in {"in", "into", "inside", "on", "onto", "at", "to"} and any(
                t in POS_WORDS or t == "__rembin__" for t in toks[i + 2:i + 5]):
            return False  # "put it right into the middle bin"
    if w in ("mid",) and nxt not in BIN_NOUNS:
        return False
    if nxt in NONBIN_NOUNS:
        return False  # "the right wall", "the robot's left hand"
    if w == "left" and prv in {"have", "has", "had", "i", "we", "you"}:
        return False
    if nxt in COUNTER_WORDS or (nxt in {"of", "part"} and (nxt2 in COUNTER_WORDS or nxt2 == "the")):
        return False  # "the middle of the counter"
    return True


def _chunk_mentions(ch: Chunk):
    toks = ch.toks
    neg_active = False
    except_active = False
    other_pending = False
    n = len(toks)
    sels_raw = []  # (index, Sel, excluded?)
    neg_at = -9
    neg_ended_at = None
    except_at = -9
    pos_before_except = False
    for i, w in enumerate(toks):
        if w in NEG_WORDS:
            neg_active = True
            neg_at = i
            continue
        if w == "__qualunk__":    # "the yellow one shaped like <something we can't read>": that selector is unsure
            for _, x, _ in sels_raw[-1:]:
                x.unknown = True
            continue
        if w in ("or", "nor", "and") and neg_ended_at is not None and i - neg_ended_at <= 2:
            neg_active = True     # "not in the left bin or the center bin"
            continue
        if w == "__except__":
            except_active = True
            except_at = i
            pos_before_except = any(not ex for _, _, ex in sels_raw)
            continue
        if w in ("other", "remaining", "leftover", "rest", "__rest_of__"):
            other_pending = True
            continue
        if w == "__correct__":
            ch.correct = True
            continue
        if w.startswith("__swap_") and w != "__swapunk__":
            ch.swaps.append(tuple(w[len("__swap_"):-2].split("_")))
            continue
        if w == "__swapunk__":
            ch.swapunk = True
            continue
        if w == "__usual__":
            ch.usual = True
            continue
        if w == "__prio__":
            ch.prio = True
            if any(x in ("shape", "shapes") for x in toks[:i]):
                ch.prio_kind.add("shape")
            elif any(x == "colour" for x in toks[:i]):
                ch.prio_kind.add("colour")
            continue
        if w == "__otherscope__":
            ch.otherscope = True
            continue
        if w == "__unique__":
            ch.unique = True
            continue
        if w in ("__cmp_left__", "__cmp_right__"):
            ch.cmp = ("left" if "left" in w else "right", i)
            continue
        if w == "__normally__":
            ch.normally = True
            continue
        if w in ("former", "latter") and (i == 0 or toks[i - 1] == "the"):
            sels_raw.append((i, Sel(ref=w, pos=i), False))
            continue
        if w == "__dontmove__":       # "don't touch/move X": leave X; "don't put X in the left bin": negation
            ch.leave = True
            neg_active = True
            continue
        if w in LEAVE_WORDS and neg_active and not except_active:
            neg_active = False        # "don't leave X on the counter": a double negative, not a leave
            ch.neg_leave = True
            continue
        if w in LEAVE_WORDS:
            ch.leave = True
        if w in COUNTER_WORDS:
            ch.counter = True
        if w in PRONOUNS:
            ch.pron = True
        if w in TOO_WORDS:
            ch.too = True
        if w.startswith("__regardless_"):
            ch.regardless.add("colour" if "col" in w else "shape")
        if w == "or" and ch.tgts:
            ch.either = True
        if w in ("__rest__", "__all__"):
            j = i + 1
            while j < n and toks[j] in ("the", "of", "those", "these", "that", "is", "are"):
                j += 1
            if j < n and (toks[j] in COLOR_WORDS or toks[j] in SHAPE_WORDS):
                other_pending = (w == "__rest__")   # "the rest of the blue pieces" / "everything yellow": a scope
                continue
            s = Sel(rest=True, pos=i, else_=(w == "__rest__"))
            sels_raw.append((i, s, except_active or (neg_active and not ch.tgts)))
            other_pending = False
            continue
        if (w in COLOR_WORDS or w in OTHER_COLOR_WORDS) and any(
                x in DISTRACTOR_NOUNS for x in toks[i + 1:i + 3]) and not any(
                x in OBJ_NOUNS or x in SHAPE_WORDS for x in toks[i + 1:i + 3]):
            continue  # "the red emergency stop", "the blue cable tray": not an object
        if w in COLOR_WORDS or w in OTHER_COLOR_WORDS:
            c = COLOR_WORDS.get(w)
            s = Sel(color=c, unknown=c is None, pos=i, other=other_pending)
            # attached shape: "red cubes", "red coloured triangles", "triangular red ones"
            for j in (i + 1, i + 2):
                if j < n and toks[j] in SHAPE_WORDS:
                    if j == i + 2 and toks[i + 1] not in {"colour", "coloured", "one"} | OBJ_NOUNS - {"ones"}:
                        break
                    s.shape = SHAPE_WORDS[toks[j]]
                    break
                if j < n and (toks[j] in COLOR_WORDS or toks[j] in POS_WORDS or toks[j].startswith("__")):
                    break
            if s.shape is None and i > 0 and toks[i - 1] in SHAPE_ADJ:
                s.shape = SHAPE_WORDS[toks[i - 1]]
            excluded = "excl" if except_active else ("neg" if neg_active and not ch.leave else False)
            if excluded == "neg" and all(x in _NEG_GLUE for x in toks[neg_at + 1:i]):
                neg_active = False    # "not blue goes right": the negation ends at the subject
                if sels_raw and not sels_raw[-1][2] and all(
                        x in _REL_GLUE or x in OBJ_NOUNS for x in toks[sels_raw[-1][0] + 1:neg_at]):
                    excluded = "excl"  # "everything that isn't blue", "any yellow object that is not a cylinder"
            sels_raw.append((i, s, excluded))
            other_pending = False
            continue
        if w in SHAPE_WORDS:
            # already consumed as a colour's shape?
            if any(s.shape == SHAPE_WORDS[w] and s.color and abs(idx - i) <= 2 for idx, s, _ in sels_raw):
                continue
            if i + 1 < n and (toks[i + 1] in COLOR_WORDS) and w in SHAPE_ADJ:
                continue  # "triangular red ones": the colour takes it
            s = Sel(shape=SHAPE_WORDS[w], pos=i, other=other_pending)
            excluded = "excl" if except_active else ("neg" if neg_active and not ch.leave else False)
            if excluded == "neg" and all(x in _NEG_GLUE for x in toks[neg_at + 1:i]):
                neg_active = False    # "not blue goes right": the negation ends at the subject
                if sels_raw and not sels_raw[-1][2] and all(
                        x in _REL_GLUE or x in OBJ_NOUNS for x in toks[sels_raw[-1][0] + 1:neg_at]):
                    excluded = "excl"  # "everything that isn't blue", "any yellow object that is not a cylinder"
            sels_raw.append((i, s, excluded))
            other_pending = False
            continue
        t = None
        if w in POS_WORDS and _is_bin_context(toks, i):
            t = Tgt(POS_WORDS[w], pos=i)
        elif w == "__rembin__":
            t = Tgt("rem", pos=i)
        elif w == "__outer__":
            t = Tgt("outer", pos=i)
        elif w in ("__rel_left__", "__rel_right__", "__rel_opp__", "__rel_same__", "__rel_adj__"):
            t = Tgt("rel:" + {"left": "-1", "right": "+1", "opp": "opp", "same": "0", "adj": "adj"}[w[6:-2]], pos=i)
        elif w.startswith("__adj_"):
            c = w[len("__adj_"):-2]
            t = Tgt(f"rel:adj:same:{c}" if c in COLORS else None, pos=i)
        elif w.startswith("__sameasdflt_"):
            c = COLOR_WORDS.get(w[len("__sameasdflt_"):-2])
            t = Tgt(f"dflt:{c}" if c else None, pos=i)
        elif w.startswith("__sameas_"):
            c = COLOR_WORDS.get(w[len("__sameas_"):-2])
            t = Tgt(f"same:{c}" if c else None, pos=i)
        if t is not None:
            t.neg = neg_active
            if not t.neg and ch.pos_tgts and ("or" in toks[max(0, i - 3):i]):
                ch.either = True
            ch.tgts.append(t)
            if t.neg:             # "red on the right not the middle yellow in the middle": the negation ends here
                neg_active, neg_ended_at = False, i
            continue
        if w in ("but", "instead", "rather"):
            neg_active = False
    # shape distribution over colour lists: "red and blue cubes" -> both cubes
    for k in range(len(sels_raw) - 1, 0, -1):
        i1, s1, _ = sels_raw[k]
        i0, s0, _ = sels_raw[k - 1]
        if s1.color and s1.shape and s0.color and not s0.shape and i1 - i0 <= 3 and all(
                toks[j] in {"and", "or", ",", "the"} for j in range(i0 + 1, i1)):
            s0.shape = s1.shape
    if except_active and not pos_before_except and any(not t.neg and t.pos > except_at for t in ch.tgts):
        # "..., except that every cylinder goes to the center bin": a rule of its own that outranks earlier ones
        sels_raw = [(i, x, False if ex == "excl" else ex) for i, x, ex in sels_raw]
        for _, x, _ in sels_raw:
            x.boost = 0.75
    for _, s, ex in sels_raw:
        if ex == "neg":   # negated subject: with a bin in this clause a constraint, else an exception
            (ch.negsels if ch.tgts else ch.excl).append(s)
        else:
            (ch.excl if ex else ch.sels).append(s)
    if ch.neg_leave:
        ch.counter = False
    ms = sorted([("S", s.pos) for s in ch.sels] + [("T", t.pos) for t in ch.pos_tgts], key=lambda x: x[1])
    ch.first = ms[0][0] if ms else ""
    if not ch.tgts:
        last_s = max([s.pos for s in ch.sels + ch.excl], default=-1)
        tail = toks[last_s + 1:]
        real_bin = BIN_NOUNS - {"one"}
        if any(w in real_bin for w in toks) or ("one" in toks and "the" in toks and not ch.sels):
            ch.dangling = True
        elif any(w in DIR_PREPS for w in tail[:-1]) and not ch.leave and not ch.counter:
            ch.soft_dangling = True   # "red to lft" or "red cylinders are an exception to that": don't merge forward
    return ch


def _split_chunks(sentence_toks):
    chunks, cur, colon, after = [], [], [], False
    leads, lead = [], ""
    toks = sentence_toks
    for i, w in enumerate(toks):
        boundary = w in {",", ":"} or w in SPLIT_WORDS
        if w == "__correct__":
            if cur:
                chunks.append(cur)
                colon.append(after)
                leads.append(lead)
            lead = "__correct__"
            after = False
            cur = [w]
            continue
        if w in ("and", "or", ",") and cur and cur[-1] in POS_WORDS and i + 1 < len(toks) and (
                toks[i + 1] in POS_WORDS or (toks[i + 1] == "the" and i + 2 < len(toks) and toks[i + 2] in POS_WORDS)):
            cur.append(w)        # "away from the left and right bins": one negated pair, not two clauses
            continue
        if w in ("and", "or") and cur and cur[-1] in COLOR_WORDS and i + 1 < len(toks) and (
                toks[i + 1] in COLOR_WORDS or (toks[i + 1] == "the" and i + 2 < len(toks) and toks[i + 2] in COLOR_WORDS)):
            cur.append(w)        # "all but the yellow and red objects", "for the yellow and blue objects"
            continue
        if w == "and" and i + 1 < len(toks) and toks[i + 1] in SHAPE_WORDS:
            boundary = False  # "red and blue cubes"? handled by distribution; "cubes and cylinders" stays together
        if boundary:
            if cur:
                chunks.append(cur)
                colon.append(after)
                leads.append(lead)
                lead = w
            elif w not in (",", "and"):
                lead = w            # ", while" -> while
            after = w == ":"
            cur = []
            continue
        cur.append(w)
    if cur:
        chunks.append(cur)
        colon.append(after)
        leads.append(lead)
    out = [Chunk(c, after_colon=a) for c, a in zip(chunks, colon)]
    for ch, ld in zip(out, leads):
        ch.lead = ld
    return out


def _sentences(toks):
    out, cur = [], []
    for w in toks:
        if w in {".", ";", "!", "?", "\n"}:
            if cur:
                out.append(cur + (["__q__"] if w == "?" else []))
            cur = []
        else:
            cur.append(w)
    if cur:
        out.append(cur)
    return out


# ---------------------------------------------------------------------------------------------------- rules
@dataclass
class Rule:
    sel: Sel
    bin: str | None              # bin name, "rem", "same:<c>", or "LEAVE"
    excl: list = field(default_factory=list)
    boost: float = 0.0
    epoch: int = 0                 # bumped by a self-correction; a later rule on the same selector wins
    default: bool = False          # stated as the normal/usual mapping
    mentioned: list = field(default_factory=list)   # for "everything else": every other selector in the text

    def matches(self, c, s):
        if not self.sel.matches(c, s) or any(e.matches(c, s) for e in self.excl if not e.unknown):
            return False
        if self.sel.else_ and any(m.matches(c, s) for m in self.mentioned):
            return False
        return True

    def spec(self):
        return self.sel.spec() + max(self.boost, self.sel.boost)


class _Parser:
    def __init__(self):
        self.rules: list[Rule] = []
        self.neg: list[tuple] = []     # (Sel, bin): this selector must not go to that bin
        self.notes: list[str] = []
        self.epoch = 0
        self.antecedent = []           # last subjects, across sentences ("... They go right.")
        self.mentioned = []            # every colour/shape selector in the text
        self.last_bin = None           # last positive target, for "the bin to the left of that one"
        self.last_rule_bin = None      # for "Blue as well." in a later sentence
        self.history = []              # distinct selectors in order of mention (former / latter)
        self.prio_kind, self.prio_sels = set(), []
        self.otherscope = False
        self.unique = False
        self.in_default = False
        self.answer_to = None
        self.cmp = []                  # (a, b): colour a is in a bin further left than colour b
        self.groups = []               # selectors that go to different bins ("the two outer bins")
        self.swaps = []                # ("red", "blue") / ("left", "right"): applied to the resolved table
        self.swapunk = False           # a swap/reverse we could not read: the whole mapping is unsure

    def add(self, sels, tgt, excl=(), regardless=()):
        for s in sels:
            if s.unknown or s.ref:
                continue
            if self.otherscope and not s.rest:
                s.other = True
            boost = 0.0
            if s.shape and not s.color and "colour" in regardless:
                boost = 0.75
            if s.color and not s.shape and "shape" in regardless:
                boost = 0.75
            if isinstance(tgt, str) and tgt.startswith("rel:"):
                tgt = tgt + ":" + str(self.last_bin) if ":" not in tgt[4:] else tgt
            if any(r.sel.else_ and any(e is s for e in r.excl) for r in self.rules):
                s.other = True     # "all other objects go right, apart from the yellow ones, which go left"
            self.rules.append(Rule(s, tgt, list(excl), boost, epoch=self.epoch, mentioned=self.mentioned,
                                   default=self.in_default))
        if tgt not in (None, "LEAVE"):
            self.last_bin = self.rules[-1].bin if self.rules else tgt
            if sels and any(not x.unknown for x in sels):
                self.last_rule_bin = self.last_bin

    def run(self, text):
        norm = normalise(text)
        self.norm = norm
        toks = tokenize(norm)
        sents = _sentences(toks)
        sents = [[w for w in x if w != "__q__"] + (["__q__"] if "__q__" in x else []) for x in sents]
        for sent in sents:   # every concrete selector, for "everything else"
            for ch in _split_chunks([w for w in sent if w != "__q__"]):
                ch = _chunk_mentions(ch)
                self.mentioned.extend(x for x in ch.sels + ch.excl if not x.rest and not x.unknown)
        for sent in sents:
            dflt = "__normally__" in sent and not any(w in ("today", "now", "tonight", "__correct__") for w in sent)
            if dflt:   # "Normally blue goes left ...": a default layer that later statements override
                saved, self.epoch, self.in_default = self.epoch, -500, True
            question = sent[-1:] == ["__q__"]
            n0 = len(self.rules)
            self._sentence([w for w in sent if w != "__q__"])
            if dflt:
                self.epoch, self.in_default = saved, False
            # "Where do the reds go? The right bin." / "The blues? Middle.": the next sentence answers
            self.answer_to = list(self.antecedent) if question and len(self.rules) == n0 and self.antecedent else None
        for r in self.rules:   # "the shape rule wins", "cylinders take priority"
            sh_only, co_only = r.sel.shape and not r.sel.color, r.sel.color and not r.sel.shape
            if (sh_only and ("shape" in self.prio_kind or any(p.shape == r.sel.shape and not p.color for p in self.prio_sels))) \
                    or (co_only and ("colour" in self.prio_kind or any(p.color == r.sel.color and not p.shape
                                                                       for p in self.prio_sels))):
                r.boost = max(r.boost, 0.75)

    # -------------------------------------------------------------- sentence level
    def _sentence(self, toks):
        if self._ordered_list(toks):
            return
        chunks = [_chunk_mentions(c) for c in _split_chunks(toks)]
        # "move the yellow | and blue cubes to the right": a chunk ending in a bare colour takes the next one's shape
        for a, b in reversed(list(zip(chunks, chunks[1:]))):
            if a.sels and a.toks and a.toks[-1] in COLOR_WORDS and b.sels and b.sels[0].pos == min(
                    [x.pos for x in b.sels]) and b.sels[0].color and b.sels[0].shape and b.toks[b.sels[0].pos] in COLOR_WORDS \
                    and all(w in {"the", "all", "both", "also"} for w in b.toks[:b.sels[0].pos]):
                for x in a.sels:
                    if x.color and not x.shape and x.pos == len(a.toks) - 1:
                        x.shape = b.sels[0].shape
        pend_s, pend_t, pend_x = [], None, []
        pend_leave = []
        pend_bare = True
        pend_t_ci = -9
        list_t = None                  # target of a bin-first list ("Center bin: red | and yellow")
        last_rule_chunk = None
        last_chunk = None
        antecedent = self.antecedent
        correcting = False
        answer = self.answer_to
        if answer:
            antecedent = answer
        last_np = []
        scope = []
        sent_excl = []
        hist_chunks = chunks
        n_sent_start = len(self.rules)
        pend_reg = set()
        n_last_rules, n_mark = 0, len(self.rules)
        for ci, ch in enumerate(chunks):
            if ci > 0:
                for x in hist_chunks[ci - 1].sels + hist_chunks[ci - 1].excl + hist_chunks[ci - 1].negsels:
                    if not x.rest and not x.unknown and not x.ref and not any(
                            (h.color, h.shape) == (x.color, x.shape) for h in self.history):
                        self.history.append(x)
            if len(self.rules) > n_mark:          # how many rules the previous chunk added
                n_last_rules = len(self.rules) - n_mark
            n_mark = len(self.rules)
            if pend_reg:
                ch.regardless = ch.regardless | pend_reg
            for x in ch.sels:      # "the former / the latter" = the earlier / later of the last two things named
                if x.ref:
                    hist = self.history[-2:]
                    h = hist[0] if (x.ref == "former" and len(hist) == 2) else (hist[-1] if x.ref == "latter" and hist else None)
                    if h is None:
                        x.unknown = True
                    else:
                        x.color, x.shape, x.ref = h.color, h.shape, None
            if ch.prio:
                self.prio_kind |= ch.prio_kind
                self.prio_sels.extend(x for x in ch.sels if not x.rest)
            if ch.otherscope:
                self.otherscope = True
            if ci > 0:
                prev = chunks[ci - 1]
                nps = [(x.pos, [x]) for x in prev.sels + prev.excl + prev.negsels]
                nps += [(t.pos, [Sel(color=t.bin[5:])]) for t in prev.tgts
                        if isinstance(t.bin, str) and t.bin.startswith("same:") and t.bin[5:] in COLORS]
                if nps:
                    last_np = max(nps, key=lambda x: x[0])[1]
            prev_last_tgt = None
            if ci > 0:
                pc = chunks[ci - 1]
                ms = [(x.pos, None) for x in pc.sels + pc.excl + pc.negsels] + [(t.pos, t) for t in pc.tgts]
                if ms:
                    prev_last_tgt = max(ms, key=lambda x: x[0])[1]
                    if prev_last_tgt is not None and prev_last_tgt.bin not in BINS:
                        prev_last_tgt = None
            nxt = chunks[ci + 1] if ci + 1 < len(chunks) else None
            next_is_target = nxt is not None and nxt.pos_tgts and not nxt.sels and not nxt.dangling
            if ch.correct or (ch.lead == "but" and ch.pron and not ch.sels and ch.pos_tgts):
                self.epoch += 1           # "... might look like the spot for blue, but they must go right"
                correcting = True
            comp = [t for t in ch.pos_tgts if isinstance(t.bin, str) and t.bin.startswith("same:")]
            conc = [t for t in ch.pos_tgts if t.bin in BINS]
            if comp and len(conc) == 1:   # "in the left bin together with the red items": red goes there too
                for t in comp:
                    self.add([Sel(color=t.bin[5:])], conc[0].bin)
                ch.tgts = [t for t in ch.tgts if t not in comp]
            if ch.toks and ch.toks[0] in ("which", "who") and last_np:
                antecedent = last_np      # relative "which" = the last noun phrase ("the same bin as blue, which ...")
            self.swaps.extend(ch.swaps)
            self.swapunk |= ch.swapunk
            if ch.usual:
                self._add_usual()
            for s_ in ch.negsels:
                for t in ch.tgts:
                    self.neg.append((s_, t.bin))
            relative = bool(ch.toks) and ch.toks[0] in ("which", "who", "that")
            transparent = (not ch.sels and not ch.tgts and (ch.negsels or ch.excl or relative)) or \
                (not ch.sels and ch.tgts and not ch.pos_tgts) or (relative and ch.sels and not ch.tgts)
            if ch.unique:
                self.unique = True
            if pend_t is not None and pend_t_ci != ci - 1:
                pend_t = None           # a pending bin only binds to the clause right after it ...
            if pend_t is not None and transparent:
                pend_t_ci = ci          # ... or passes through "not blue"-only clauses
            sels, pt = ch.sels, ch.pos_tgts
            if ch.cmp is not None and not pt:
                d, at = ch.cmp           # "red goes further left than blue": order constraint only
                a = [x for x in ch.sels if x.pos < at and x.color and not x.shape]
                b = [x for x in ch.sels if x.pos > at and x.color and not x.shape]
                for x in a:
                    for y in b:
                        self.cmp.append((x.color, y.color) if d == "left" else (y.color, x.color))
                antecedent = a or antecedent
                last_chunk = ch
                continue
            if ch.after_colon and pend_s and ch.sels:
                pend_s, pend_x = [], []  # a header before ':' never merges into the subjects that follow it
            outer = [t for t in ch.tgts if t.bin == "outer"]
            if outer:                    # "red and yellow go into the two outer bins": not center, one each
                grp = [x for x in (ch.sels or antecedent) if not x.rest]
                for x in grp:
                    self.neg.append((x, "center"))
                if len(grp) >= 2:
                    self.groups.append(grp)
                ch.tgts = [t for t in ch.tgts if t.bin != "outer"]
                pt = ch.pos_tgts
                if not pt:
                    antecedent = grp or antecedent
                    last_chunk = ch
                    continue
            shape_hdr = any(x in ("by",) for x in ch.toks) and ("colour" in ch.toks or ch.otherscope) and ch.sels and all(
                x.shape and not x.color and not x.rest for x in ch.sels)
            colour_hdr = ch.toks[:1] in (["for"], ["regarding"], ["concerning"], ["about"]) or ch.toks[:2] == ["as", "for"]
            colour_hdr = colour_hdr and ci == 0 and ch.sels and all(not x.rest and (x.color is None) != (x.shape is None) for x in ch.sels)
            if (shape_hdr or colour_hdr) and not pt and not ch.tgts:
                scope = list(ch.sels)   # "Sort the cylinders by colour: red left ..." / "For the red objects: cubes left"
                self.notes.append(f"rules scoped to {[(x.color, x.shape) for x in scope]}")
                last_chunk = ch
                continue
            if scope and sels:
                exp = []
                for x in sels:
                    if x.rest:
                        exp += [Sel(color=y.color, shape=y.shape, pos=x.pos) for y in scope]
                    elif x.color and not x.shape and any(y.shape and not y.color for y in scope):
                        exp += [Sel(color=x.color, shape=y.shape, pos=x.pos) for y in scope if y.shape]
                    elif x.shape and not x.color and any(y.color and not y.shape for y in scope):
                        exp += [Sel(color=y.color, shape=x.shape, pos=x.pos) for y in scope if y.color]
                    else:
                        exp.append(x)
                sels = ch.sels = exp
            if sels and pt and ch.lead in ("but", "however", "though", "although", "yet"):
                for x in sels:   # "sort by colour (...), but the cubes all go into the middle": an exception
                    if any((r.sel.color, r.sel.shape) != (x.color, x.shape) and not r.sel.rest and any(
                            r.matches(c, sh) and x.matches(c, sh) for c in COLORS for sh in SHAPES) for r in self.rules):
                        x.boost = max(x.boost, 0.75)
            if sels and pt:
                list_t = pt[0] if (ch.first == "T" and len(pt) == 1) else None
            for t in ch.neg_tgts:
                # a negated bin constrains the subjects right before it (after the previous positive bin), else all
                before = [x.pos for x in sels if x.pos < t.pos]
                near = []
                if before:   # the subject group nearest before it ("red on the right, not the middle")
                    last = max(before)
                    lo = max([u.pos for u in pt if u.pos < last], default=-1)
                    near = [x for x in sels if lo < x.pos <= last]
                for s in (near or sels or ([] if ch.negsels else antecedent)):
                    self.neg.append((s, t.bin))
            if ch.negsels and not sels:
                antecedent = ch.negsels
            if ch.either and len(pt) >= 2:
                self.notes.append("either/or target: ambiguous")
                antecedent = sels or antecedent
                pend_s = []
                last_chunk = ch
                continue
            if len(pt) >= 2:
                self._multi(ch)
                pend_s, pend_t = [], None
                last_rule_chunk = last_chunk = ch
                antecedent = sels or antecedent
                continue
            if sels and pt and pend_s and ch.after_colon:
                self.notes.append("header before ':' not merged")   # "Sort red and yellow only: red right, ..."
                pend_s, pend_x = [], []
            if sels and pt and pend_s and (not pend_bare or ch.lead not in ("and", ",", "or", "")):
                self.notes.append("pending subjects not merged across a clause")   # "..., while the yellow | ..."
                pend_s, pend_x = [], []
            if sels and pt:
                excl = ch.excl + pend_x
                self.add(sels + pend_s, pt[0].bin, excl, ch.regardless)
                pend_s, pend_t, pend_x = [], None, []
                last_rule_chunk = last_chunk = ch
                antecedent = [s for s in sels if not s.rest] or antecedent
                if ch.excl:
                    antecedent = ch.excl
                continue
            if sels and not pt and relative and ci > 0 and prev_last_tgt is not None:
                self.add(sels, prev_last_tgt.bin, ch.excl, ch.regardless)   # "the left one, which is for red"
                last_chunk = ch
                antecedent = sels
                continue
            if sels and not pt:
                if ch.too and self.last_rule_bin is not None and not next_is_target:
                    self.add(sels, self.last_rule_bin, ch.excl, ch.regardless)   # "Blue as well."
                    last_chunk = ch
                    antecedent = sels
                    continue
                if pend_t is not None and not ch.dangling:
                    self.add(sels + pend_s, pend_t.bin, ch.excl + pend_x, ch.regardless)
                    list_t = pend_t
                    pend_s, pend_t, pend_x = [], None, []
                    last_rule_chunk = last_chunk = ch
                    ch.tgts = ch.tgts + [Tgt(self.rules[-1].bin if self.rules else None)]
                    antecedent = sels
                    continue
                if list_t is not None and ch.lead in ("and", ",", "or") and not (
                        ch.leave or ch.counter or ch.dangling or ch.neg_tgts or next_is_target):
                    self.add(sels, list_t.bin, ch.excl, ch.regardless)   # bin-first list continues
                    antecedent = sels
                    last_chunk = ch
                    continue
                if ch.leave and not ch.neg_tgts:
                    self.add(sels, "LEAVE", ch.excl)
                    last_chunk = ch
                    antecedent = sels
                    continue
                if ch.soft_dangling and not next_is_target:
                    last_chunk = ch
                    antecedent = sels
                    continue
                if ch.dangling:  # "red in the lft one": its bin is unreadable -> leave red unmapped
                    self.notes.append(f"unreadable target for {[(s.color, s.shape) for s in sels]}")
                    last_chunk = ch
                    antecedent = sels
                    pend_s, pend_x = [], []
                    continue
                if ch.neg_tgts:  # "blue does not go left" -> constraint only; maybe a target follows
                    last_chunk = ch
                    antecedent = sels
                    continue
                if (last_chunk is not None and last_chunk is last_rule_chunk and last_chunk.first == "T"
                        and not ch.leave and len(last_chunk.pos_tgts) == 1 and not next_is_target
                        and ch.lead in ("and", ",", "or")):
                    # "the left bin takes red, (and) yellow" -> list continues backwards
                    self.add(sels, last_chunk.pos_tgts[0].bin, ch.excl, ch.regardless)
                    antecedent = sels
                    continue
                if ch.counter:          # "... on the counter": leave them, unless a bin follows in this sentence
                    pend_leave = pend_leave + sels
                pend_bare = (pend_bare if pend_s else True) and _is_bare(ch.toks)
                pend_s = pend_s + sels
                pend_x = pend_x + ch.excl
                antecedent = sels
                last_chunk = ch
                continue
            if not sels and not ch.excl and not ch.tgts and ch.regardless:
                for r in self.rules[len(self.rules) - n_last_rules:]:     # "cylinders go left, regardless of colour"
                    if (r.sel.shape and not r.sel.color and "colour" in ch.regardless) or \
                            (r.sel.color and not r.sel.shape and "shape" in ch.regardless):
                        r.boost = max(r.boost, 0.75)
                pend_reg |= ch.regardless
                last_chunk = ch
                continue
            if not sels and ch.excl and not pt and len(ch.toks) > 1 and ch.toks[0] == "__except__" \
                    and ch.toks[1] in ("that", "when", "if", "where"):
                # "..., except that every cylinder, ..., goes to the center": an exception clause with its own bin
                for x in ch.excl:
                    x.boost = 0.75
                pend_s = pend_s + ch.excl
                antecedent = ch.excl
                last_chunk = ch
                continue
            if not sels and ch.excl and not pt and ci == 0:
                sent_excl = sent_excl + ch.excl     # "Apart from cylinders, red goes left and ...": the whole sentence
                antecedent = ch.excl
                last_chunk = ch
                continue
            if not sels and ch.excl and not pt:
                # "except blue" / "not the red ones": attach to the rules of this sentence so far, else to the next
                if last_rule_chunk is not None:
                    for r in self.rules[n_sent_start:] or self.rules[-max(1, len(last_rule_chunk.sels)):]:
                        r.excl.extend(ch.excl)
                else:
                    pend_x = pend_x + ch.excl
                antecedent = ch.excl
                last_chunk = ch
                continue
            if not sels and pt:
                t = pt[0]
                if pend_s:
                    self.add(pend_s, t.bin, ch.excl + pend_x, ch.regardless)
                    pend_s, pend_x = [], []
                    last_rule_chunk = last_chunk = ch
                    ch.sels = []
                    continue
                if ch.excl:
                    # "except blue, which goes right" style handled by pronoun; here: "left bin except blue"
                    pass
                if (ch.pron or correcting or answer or (last_chunk is not None and not last_chunk.pos_tgts
                                                        and last_chunk.sels and not last_chunk.dangling)) and antecedent:
                    answer = None
                    if last_chunk is not None and last_chunk.excl and not last_chunk.sels and not ch.pron:
                        pass
                    self.add(antecedent, t.bin, [], ch.regardless)
                    last_rule_chunk = last_chunk = ch
                    continue
                # "blue in the middle box and ALSO in the right box": one colour, two bins, no correction -> a conflict
                # (same epoch, same specificity -> unmapped by _best), not a silent pick of the first bin.
                if ("also" in ch.toks or ch.too) and last_chunk is last_rule_chunk and last_chunk is not None \
                        and last_chunk.sels and last_chunk.pos_tgts and t.bin in BINS and not ch.neg_tgts and not correcting:
                    self.add(last_chunk.sels, t.bin, [], ch.regardless)
                    last_rule_chunk = last_chunk = ch
                    continue
                pend_t = t
                pend_t_ci = ci
                last_chunk = ch
                continue
            if (ch.leave or ch.counter) and not sels and not ch.pos_tgts and ch.pron and antecedent:
                self.add(antecedent, "LEAVE")      # "... and put it back on the counter"
            last_chunk = ch
        for r in self.rules[n_sent_start:]:
            r.excl.extend(x for x in sent_excl if not (x.color == r.sel.color and x.shape == r.sel.shape))
        self.antecedent = antecedent
        if pend_s and pend_leave and all(any(x is y for y in pend_leave) for x in pend_s):
            self.add(pend_s, "LEAVE", pend_x)
            pend_s = []
        if pend_s:
            self.notes.append(f"unassigned subjects {[(s.color, s.shape) for s in pend_s]}")

    def _add_usual(self):
        """'the usual setup': the rung-1 mapping, overridable by anything said later (lower epoch)."""
        ep, self.epoch, dd = self.epoch, self.epoch - 1000, self.in_default
        self.in_default = True
        for c, b in (("red", "left"), ("yellow", "center"), ("blue", "right")):
            self.add([Sel(color=c)], b)
        self.epoch, self.in_default = ep, dd

    def _multi(self, ch):
        """Several target bins in one chunk without punctuation: pair subjects and targets in order."""
        ms = sorted([("S", s.pos, s) for s in ch.sels] + [("T", t.pos, t) for t in ch.pos_tgts], key=lambda x: x[1])
        if not ms:
            return
        groups = []
        if ms[0][0] == "S":
            cur = []
            for kind, _, obj in ms:
                if kind == "S":
                    cur.append(obj)
                elif cur:
                    groups.append((cur, obj))
                    cur = []
        else:
            cur_t = None
            for kind, _, obj in ms:
                if kind == "T":
                    cur_t = obj
                elif cur_t is not None:
                    groups.append(([obj], cur_t))
        for sels, t in groups:
            self.add(sels, t.bin, ch.excl, ch.regardless)

    def _ordered_list(self, toks):
        """'red, yellow and blue go into the left, center and right bins respectively' / 'from left to right: ...'."""
        resp = "__resp__" in toks
        ltr = "__ltr__" in toks or "__rtl__" in toks
        if not (resp or ltr):
            # bare runs: S S S T T T or T T T S S S
            ch = _chunk_mentions(Chunk(list(toks)))
            if ch.excl or ch.neg_tgts:
                return False
            ms = sorted([("S", s.pos) for s in ch.sels] + [("T", t.pos) for t in ch.pos_tgts], key=lambda x: x[1])
            kinds = "".join(k for k, _ in ms)
            m = re.fullmatch(r"(S{2,})(T{2,})|(T{2,})(S{2,})", kinds)
            if not m:
                return False
            a = m.group(1) or m.group(3)
            b = m.group(2) or m.group(4)
            if len(a) != len(b) or any(s.rest for s in ch.sels):
                return False
            listy = {",", "and", "the", "or", "colour", "coloured"} | OBJ_NOUNS | BIN_NOUNS
            for (k0, p0), (k1, p1) in zip(ms, ms[1:]):
                if k0 == k1 and not all(w in listy for w in toks[p0 + 1:p1]):
                    return False  # not a plain list: "rest after putting red in the center goes right"
            sels, tgts = ch.sels, ch.pos_tgts
            for s, t in zip(sorted(sels, key=lambda s: s.pos), sorted(tgts, key=lambda t: t.pos)):
                self.add([s], t.bin, [], ch.regardless)
            return True
        ch = _chunk_mentions(Chunk([w for w in toks if w not in {"__ltr__", "__rtl__", "__resp__"}]))
        sels = sorted(ch.sels, key=lambda s: s.pos)
        tgts = sorted(ch.pos_tgts, key=lambda t: t.pos)
        if ltr and not tgts:
            order = list(BINS) if "__ltr__" in toks else list(BINS[::-1])
            if len(sels) != 3:
                self.notes.append("left-to-right list without 3 subjects")
                return True
            for s, b in zip(sels, order):
                self.add([s], b, ch.excl, ch.regardless)
            return True
        if resp and len(sels) == len(tgts) and len(sels) >= 2:
            for s, t in zip(sels, tgts):
                self.add([s], t.bin, ch.excl, ch.regardless)
            return True
        if resp:
            self.notes.append("respectively with unequal lists")
            return False
        return False

    # -------------------------------------------------------------- resolution
    def table(self):
        """(colour, shape) -> bin | None for all 9 object types."""
        self._used = None
        out = self._table_once()
        out = self._apply_swaps(out)
        rem = [r for r in self.rules if r.bin == "rem"]
        if rem:
            used = set()
            for (c, s), v in out.items():
                if v in BINS and not any(r.bin == "rem" for r in self._top(c, s)):
                    used.add(v)
            self._used = used
            out = self._apply_swaps(self._table_once())
        return out

    def _apply_swaps(self, out):
        if self.swapunk:
            self.notes.append("unreadable swap/reverse: nothing mapped")
            return {k: None for k in out}
        seen = []
        for a, b in self.swaps:
            if {a, b} in seen:
                continue                 # "mirror it, so that left and right trade places" is ONE swap
            seen.append({a, b})
            if a in COLORS:
                for s in SHAPES:
                    out[(a, s)], out[(b, s)] = out[(b, s)], out[(a, s)]
            else:
                sw = {a: b, b: a}
                out = {k: sw.get(v, v) for k, v in out.items()}
        return out

    def _top(self, c, s):
        rs = self._matching(c, s)
        if not rs:
            return []
        top = max(r.spec() for r in rs)
        return [r for r in rs if r.spec() == top]

    def _table_once(self):
        # pass 1: concrete bins (and LEAVE)
        out = {}
        for c in COLORS:
            for s in SHAPES:
                out[(c, s)] = self._resolve(c, s, concrete=True)
        # pass 2: remaining bin / same-as references
        for c in COLORS:
            for s in SHAPES:
                if out[(c, s)] == "?rem" or (isinstance(out[(c, s)], str) and out[(c, s)].startswith("?same")):
                    out[(c, s)] = None
        for k, v in list(out.items()):
            if v == "LEAVE":
                out[k] = None
        # negation deduction: an object type with no rule but two excluded bins goes to the third
        for c in COLORS:
            for s in SHAPES:
                if out[(c, s)] is None and not self._has_leave(c, s):
                    banned = {b for sel, b in self.neg if sel.matches(c, s) and b in BINS}
                    if len(banned) == 2 and not self._matching(c, s):
                        out[(c, s)] = next(b for b in BINS if b not in banned)
        for grp in self.groups:   # "red and yellow into the two outer bins, yellow into the left one" -> red right
            for x in grp:
                for c in COLORS:
                    for s in SHAPES:
                        if not x.matches(c, s) or out[(c, s)] is not None or self._has_leave(c, s) or self._matching(c, s):
                            continue
                        banned = {b for sel, b in self.neg if sel.matches(c, s) and b in BINS}
                        for y in grp:
                            if y is not x:
                                banned |= {out[(c2, s2)] for c2 in COLORS for s2 in SHAPES
                                           if y.matches(c2, s2) and out[(c2, s2)] in BINS and (c2 != c or s2 == s)}
                        left = [b for b in BINS if b not in banned]
                        if len(left) == 1:
                            out[(c, s)] = left[0]
        if self.unique:   # "each colour gets its own bin": eliminate
            for _ in range(3):
                taken = {}
                for c in COLORS:
                    v = {out[(c, s)] for s in SHAPES}
                    if len(v) == 1 and None not in v:
                        taken[c] = v.pop()
                for c in COLORS:
                    if c in taken or any(self._has_leave(c, s) or self._matching(c, s) for s in SHAPES):
                        continue
                    banned = {b for sel, b in self.neg if sel.color == c and not sel.shape and b in BINS}
                    allowed = [b for b in BINS if b not in banned and b not in taken.values()]
                    if len(allowed) == 1:
                        for s in SHAPES:
                            out[(c, s)] = allowed[0]
        if self.cmp:      # order constraints: fill colours without rules if exactly one assignment fits
            import itertools
            fixed, free = {}, []
            for c in COLORS:
                v = {out[(c, s)] for s in SHAPES}
                if len(v) == 1 and None not in v:
                    fixed[c] = v.pop()
                elif not any(self._has_leave(c, s) or self._matching(c, s) for s in SHAPES):
                    free.append(c)
            sols = []
            for combo in itertools.product(BINS, repeat=len(free)):
                a = dict(fixed, **dict(zip(free, combo)))
                if self.unique and len(set(a.values())) < len(a):
                    continue
                if any(x in a and y in a and not BINS.index(a[x]) < BINS.index(a[y]) for x, y in self.cmp):
                    continue
                if any(sel.color in a and not sel.shape and a[sel.color] == b for sel, b in self.neg):
                    continue
                sols.append(a)
            if len(sols) == 1:
                for c in free:
                    for s in SHAPES:
                        out[(c, s)] = sols[0][c]
        return out

    def _matching(self, c, s):
        return [r for r in self.rules if r.matches(c, s)]

    def _has_leave(self, c, s):
        return any(r.bin == "LEAVE" for r in self._matching(c, s))

    def _rule_bin(self, r, c, s, depth=0):
        b = r.bin
        if b in BINS or b == "LEAVE":
            return b
        if b == "rem":
            used = self._used
            if used is None:
                used = {r2.bin for r2 in self.rules if r2 is not r and r2.bin in BINS}
            free = [x for x in BINS if x not in used]
            return free[0] if len(free) == 1 else None
        if isinstance(b, str) and b.startswith("dflt:") and depth < 3:
            rs = [r2 for r2 in self._matching(b[5:], s) if r2.default]
            if not rs:
                return None
            top = max(r2.spec() for r2 in rs)
            bins = {self._rule_bin(r2, b[5:], s, depth + 1) for r2 in rs if r2.spec() == top}
            return bins.pop() if len(bins) == 1 and None not in bins else None
        if isinstance(b, str) and b.startswith("rel:") and depth < 3:
            _, d, inner = b.split(":", 2)
            if inner in BINS:
                ib = inner
            else:
                ib = self._rule_bin(Rule(r.sel, inner if inner != "None" else None), c, s, depth + 1)
            if ib not in BINS:
                return None
            if d == "opp":
                return {"left": "right", "right": "left"}.get(ib)
            if d == "adj":
                return {"left": "center", "right": "center"}.get(ib)
            i = BINS.index(ib) + int(d)
            return BINS[i] if 0 <= i < 3 else None
        if isinstance(b, str) and b.startswith("same:") and depth < 3:
            c2 = b.split(":", 1)[1]
            r2 = self._best(c2, s, depth + 1)
            return r2
        return None

    def _best(self, c, s, depth=0):
        rs = self._matching(c, s)
        if not rs:
            return None
        top = max(r.spec() for r in rs)
        best = [r for r in rs if r.spec() == top]
        key = lambda r: (r.sel.color, r.sel.shape, r.sel.rest, r.sel.other)   # noqa: E731
        latest = {}
        for r in best:
            latest[key(r)] = max(latest.get(key(r), r.epoch), r.epoch)
        if len(latest) == 1:           # a self-correction replaces what it corrects
            best = [r for r in best if r.epoch == latest[key(r)]]
        bins = {self._rule_bin(r, c, s, depth) for r in best}
        if len(bins) != 1:
            return None
        return bins.pop()

    def _resolve(self, c, s, concrete=True):
        b = self._best(c, s)
        if b is None or b == "LEAVE":
            return b
        for sel, nb in self.neg:
            if nb == b and sel.matches(c, s):
                return None  # told both "goes X" and "not X": unsure
        return b


def parse_debug(text):
    p = _Parser()
    p.run(text or "")
    return p


def parse_table(text):
    """(colour, shape) -> bin or None for all 9 object types."""
    return parse_debug(text).table()


def table_to_mapping(tab):
    """Full 9-type table -> compact mapping with colour keys where all shapes of a colour agree."""
    m = {}
    for c in COLORS:
        bins = [tab.get((c, s)) for s in SHAPES]
        if bins[0] is not None and all(b == bins[0] for b in bins):
            m[c] = bins[0]
        else:
            for s, b in zip(SHAPES, bins):
                if b is not None:
                    m[(c, s)] = b
    return m


def parse_instruction(text):
    """English instruction -> mapping {colour | (colour, shape): bin}. Unmapped types are absent."""
    return table_to_mapping(parse_table(text))


def mapping_for_task(task, instruction):
    """sort_fixed keeps the fixed mapping whatever the text; insert_shapes has no bin mapping."""
    from aaltonomous.interface.tasks import FIXED_MAPPING
    if task == "sort_fixed":
        return dict(FIXED_MAPPING)
    if task == "insert_shapes":
        return {}
    return parse_instruction(instruction or "")


def resolve(mapping, color, shape):
    """scene.resolve_bin semantics (most specific key wins), without importing the scene module."""
    for k in ((color, shape), color, shape, "default"):
        if k is not None and k in mapping:
            return mapping[k]
    return None


def target_bin(mapping, color, shape=None):
    """Bin for a detected object. An unknown (None) colour or shape is any of its values: only answer if every
    possible type agrees (an unknown colour must never fall through to a 'default' / 'the rest' key)."""
    if not mapping:
        return None
    bins = {resolve(mapping, c, s) for c in ((color,) if color is not None else COLORS)
            for s in ((shape,) if shape is not None else SHAPES)}
    return bins.pop() if len(bins) == 1 else None


def mapping_to_json(mapping):
    """(colour, shape) keys -> 'colour/shape' strings (sim scene service format)."""
    return {(f"{k[0]}/{k[1]}" if isinstance(k, tuple) else k): v for k, v in mapping.items()}


def mapping_from_json(mapping):
    return {(tuple(k.split("/")) if isinstance(k, str) and "/" in k else k): v for k, v in (mapping or {}).items()}


def expand(mapping):
    """Any mapping (colour / shape / (colour, shape) / default keys, or 'c/s' strings) -> 9-type table."""
    m = mapping_from_json(mapping)
    return {(c, s): resolve(m, c, s) for c in COLORS for s in SHAPES}


if __name__ == "__main__":
    import sys
    for line in (sys.argv[1:] or sys.stdin.read().splitlines()):
        p = parse_debug(line)
        print(line)
        print("   norm:", p.norm)
        print("   ->", table_to_mapping(p.table()), p.notes)
