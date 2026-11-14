"""Constants shared by the training env and the deployment skill (numpy-free, no Isaac imports)."""
ARM_VEL_CAP, HAND_VEL_CAP = 1.0, 3.0
CAP_FRAC = 0.97                   # use 97 % of the cap (float + filter rounding)
# Action spaces: indices into the 14-vector [left arm 7, left hand 7 (left DDS order)].
ACTION_DIMS = {"hand": list(range(7, 14)), "hand_wrist": [4, 5, 6] + list(range(7, 14)), "full": list(range(14))}
