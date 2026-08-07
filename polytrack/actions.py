"""Discrete action space for PolyTrack.

PolyTrack is driven with four directional inputs (up/down/left/right, bound to
WASD or the arrow keys) plus a reset key. Two facts from the game's changelog
shape the action space:

  * v0.6.0: "Left & right inputs now cancel out instead of preferring steering
    to the left" -- so (left AND right) is behaviourally identical to neither.
  * Forward and back are likewise mutually exclusive in any useful policy.

That leaves 3 longitudinal states x 3 lateral states = 9 meaningful actions.
Reset is deliberately NOT in the agent's action space: it is an episode control
signal owned by the environment, not something a policy should learn to spam.

Key names follow the .tas convention used by beatrixwashere/polytracktas, which
writes held keys as a subset of the letters "wasd".
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final

# Bit flags for the raw input mask. Order matches the recording format's
# input ordering (up, right, down, left) so the mask maps cleanly onto the
# game's own frame-toggle encoding.
UP: Final = 1 << 0
RIGHT: Final = 1 << 1
DOWN: Final = 1 << 2
LEFT: Final = 1 << 3

# Canonical letter for each flag, and the order they are emitted in.
_FLAG_TO_LETTER: Final = ((UP, "w"), (LEFT, "a"), (DOWN, "s"), (RIGHT, "d"))
_LETTER_TO_FLAG: Final = {"w": UP, "a": LEFT, "s": DOWN, "d": RIGHT}


class Action(IntEnum):
    """The 9 discrete actions a policy chooses between, one per physics frame."""

    COAST = 0
    COAST_LEFT = 1
    COAST_RIGHT = 2
    THROTTLE = 3
    THROTTLE_LEFT = 4
    THROTTLE_RIGHT = 5
    BRAKE = 6
    BRAKE_LEFT = 7
    BRAKE_RIGHT = 8


#: Action index -> raw input bitmask.
ACTION_TO_MASK: Final = (
    0,
    LEFT,
    RIGHT,
    UP,
    UP | LEFT,
    UP | RIGHT,
    DOWN,
    DOWN | LEFT,
    DOWN | RIGHT,
)

#: Inverse lookup, built once. Masks with both L+R or both U+D never appear.
MASK_TO_ACTION: Final = {mask: idx for idx, mask in enumerate(ACTION_TO_MASK)}

N_ACTIONS: Final = len(ACTION_TO_MASK)


def mask_to_keys(mask: int) -> str:
    """Render an input bitmask as .tas key letters, e.g. ``UP | RIGHT -> "wd"``.

    Letters are emitted in a fixed order so that the same mask always produces
    the same string -- byte-identical output matters when diffing TAS files.
    """
    return "".join(letter for flag, letter in _FLAG_TO_LETTER if mask & flag)


def keys_to_mask(keys: str) -> int:
    """Parse .tas key letters into an input bitmask. Case- and order-insensitive.

    Raises:
        ValueError: if an unrecognised character is present. We fail loudly
            rather than silently dropping input, because a silently mangled TAS
            file produces a run that desyncs thousands of frames later with no
            obvious cause.
    """
    mask = 0
    for char in keys.strip().lower():
        try:
            mask |= _LETTER_TO_FLAG[char]
        except KeyError:
            raise ValueError(
                f"unknown input key {char!r} in {keys!r}; expected only w/a/s/d"
            ) from None
    return mask


def canonicalize(mask: int) -> int:
    """Collapse a raw mask onto one of the 9 canonical actions' masks.

    Opposing inputs cancel, matching the game's own behaviour. Useful when
    importing human recordings, where both left and right can legitimately be
    held during the same frame.
    """
    if mask & LEFT and mask & RIGHT:
        mask &= ~(LEFT | RIGHT)
    if mask & UP and mask & DOWN:
        mask &= ~(UP | DOWN)
    return mask


def mask_to_action(mask: int) -> int:
    """Map any raw input mask to a discrete action index, cancelling opposites."""
    return MASK_TO_ACTION[canonicalize(mask)]
