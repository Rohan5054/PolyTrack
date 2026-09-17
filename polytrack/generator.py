"""Procedural track generation.

The roadmap item this project needs most. Four hand-made tracks make a
memorisation problem: a policy can learn them without learning to drive. An
endless supply of tracks drawn from one distribution turns the same setup into a
generalisation problem, and PolyTrack gives that away cheaply, because a track
is a short string and the game will load any string we can write.

It also **retires the reference-line bootstrap**. A reference line is a route,
not a centreline, and until now the only way to know a track's route was to
replay a lap somebody had driven on it -- which is why ``tracks.npz`` exists and
why a new track needed a human before it could be trained on. A generated track
knows its own route by construction: the walk that lays the pieces down *is* the
route, in order, before a single frame is simulated.

## The grid

Every constant here was measured rather than guessed. ``PART_SIZE`` is 5 and a
``Straight`` is 20 world units long, so a piece spans exactly **four grid
units** -- which is why every coordinate in a shipped track is a multiple of
four.

``TurnSharp``'s road surface is open on local +z and local +x and closed on
local -z. Where those local axes point in the world is the part that went
wrong the first time, and it is worth recording how.

``start_transform`` composes the part rotation with a half turn about Y, and
reading the axes off that transform says a part at rotation *r* points local +z
along ``_HEADINGS[r]``. That is true of the **car**, which the half turn exists
to face down the road -- and false of the part. The module places parts with the
rotation alone, so a part's local +z actually lands on ``_HEADINGS[r + 2]`` and
its local +x on ``_HEADINGS[r + 3]``. A turn at rotation *r* therefore joins
``_HEADINGS[r + 2]`` and ``_HEADINGS[r + 3]``.

Nothing straight could reveal this: a straight is symmetric under a half turn,
so straight-only tracks drove to the finish under the wrong model. What settled
it was one corner built at all four rotations and driven -- exactly the two the
corrected model predicts got through, and the two the original model chose went
nowhere. ``tests/test_generator.py`` drives generated tracks with corners on the
real module so this cannot regress quietly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .physics.world import PART_SIZE
from .trackcode import Environment, Part, TrackCode

#: The direction a car spawned on a start at rotation r faces, read off the
#: client's own start transform. This is **not** where the part's own local +z
#: points -- that is two entries further on; see the module docstring.
_HEADINGS: tuple[tuple[int, int, int], ...] = (
    (0, 0, -1),
    (-1, 0, 0),
    (0, 0, 1),
    (1, 0, 0),
)

#: Grid units per piece. ``PART_SIZE * STEP`` is the 20-unit length of a Straight.
STEP = 4

STRAIGHT, TURN, START, FINISH, CHECKPOINT = 0, 1, 5, 6, 52


def _rotation_for(heading: tuple[int, int, int]) -> int:
    return _HEADINGS.index(heading)


def _advance(cell: tuple[int, int, int], heading: tuple[int, int, int]):
    return tuple(cell[i] + STEP * heading[i] for i in range(3))


@dataclass(frozen=True, slots=True)
class GeneratedTrack:
    """A track together with the route through it, which is the point."""

    code: TrackCode
    #: Grid cells in the order they are driven, start first.
    cells: tuple[tuple[int, int, int], ...]

    @property
    def route(self) -> np.ndarray:
        """The route as world-space points, ready for a reference line.

        Cell centres, not a driven line. This is deliberately not a racing line
        -- it is the *route*, which is the thing a reference line has always
        actually needed and the thing that used to cost a human lap to obtain.
        """
        return np.array(self.cells, dtype=np.float64) * PART_SIZE

    def __len__(self) -> int:
        return len(self.cells)


def generate(
    seed: int = 0,
    *,
    pieces: int = 60,
    checkpoints: int = 3,
    turn_chance: float = 0.35,
    min_straight: int = 1,
    max_attempts: int = 200,
    environment: int = Environment.SUMMER,
    name: str | None = None,
) -> GeneratedTrack:
    """Lay out a track by walking a grid, and return it with its route.

    The walk refuses to reuse a cell, which is what keeps the result drivable:
    two pieces in one cell is not a crossover, it is two roads in the same
    space. It does not otherwise avoid running alongside itself, so generated
    tracks do pass close to themselves -- which is exactly the case
    ``ProgressTracker`` exists to handle, and a good thing to train against.

    ``min_straight`` is the fewest straights allowed between two corners. At
    zero the walk can place two right angles back to back, and when both turn
    the same way that is a 180-degree hairpin inside two 20 m pieces with the
    return leg in the adjacent lane -- a scripted driver at 11 m/s stopped dead
    in exactly that on the one generated track out of ten it failed. One
    straight spreads the same U-turn over three pieces.

    Raises:
        RuntimeError: if no walk of the requested length was found. The walk can
            trap itself in a dead end, so restarts are normal; a persistent
            failure means ``pieces`` is too large for ``turn_chance`` to route.
    """
    rng = random.Random(seed)

    walk = None
    for _ in range(max_attempts):
        walk = _walk(rng, pieces, turn_chance, min_straight)
        if walk is not None:
            break
    if walk is None:
        raise RuntimeError(
            f"no self-avoiding walk of {pieces} pieces after {max_attempts} "
            f"attempts at turn_chance={turn_chance}"
        )
    cells, rotations, is_turn = walk

    # Checkpoints go on straights, spread through the middle. The first and last
    # pieces are the start and the finish, and a checkpoint laid over a turn is
    # legal but drives badly.
    interior = [i for i in range(2, len(cells) - 2) if not is_turn[i]]
    chosen: list[int] = []
    for k in range(checkpoints):
        if not interior:
            break
        position = min(
            int((k + 1) * len(interior) / (checkpoints + 1)), len(interior) - 1
        )
        if interior[position] not in chosen:
            chosen.append(interior[position])

    parts: list[Part] = []
    for index, (cell, rotation) in enumerate(zip(cells, rotations)):
        if index == 0:
            part_id, checkpoint_order, start_order = START, None, 0
        elif index == len(cells) - 1:
            part_id, checkpoint_order, start_order = FINISH, None, None
        elif index in chosen:
            part_id = CHECKPOINT
            checkpoint_order, start_order = chosen.index(index), None
        else:
            part_id = TURN if is_turn[index] else STRAIGHT
            checkpoint_order, start_order = None, None
        parts.append(
            Part(
                x=cell[0], y=cell[1], z=cell[2], part_id=part_id,
                rotation=rotation, rotation_axis=0, color=0,
                checkpoint_order=checkpoint_order, start_order=start_order,
            )
        )

    code = TrackCode(
        parts=tuple(parts),
        environment=environment,
        sun_direction=90,
        name=name or f"Generated {seed}",
        author="polytrack-agent",
        last_modified=None,
        payload=b"",
    )
    return GeneratedTrack(code=code, cells=tuple(cells))


def _walk(rng: random.Random, pieces: int, turn_chance: float, min_straight: int):
    """One self-avoiding walk. Returns ``(cells, rotations, is_turn)`` or None."""
    heading = _HEADINGS[rng.randrange(4)]
    cell = (0, 0, 0)
    cells = [cell]
    rotations = [_rotation_for(heading)]
    is_turn = [False]
    used = {cell}
    last_turn = -(10 ** 9)

    while len(cells) < pieces:
        options: list[tuple[tuple[int, int, int], int, bool]] = []

        forward = _advance(cell, heading)
        if forward not in used:
            options.append((forward, _rotation_for(heading), False))

        # Turning happens in the cell the car is *in*, so the piece already
        # placed there becomes the turn. The first piece is the start, whose
        # geometry is a straight -- turning there would place start geometry
        # where the walk assumed a corner and break the road at piece one.
        here = len(cells) - 1
        if here > 0 and here - last_turn - 1 >= min_straight:
            back = tuple(-component for component in heading)
            index = _HEADINGS.index(back)
            # A turn at rotation r joins _HEADINGS[r + 2] and _HEADINGS[r + 3]
            # (see the module docstring for why it is not r and r + 1). The car
            # enters through the face pointing back the way it came, so one of
            # those two must be `back`; each way of satisfying that is one of
            # the two turns, left and right.
            #   _HEADINGS[r + 2] == back  ->  r = index - 2, exit on r + 3
            #   _HEADINGS[r + 3] == back  ->  r = index + 1, exit on r + 2
            for rotation, exit_heading in (
                ((index - 2) % 4, _HEADINGS[(index + 1) % 4]),
                ((index + 1) % 4, _HEADINGS[(index - 1) % 4]),
            ):
                target = _advance(cell, exit_heading)
                if target not in used:
                    options.append((target, rotation, True))

        if not options:
            return None

        straights = [option for option in options if not option[2]]
        turns = [option for option in options if option[2]]
        if turns and (not straights or rng.random() < turn_chance):
            target, rotation, turned = rng.choice(turns)
        else:
            target, rotation, turned = rng.choice(straights)

        if turned:
            rotations[-1] = rotation
            is_turn[-1] = True
            last_turn = here
            heading = tuple(
                (target[i] - cell[i]) // STEP for i in range(3)
            )

        cells.append(target)
        rotations.append(_rotation_for(heading))
        is_turn.append(False)
        used.add(target)
        cell = target

    return cells, rotations, is_turn
