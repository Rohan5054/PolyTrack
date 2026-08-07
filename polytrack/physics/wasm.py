"""Skeleton for the real PolyTrack physics backend.

This is a template, not a working backend. It cannot be finished without the
actual module in front of you, because the function names, calling convention
and memory layout are only discoverable by inspection -- run
``tools/inspect_wasm.py`` first and fill in :class:`WasmLayout` from what it
reports.

Obtain the module yourself from a copy of the game you have legitimately, keep
it local, and do not redistribute it. What you are building is the PolyTrack
equivalent of TMInterface: the instrumentation layer that lets a program step
the simulation, read car state, and save/restore. In Trackmania that tool is
what made Linesight and every other serious agent possible.

Two hosting options, decided by the import count the inspector reports:

**Python host (few or no imports).** Load the module with wasmtime, satisfy any
imports with stubs, call the step function directly. Fastest and simplest --
one process, no IPC, trivially parallel across worker processes.

**Node host (many Emscripten imports).** The generated JS glue already
satisfies them. Run a small Node process that loads the module and speaks a
line- or binary-protocol over stdio, and drive it from Python. Batch many frames
per round trip; per-frame IPC will dominate your runtime otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import CarState, StepResult


@dataclass(slots=True)
class WasmLayout:
    """Where things live in the module. Fill this in from the inspector output.

    Function names are the exports you call. Offsets are byte positions in the
    module's linear memory where the car state is written after each step --
    find them by stepping once, dumping memory, stepping again, and diffing to
    see which floats move plausibly with the car.
    """

    # Exported function names.
    fn_init: str = "TODO_init"
    fn_reset: str = "TODO_reset"
    fn_step: str = "TODO_step"
    fn_load_track: str = "TODO_load_track"

    # Byte offsets into linear memory for car state, as float64 triples.
    offset_position: int = -1
    offset_velocity: int = -1
    offset_forward: int = -1
    offset_up: int = -1
    offset_angular_velocity: int = -1
    offset_wheel_contact: int = -1  # 4 bytes/flags
    offset_checkpoint: int = -1  # i32
    offset_finished: int = -1  # i32 / bool

    #: Byte ranges that constitute the full simulation state, for save/restore.
    #: Start with the whole heap; narrow it once you know what actually matters.
    state_ranges: tuple[tuple[int, int], ...] = field(default_factory=tuple)

    def validate(self) -> None:
        missing = [
            name
            for name, value in (
                ("offset_position", self.offset_position),
                ("offset_velocity", self.offset_velocity),
                ("offset_forward", self.offset_forward),
            )
            if value < 0
        ]
        if missing:
            raise NotImplementedError(
                "WasmLayout is incomplete: " + ", ".join(missing) + ". Run "
                "tools/inspect_wasm.py against the module and fill these in."
            )


class WasmPhysics:
    """Drives PolyTrack's compiled physics module.

    Satisfies the same :class:`~polytrack.physics.base.PhysicsBackend` protocol
    as the reference stand-in, so swapping it in changes nothing above this file.
    """

    def __init__(self, wasm_path: str, track_code: str, layout: WasmLayout) -> None:
        self.layout = layout
        self.layout.validate()
        self.wasm_path = wasm_path
        self.track_code = track_code
        self._frame = 0
        raise NotImplementedError(
            "Fill in WasmLayout and implement the load/step calls. "
            "See the module docstring and README section 'Stage 0'."
        )

    @property
    def frame(self) -> int:
        return self._frame

    def reset(self) -> CarState:
        raise NotImplementedError

    def step(self, input_mask: int) -> StepResult:
        """Advance one physics frame.

        Two things to get right, both of which cause silent desyncs:

        * Run the module **single-threaded**. PolyTrack's car simulation is
          multithreaded, and thread scheduling is the hardest case for
          deterministic replay.
        * Feed the input mask in the game's own bit order (up, right, down,
          left) -- see :mod:`polytrack.actions`, which already uses it.
        """
        raise NotImplementedError

    def save_state(self) -> bytes:
        """Snapshot the memory ranges listed in the layout.

        Be generous at first: copying the whole heap is slow but correct, and a
        correct-but-slow save is far better than a fast one that misses hidden
        state. Narrow it only after the round-trip test passes consistently.
        """
        raise NotImplementedError

    def load_state(self, blob: bytes) -> CarState:
        raise NotImplementedError

    def _read_state(self) -> CarState:
        """Read car state out of linear memory using the configured offsets."""
        raise NotImplementedError


def read_vec3(memory: memoryview, offset: int) -> np.ndarray:
    """Read three float64s from linear memory. Little-endian, as WASM mandates."""
    return np.frombuffer(memory, dtype="<f8", count=3, offset=offset).copy()
