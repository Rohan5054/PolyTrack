"""The real backend: PolyTrack's own physics module, hosted from Python.

This is the piece the whole project is gated on. It satisfies the same
:class:`~polytrack.physics.base.PhysicsBackend` protocol as the stand-in, so
swapping it in changes nothing above this file.

**Everything here is transcribed from the game's own JavaScript, not inferred.**
``lib/polytrack_physics.js`` is the Emscripten glue and gives the exact export
letters and import list; ``simulation_worker.bundle.js`` gives the call sequence
and the buffer layouts. An earlier attempt to find the API by stepping the
module and diffing memory produced confident nonsense -- a determinism PASS on a
call sequence that did nothing, and "car position" candidates that were integer
counters reinterpreted as denormal floats. Read the glue; do not probe.

The module needs no Node host. All nine imports are ordinary Emscripten runtime
hooks and only ``emscripten_resize_heap`` needs a real implementation, so
wasmtime in-process is enough: one process, no IPC, and parallelism comes from
running many instances.

The call sequence, once per instance::

    k()                                          __wasm_call_ctors
    n(massOffset, hullVerts, nFloats)            initializeCarCollisionShape
    o(id, verts, n, detector..., startOffset...) addTrackPartConfiguration, per part type
    p(carId, mountain..., trackData, nParts, startPos, startQuat)   createCarModel
    r(carId, up, right, down, left, reset, out)  updateCarModel, per frame

``updateCarModel`` takes no timestep, because the timestep is fixed inside the
module -- and it is **one millisecond, not one 60th of a second**. Three
independent confirmations: the worker's realtime loop consumes its accumulator
in ``o -= .001`` increments; ``maxFrames`` is 5999999, which is the game's
99:59.999 cap counted in milliseconds; and ``formatTimeString`` renders
``numberOfFrames`` by dividing by 60000 and 1000. So a "frame" throughout
PolyTrack -- in the record, in recordings, in every published time -- is a
millisecond, and this backend steps at 1000 Hz.

That has a consequence for anything above this file: 1000 decisions a second is
not an action space. The game itself does not work that way either -- it records
inputs as *changes*, so a held key spans hundreds of ticks. Use
:meth:`WasmPhysics.step_many` to hold one action across a control interval; it
is both the right semantics and far faster than stepping tick by tick from
Python.

Two things the module does *not* give you, and how they are handled:

* **No velocity or angular velocity.** The state record carries ``speedKmh`` and
  an orientation quaternion, nothing more. Both vectors are finite-differenced
  here at 60 Hz. They are observation-only -- nothing derived feeds back into
  the simulation -- so this cannot affect a replay. The raw record travels
  intact in ``StepResult.info`` for anything that wants the authoritative
  numbers.
* **No save-state call.** :meth:`WasmPhysics.save_state` snapshots the whole of
  linear memory plus the stack pointer. That is ~17 MB per call and will
  dominate beam search; it is deliberately correct-before-fast, because a save
  that misses hidden state desyncs only under search, which is far worse than
  being slow.
"""

from __future__ import annotations

import ctypes
import math
import os
import pickle
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from wasmtime import Engine, Func, Instance, Memory, Module, Store, Trap, WasmtimeError
from wasmtime import _ffi as _wasmtime_ffi
from wasmtime._func import maybe_raise_last_exn as _maybe_raise_last_exn

from ..actions import DOWN, LEFT, RIGHT, UP
from ..trackcode import TrackCode, decode
from .assets import GeometryAssets
from .base import CarState, StepResult
from .world import (
    angular_velocity,
    forward_vector,
    mountain_vertices,
    start_transform,
    up_vector,
)

#: Physics ticks per second, fixed inside the module. One tick is one
#: millisecond and one unit of every time PolyTrack reports. See the module
#: docstring -- this is not 60.
RATE = 1000.0

#: The in-game respawn key. Not part of the 9-action space -- a policy should
#: not learn to spam it -- but recordings encode it, so the mask carries it.
RESET = 1 << 4

#: The maximum extent of the car state record: ``i32 carId`` plus 223 bytes.
STATE_BUFFER_SIZE = 227

_WASM_NAME = "polytrack_physics.wasm"
_PAGE = 65536
_MAX_HEAP = 2147483648


class WasmUnavailable(RuntimeError):
    """The module could not be found or would not instantiate."""


class ModuleAborted(RuntimeError):
    """The module called ``abort``, ``exit``, ``__assert_fail`` or threw.

    Raised rather than swallowed. Any of these means the module was driven
    wrongly, and continuing would produce a plausible-looking but meaningless
    trajectory.
    """


# ------------------------------------------------------------ state record --


@dataclass(frozen=True, slots=True)
class CarRecord:
    """The car state record ``updateCarModel`` writes, parsed.

    Variable length -- ``finishFrames`` is only present once the car has
    finished, and per-wheel contact data only for wheels actually touching
    something. Parse it forwards; do not index fixed offsets.
    """

    frames: int
    speed_kmh: float
    has_started: bool
    has_finished: bool
    has_checkpoint_to_respawn_at: bool
    wheel_contact: tuple[bool, bool, bool, bool]
    finish_frames: int | None
    next_checkpoint_index: int
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]
    collision_impulses: tuple[float, ...]
    #: ``(point, normal)`` per wheel in contact, ``None`` for wheels in the air.
    wheel_contact_points: tuple[tuple[tuple[float, ...], tuple[float, ...]] | None, ...]
    wheel_suspension_length: tuple[float, ...]
    wheel_suspension_velocity: tuple[float, ...]
    wheel_delta_rotation: tuple[float, ...]
    wheel_skid_info: tuple[float, ...]
    steering: float
    brake_light_enabled: bool
    controls: int
    n_bytes: int


def parse_car_record(buffer: bytes) -> CarRecord:
    """Decode the record, mirroring the client's ``CarState`` deserialiser.

    Raises:
        ValueError: on anything inconsistent. The impulse count in particular is
            a good tripwire -- it is bounded by 4, so a large value means the
            cursor has drifted and everything after it is noise.
    """
    offset = 0

    def u8() -> int:
        nonlocal offset
        value = buffer[offset]
        offset += 1
        return value

    def u24() -> int:
        nonlocal offset
        value = int.from_bytes(buffer[offset : offset + 3], "little")
        offset += 3
        return value

    def u16() -> int:
        nonlocal offset
        value = int.from_bytes(buffer[offset : offset + 2], "little")
        offset += 2
        return value

    def f32(count: int) -> tuple[float, ...]:
        nonlocal offset
        value = struct.unpack_from(f"<{count}f", buffer, offset)
        offset += 4 * count
        return value

    frames = u24()
    speed_kmh = f32(1)[0]

    flags = u8()
    has_started = bool(flags & 1)
    has_finished = bool(flags & 2)
    has_respawn = bool(flags & 4)
    contact = (
        bool(flags & 8), bool(flags & 16), bool(flags & 32), bool(flags & 64)
    )

    finish_frames = u24() if has_finished else None
    next_checkpoint = u16()
    position = f32(3)
    quaternion = f32(4)

    n_impulses = u8()
    if n_impulses > 4:
        raise ValueError(
            f"collision impulse count {n_impulses} exceeds the maximum of 4 at "
            f"byte {offset - 1}: the record cursor has drifted"
        )
    impulses = f32(n_impulses)

    points: list[tuple[tuple[float, ...], tuple[float, ...]] | None] = []
    for wheel in range(4):
        if contact[wheel]:
            values = f32(6)
            points.append((values[:3], values[3:]))
        else:
            points.append(None)

    suspension_length = f32(4)
    suspension_velocity = f32(4)
    delta_rotation = f32(4)
    skid = f32(4)
    steering = f32(1)[0]
    controls = u8()

    return CarRecord(
        frames=frames,
        speed_kmh=speed_kmh,
        has_started=has_started,
        has_finished=has_finished,
        has_checkpoint_to_respawn_at=has_respawn,
        wheel_contact=contact,
        finish_frames=finish_frames,
        next_checkpoint_index=next_checkpoint,
        position=position,
        quaternion=quaternion,
        collision_impulses=impulses,
        wheel_contact_points=tuple(points),
        wheel_suspension_length=suspension_length,
        wheel_suspension_velocity=suspension_velocity,
        wheel_delta_rotation=delta_rotation,
        wheel_skid_info=skid,
        steering=steering,
        brake_light_enabled=bool(controls & 32),
        controls=controls & 31,
        n_bytes=offset,
    )


def _read_pose(
    buffer: bytes,
) -> tuple[tuple[float, ...], tuple[float, ...], bool, int]:
    """Pull just pose, finish flag and checkpoint index out of a raw buffer.

    A full :func:`parse_car_record` per tick is wasted work when a step holds an
    input for hundreds of them. Everything here sits at a fixed offset, save for
    the three bytes ``finishFrames`` inserts once the car has finished.
    """
    finished = bool(buffer[11] & 2)
    at = 15 if finished else 12  # skip carId, frames, speedKmh, flags[, finish]
    checkpoint = int.from_bytes(buffer[at : at + 2], "little")
    values = struct.unpack_from("<7f", buffer, at + 2)
    return values[:3], values[3:], finished, checkpoint


def has_finished(buffer: bytes) -> bool:
    """Read the finish flag without parsing the record.

    Byte 11 bit 1 of the whole buffer, ``i32 carId`` prefix included. This is
    exactly what the game's own replay verifier reads, and it is what the
    acceptance gate compares against.
    """
    return bool(buffer[11] & 2)


# ----------------------------------------------------------------- hosting --


class _FastCall:
    """A wasmtime export bound for repeated calling.

    ``wasmtime-py``'s ordinary call path boxes every argument into a ``Val``
    object: measured here at ~100 us for a seven-argument call, against ~4 us
    through the C API's unchecked entry point. At 1000 physics ticks per
    simulated second that difference is most of the budget, so the hot path goes
    through the raw ABI.

    "Unchecked" means the signature is not validated per call, so it is read
    once from the module and the argument slots are typed from it -- passing the
    wrong shape here would be undefined behaviour rather than a Python error.
    Traps and host-side exceptions are surfaced exactly as the wrapper does.
    """

    __slots__ = ("_context", "_func", "_pointer", "_slots", "_setters", "_result", "_n")

    def __init__(self, store: Store, func: Func) -> None:
        signature = func.type(store)
        params = [str(p) for p in signature.params]
        results = [str(r) for r in signature.results]
        if len(results) > 1:
            raise WasmUnavailable("multi-value exports are not supported here")

        self._context = store._context()
        self._func = func  # keep alive; _pointer borrows from it
        self._pointer = ctypes.byref(func._func)
        self._n = max(len(params), len(results), 1)
        self._slots = (_wasmtime_ffi.wasmtime_val_raw_t * self._n)()
        self._setters = tuple(_slot_setter(kind) for kind in params)
        self._result = results[0] if results else None

    def __call__(self, *values):
        slots = self._slots
        for index, (setter, value) in enumerate(zip(self._setters, values)):
            setter(slots[index], value)

        trap = ctypes.POINTER(_wasmtime_ffi.wasm_trap_t)()
        error = _wasmtime_ffi.wasmtime_func_call_unchecked(
            self._context, self._pointer, slots, self._n, ctypes.byref(trap)
        )
        if error:
            # A host import raised; wasmtime stashed the real exception.
            _maybe_raise_last_exn()
            raise WasmtimeError._from_ptr(error)
        if trap:
            trap_object = Trap._from_ptr(trap)
            _maybe_raise_last_exn()
            raise trap_object

        if self._result is None:
            return None
        return getattr(slots[0], "f32" if self._result == "f32" else
                       "f64" if self._result == "f64" else
                       "i64" if self._result == "i64" else "i32")


def _slot_setter(kind: str):
    if kind == "i32":
        def set_i32(slot, value) -> None:
            slot.i32 = int(value)
        return set_i32
    if kind == "i64":
        def set_i64(slot, value) -> None:
            slot.i64 = int(value)
        return set_i64
    if kind == "f32":
        def set_f32(slot, value) -> None:
            slot.f32 = float(value)
        return set_f32
    if kind == "f64":
        def set_f64(slot, value) -> None:
            slot.f64 = float(value)
        return set_f64
    raise WasmUnavailable(f"unsupported parameter type {kind!r}")


class WasmModule:
    """One instance of ``polytrack_physics.wasm`` under wasmtime.

    The nine imports are named ``a.a`` through ``a.i`` and are satisfied here the
    way the Emscripten glue satisfies them. Only ``emscripten_resize_heap`` does
    real work; the rest either never fire or indicate that something has gone
    wrong, and those raise.
    """

    __slots__ = (
        "store", "instance", "memory", "base_address", "_exports", "_bound",
        "_data_ptr", "_data_len", "_stdout",
    )

    def __init__(self, path: str | Path | None = None, *, now_ms: float = 0.0) -> None:
        resolved = _resolve_wasm(path)
        self.store = Store(Engine())
        module = Module(self.store.engine, resolved.read_bytes())
        self._stdout: dict[int, bytearray] = {1: bytearray(), 2: bytearray()}
        self.memory: Memory | None = None
        self._data_ptr = None
        self._data_len = 0

        handlers = {
            "a": self._cxa_throw,
            "b": self._exit,
            "c": lambda *_: None,  # clear the Emscripten keepalive counter
            "d": lambda *_: 0,  # setTimeout; the timer never fires for us
            "e": self._abort,
            "f": self._resize_heap,
            "g": self._fd_write,
            "h": lambda *_: now_ms,  # Date.now, read once during __wasm_call_ctors
            "i": self._assert_fail,
        }
        externs = []
        for imported in module.imports:
            handler = handlers.get(imported.name or "")
            if handler is None:
                raise WasmUnavailable(
                    f"unexpected import {imported.module}.{imported.name}: this "
                    "module is not the 0.6.2 physics build"
                )
            externs.append(Func(self.store, imported.type, handler))

        self.instance = Instance(self.store, module, externs)
        self._exports = self.instance.exports(self.store)
        self._bound: dict[str, _FastCall] = {}
        self.memory = self._exports["j"]
        self._refresh_memory()
        self.call("k")  # __wasm_call_ctors

    # -- exports ------------------------------------------------------------

    def bind(self, name: str) -> _FastCall:
        """A callable for one export, bound once for repeated use."""
        bound = self._bound.get(name)
        if bound is None:
            bound = _FastCall(self.store, self._exports[name])
            self._bound[name] = bound
        return bound

    def call(self, name: str, *args):
        return self.bind(name)(*args)

    def malloc(self, size: int) -> int:
        pointer = self.call("l", size)
        if not pointer:
            raise MemoryError(f"module malloc({size}) returned null")
        return pointer

    def free(self, pointer: int) -> None:
        self.call("m", pointer)

    # -- memory -------------------------------------------------------------

    def _refresh_memory(self) -> None:
        assert self.memory is not None
        self._data_ptr = self.memory.data_ptr(self.store)
        self._data_len = self.memory.data_len(self.store)
        self.base_address = ctypes.cast(self._data_ptr, ctypes.c_void_p).value or 0

    def view(self) -> np.ndarray:
        """A zero-copy uint8 view of linear memory. Invalid after any growth."""
        return np.ctypeslib.as_array(
            ctypes.cast(self._data_ptr, ctypes.POINTER(ctypes.c_uint8)),
            shape=(self._data_len,),
        )

    def read(self, pointer: int, size: int) -> bytes:
        # string_at off a cached base address rather than a numpy slice: this is
        # on the per-tick path, and at 1000 ticks per simulated second every
        # allocation in here shows up in the throughput figure.
        return ctypes.string_at(self.base_address + pointer, size)

    def write(self, pointer: int, data: bytes) -> None:
        self.view()[pointer : pointer + len(data)] = np.frombuffer(data, dtype=np.uint8)

    def write_f32(self, pointer: int, values: np.ndarray) -> None:
        flat = np.ascontiguousarray(values, dtype=np.float32)
        self.view()[pointer : pointer + flat.nbytes] = flat.view(np.uint8)

    # -- imports ------------------------------------------------------------

    def _resize_heap(self, requested: int) -> int:
        """``emscripten_resize_heap``, including the glue's over-allocation.

        The over-allocation schedule is copied rather than simplified. Heap size
        does not affect any float the physics computes, but it does affect where
        the allocator puts things, and a physics engine is entitled to key its
        broadphase off allocation order.
        """
        requested &= 0xFFFFFFFF
        if requested > _MAX_HEAP:
            return 0
        old_size = self._data_len
        for cut_down in (1, 2, 4):
            over_grown = old_size * (1 + 0.2 / cut_down)
            over_grown = min(over_grown, requested + 100663296)
            target = min(_MAX_HEAP, _align_up(max(requested, over_grown), _PAGE))
            if self._grow_to(int(target)):
                return 1
        return 0

    def _grow_to(self, target: int) -> bool:
        assert self.memory is not None
        pages = (target - self._data_len + _PAGE - 1) // _PAGE
        if pages <= 0:
            return True
        try:
            self.memory.grow(self.store, pages)
        except Exception:
            return False
        self._refresh_memory()
        return True

    def _fd_write(self, fd: int, iov: int, iovcnt: int, written_ptr: int) -> int:
        view = self.view()
        total = 0
        sink = self._stdout.setdefault(fd, bytearray())
        for _ in range(iovcnt):
            base = int.from_bytes(view[iov : iov + 4].tobytes(), "little")
            length = int.from_bytes(view[iov + 4 : iov + 8].tobytes(), "little")
            iov += 8
            sink += view[base : base + length].tobytes()
            total += length
        view[written_ptr : written_ptr + 4] = np.frombuffer(
            total.to_bytes(4, "little"), dtype=np.uint8
        )
        return 0

    def messages(self, fd: int = 2) -> str:
        """Anything the module printed. Usually empty; useful after an abort."""
        return self._stdout.get(fd, bytearray()).decode("utf-8", "replace")

    def _abort(self, *_) -> None:
        raise ModuleAborted("module called abort(): " + self.messages())

    def _exit(self, code: int = 0) -> None:
        raise ModuleAborted(f"module called exit({code}): " + self.messages())

    def _cxa_throw(self, *_) -> None:
        raise ModuleAborted("module threw a C++ exception: " + self.messages())

    def _assert_fail(self, condition: int, file: int, line: int, function: int) -> None:
        raise ModuleAborted(
            f"assertion failed in the module at line {line}: " + self.messages()
        )


def _align_up(value: float, alignment: int) -> int:
    return math.ceil(value / alignment) * alignment


def _resolve_wasm(path: str | Path | None) -> Path:
    if path is not None:
        candidate = Path(path)
        if not candidate.exists():
            raise WasmUnavailable(f"no physics module at {candidate}")
        return candidate

    candidates = []
    from_env = os.environ.get("POLYTRACK_WASM")
    if from_env:
        candidates.append(Path(from_env))
    candidates.append(Path(__file__).resolve().parents[2] / _WASM_NAME)
    candidates.append(Path.cwd() / _WASM_NAME)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise WasmUnavailable(
        f"no {_WASM_NAME} found (looked in "
        + ", ".join(str(c) for c in candidates)
        + "). Extract it from a copy of the game you own -- it is Kodub's code, "
        "so keep it local and do not redistribute it."
    )


# ----------------------------------------------------------------- backend --


@dataclass(slots=True)
class _Snapshot:
    memory: bytes
    stack_pointer: int
    frame: int
    previous_position: tuple[float, float, float]
    previous_quaternion: tuple[float, float, float, float]
    record: CarRecord | None


class WasmPhysics:
    """PolyTrack's physics, driving one car on one track.

    Args:
        track: A decoded :class:`~polytrack.trackcode.TrackCode`, or a track code
            string to decode.
        assets: Collision geometry. Loaded from ``geometry.npz`` if omitted.
        wasm_path: The module. Searched for if omitted.
        car_id: Passed through to the module. Only matters if you ever drive
            more than one car per instance, which nothing here does.
        mountains: Whether to include the terrain ring the client generates.
            On by default because the client always sends it. Turning it off is
            for measuring its cost, not for normal use.
    """

    __slots__ = (
        "track", "assets", "module", "car_id", "_state_ptr", "_start_position",
        "_start_quaternion", "_mountain", "_mountain_offset", "_track_buffer",
        "_frame", "_record", "_previous_position", "_previous_quaternion",
        "_update",
    )

    def __init__(
        self,
        track: TrackCode | str,
        *,
        assets: GeometryAssets | None = None,
        wasm_path: str | Path | None = None,
        car_id: int = 0,
        mountains: bool = True,
    ) -> None:
        self.track = decode(track) if isinstance(track, str) else track
        self.assets = assets if assets is not None else GeometryAssets.load()
        self.car_id = car_id

        self._start_position, self._start_quaternion = start_transform(
            self.track, self.assets
        )
        if mountains:
            self._mountain, self._mountain_offset = mountain_vertices(self.track)
        else:
            self._mountain = np.zeros(0, dtype=np.float32)
            self._mountain_offset = (0.0, 0.0, 0.0)
        self._track_buffer = self.track.physics_buffer()

        self.module = WasmModule(wasm_path)
        self._update = self.module.bind("r")  # updateCarModel, the hot path
        self._state_ptr = self.module.malloc(STATE_BUFFER_SIZE)
        self._register_geometry()

        self._frame = 0
        self._record: CarRecord | None = None
        self._previous_position = self._start_position
        self._previous_quaternion = self._start_quaternion
        self._create_car()

    # -- setup --------------------------------------------------------------

    def _register_geometry(self) -> None:
        """``initializeCarCollisionShape`` then ``addTrackPartConfiguration``.

        Every part is registered, not just the ones this track places, and in
        the order the game's own configuration table declares them. The game
        does this once at worker start-up; matching it is free.
        """
        module = self.module
        hull = self.assets.car_collision_vertices
        pointer = module.malloc(hull.nbytes)
        module.write_f32(pointer, hull)
        module.call("n", self.assets.mass_offset, pointer, hull.size)
        module.free(pointer)

        for part in self.assets.parts:
            pointer = module.malloc(part.vertices.nbytes)
            module.write_f32(pointer, part.vertices)
            module.call(
                "o",
                part.id,
                pointer,
                part.vertices.size,
                part.detector_type,
                *part.detector_center,
                *part.detector_size,
                1 if part.start_offset is not None else 0,
                *(part.start_offset or (0.0, 0.0, 0.0)),
            )
            module.free(pointer)

    def _create_car(self) -> None:
        module = self.module
        mountain_ptr = module.malloc(max(self._mountain.nbytes, 1))
        if self._mountain.size:
            module.write_f32(mountain_ptr, self._mountain)
        track_ptr = module.malloc(max(len(self._track_buffer), 1))
        module.write(track_ptr, self._track_buffer)

        module.call(
            "p",
            self.car_id,
            mountain_ptr,
            int(self._mountain.size),
            *self._mountain_offset,
            track_ptr,
            len(self.track),
            *self._start_position,
            *self._start_quaternion,
        )
        module.free(mountain_ptr)
        module.free(track_ptr)

    # -- protocol -----------------------------------------------------------

    @property
    def rate(self) -> float:
        """Ticks per simulated second: 1000, fixed inside the module."""
        return RATE

    @property
    def frame(self) -> int:
        return self._frame

    @property
    def record(self) -> CarRecord | None:
        """The last raw record, or ``None`` before the first step."""
        return self._record

    def reset(self) -> CarState:
        """Respawn at the start and zero the frame counter.

        A full teardown -- ``deleteCarModel`` then ``createCarModel`` -- not the
        module's ``reset`` control bit. That bit is the in-game respawn key: it
        returns the car to its last checkpoint and does **not** rewind the frame
        counter, so using it here would silently keep the clock running.
        """
        self.module.call("q", self.car_id)
        self._create_car()
        self._frame = 0
        self._record = None
        self._previous_position = self._start_position
        self._previous_quaternion = self._start_quaternion
        return self._initial_state()

    def step(self, input_mask: int) -> StepResult:
        """Advance one physics tick -- **one millisecond**, not one 60th.

        This is the module's actual quantum and the unit every PolyTrack time is
        counted in, so it is what the protocol's ``step`` maps onto. For an
        agent, hold an action across an interval with :meth:`step_many` instead.

        The mask goes through unchanged. Opposing inputs are cancelled inside
        the module -- that is what the v0.6.0 changelog note is about -- so
        cancelling them here as well would be redundant, and would quietly
        diverge from a human recording that holds both.
        """
        return self.step_many(input_mask, 1)

    def step_many(self, input_mask: int, ticks: int) -> StepResult:
        """Hold one input for ``ticks`` milliseconds, stopping early on finish.

        This is the shape the game itself uses -- recordings store input
        *changes*, so a held key spans hundreds of ticks -- and it keeps the
        per-tick cost down to one call plus a seven-float read, with the full
        record parsed once at the end.

        Returns the result of the last tick actually taken. ``passed_checkpoint``
        covers the whole interval, not just the final tick.
        """
        if ticks < 1:
            raise ValueError(f"ticks must be at least 1, got {ticks}")

        up = 1 if input_mask & UP else 0
        right = 1 if input_mask & RIGHT else 0
        down = 1 if input_mask & DOWN else 0
        left = 1 if input_mask & LEFT else 0
        reset = 1 if input_mask & RESET else 0

        module = self.module
        update = self._update
        read = module.read
        car_id = self.car_id
        state_ptr = self._state_ptr
        checkpoint_before = (
            0 if self._record is None else self._record.next_checkpoint_index
        )
        highest_checkpoint = checkpoint_before
        buffer = b""

        for tick in range(ticks):
            update(car_id, up, right, down, left, reset, state_ptr)
            buffer = read(state_ptr, STATE_BUFFER_SIZE)
            self._frame += 1
            position, quaternion, finished, checkpoint = _read_pose(buffer)
            if checkpoint > highest_checkpoint:
                highest_checkpoint = checkpoint
            if finished or tick == ticks - 1:
                break
            # Only the second-to-last pose is needed, for the finite difference.
            self._previous_position = position
            self._previous_quaternion = quaternion

        # The record is prefixed with the car id it belongs to; skip it.
        record = parse_car_record(buffer[4:])
        state = self._to_car_state(record)
        self._record = record
        self._previous_position = record.position
        self._previous_quaternion = record.quaternion

        return StepResult(
            state=state,
            passed_checkpoint=highest_checkpoint > checkpoint_before,
            finished=record.has_finished,
            terminated=False,
            info={
                "record": record,
                "speed_kmh": record.speed_kmh,
                "module_frames": record.frames,
                "finish_frames": record.finish_frames,
                "has_started": record.has_started,
                "steering": record.steering,
            },
        )

    def save_state(self) -> bytes:
        """Snapshot linear memory and the stack pointer.

        The module exposes no save-state call, so this is the whole heap: ~17 MB
        a time. That will dominate search and is meant to be narrowed later, but
        only after the gate passes -- a save that misses hidden state produces a
        run that looks fine and desyncs only under search.
        """
        snapshot = _Snapshot(
            memory=self.module.view().tobytes(),
            stack_pointer=self.module.call("w"),  # stackSave
            frame=self._frame,
            previous_position=self._previous_position,
            previous_quaternion=self._previous_quaternion,
            record=self._record,
        )
        return pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL)

    def load_state(self, blob: bytes) -> CarState:
        snapshot: _Snapshot = pickle.loads(blob)
        current = self.module.view()
        if len(snapshot.memory) != current.size:
            # The heap grew (or shrank) since the snapshot. Growth is one-way in
            # WebAssembly, so a larger snapshot cannot be restored into a
            # smaller heap without growing first.
            if len(snapshot.memory) > current.size:
                self.module._grow_to(len(snapshot.memory))
                current = self.module.view()
            else:
                raise ValueError(
                    f"snapshot is {len(snapshot.memory)} bytes but the heap is "
                    f"now {current.size}; memory cannot shrink"
                )
        current[: len(snapshot.memory)] = np.frombuffer(
            snapshot.memory, dtype=np.uint8
        )
        self.module.call("u", snapshot.stack_pointer)  # stackRestore

        self._frame = snapshot.frame
        self._record = snapshot.record
        self._previous_position = snapshot.previous_position
        self._previous_quaternion = snapshot.previous_quaternion
        if snapshot.record is None:
            return self._initial_state()
        return self._to_car_state(snapshot.record)

    def test_determinism(self) -> bool:
        """Run the module's own determinism self-test.

        The client runs this and refuses to submit runs that fail it. It is a
        check on the host, not on us -- but a false here means nothing this
        backend produces is worth anything.
        """
        return bool(self.module.call("s"))

    def close(self) -> None:
        self.module.call("q", self.car_id)

    # -- state --------------------------------------------------------------

    def _initial_state(self) -> CarState:
        """The state before the first step, as the client constructs it.

        The module writes nothing until ``updateCarModel`` is called, so the
        client builds this itself from the start transform. Same here.
        """
        return CarState(
            frame=0,
            position=np.array(self._start_position, dtype=np.float64),
            velocity=np.zeros(3, dtype=np.float64),
            forward=forward_vector(self._start_quaternion),
            up=up_vector(self._start_quaternion),
            angular_velocity=np.zeros(3, dtype=np.float64),
            wheel_contact=np.zeros(4, dtype=bool),
            checkpoint=0,
            finished=False,
        )

    def _to_car_state(self, record: CarRecord) -> CarState:
        position = np.array(record.position, dtype=np.float64)
        velocity = (position - np.array(self._previous_position, dtype=np.float64)) * RATE
        return CarState(
            frame=self._frame,
            position=position,
            velocity=velocity,
            forward=forward_vector(record.quaternion),
            up=up_vector(record.quaternion),
            angular_velocity=angular_velocity(
                self._previous_quaternion, record.quaternion, RATE
            ),
            wheel_contact=np.array(record.wheel_contact, dtype=bool),
            checkpoint=record.next_checkpoint_index,
            finished=record.has_finished,
        )
