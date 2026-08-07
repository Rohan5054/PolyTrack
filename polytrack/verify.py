"""The acceptance gate: replay a known recording, land on its exact finish frame.

This is the test the whole project turns on. Everything below it -- the decoder,
the geometry, the backend, save/restore -- can be individually correct and still
add up to a simulator that is not PolyTrack's. Nothing else distinguishes those
two cases.

It is deliberately not a test of our own devising. The game implements this
check itself, in ``simulation_worker.bundle.js``::

    for (; !car.hasFinished && car.frames < targetFrames; ) {
        const buffer = update(car, recording.getControls(car.frames));
        car.hasFinished = hasFinished(buffer);
        car.frames++;
    }
    result = car.hasFinished && car.frames == targetFrames;

:func:`verify` mirrors that loop. Two properties of it matter and are easy to
lose by "improving" it:

* **The stopping condition is exact.** Finishing one millisecond early is a
  failure, not a rounding difference. A simulator that is 0.01% off will finish
  a 40-second run four frames out, and that is precisely the class of error this
  is here to catch.
* **The loop stops at ``targetFrames`` regardless.** A run that never finishes
  fails rather than hanging.

Not passing this means the backend is wrong, however plausible its output.
Passing it means a run measured here is a run the game agrees with.
"""

from __future__ import annotations

from dataclasses import dataclass

from .physics.wasm import WasmPhysics
from .recording import Recording


@dataclass(frozen=True, slots=True)
class VerifyResult:
    """The outcome, with enough detail to say *how* a failure failed."""

    passed: bool
    #: The frame the run was supposed to finish on.
    target_frames: int
    #: Where the simulation actually got to.
    frames: int
    finished: bool
    #: Checkpoints reached, against how many the track has.
    checkpoints: int
    expected_checkpoints: int
    #: The module's own finish frame, when it finished at all.
    finish_frames: int | None

    @property
    def drift(self) -> int:
        """Frames -- milliseconds -- between us and the real run."""
        return self.frames - self.target_frames

    def describe(self) -> str:
        if self.passed:
            return (
                f"PASS  finished on frame {self.frames} "
                f"({self.frames / 1000:.3f} s), exactly as recorded"
            )
        if not self.finished:
            return (
                f"FAIL  did not finish: reached frame {self.frames} "
                f"({self.frames / 1000:.3f} s) and passed "
                f"{self.checkpoints}/{self.expected_checkpoints} checkpoints, "
                f"target was {self.target_frames}"
            )
        return (
            f"FAIL  finished on frame {self.frames}, expected "
            f"{self.target_frames} -- {self.drift:+d} ms "
            f"({abs(self.drift) / max(self.target_frames, 1) * 100:.4f}% off)"
        )


def verify(
    backend: WasmPhysics,
    recording: Recording,
    target_frames: int,
    *,
    progress=None,
) -> VerifyResult:
    """Replay ``recording`` and check it finishes on frame ``target_frames``.

    Args:
        backend: A backend already built on the track the recording was set on.
            It is stepped from wherever it currently is, so pass a fresh one or
            call ``reset()`` first.
        recording: The input sequence.
        target_frames: The finish frame the run is claimed to have. From the
            leaderboard entry, not from the recording -- recordings carry no
            length.
        progress: Optional callback, invoked as ``(frames, target_frames)``.

    Inputs are replayed as runs of held keys rather than tick by tick. That is
    not an optimisation of the game's loop so much as a restatement of it: the
    recording *is* a list of changes, and between two changes every tick takes
    the same mask.
    """
    if target_frames <= 0:
        raise ValueError(f"target_frames must be positive, got {target_frames}")

    expected_checkpoints = sum(
        1 for part in backend.track.parts if part.checkpoint_order is not None
    )
    result = None
    for mask, ticks in recording.runs(target_frames):
        remaining = target_frames - backend.frame
        if remaining <= 0:
            break
        result = backend.step_many(mask, min(ticks, remaining))
        if progress is not None:
            progress(backend.frame, target_frames)
        if result.finished:
            break

    record = backend.record
    finished = bool(record and record.has_finished)
    return VerifyResult(
        passed=finished and backend.frame == target_frames,
        target_frames=target_frames,
        frames=backend.frame,
        finished=finished,
        checkpoints=record.next_checkpoint_index if record else 0,
        expected_checkpoints=expected_checkpoints,
        finish_frames=record.finish_frames if record else None,
    )
