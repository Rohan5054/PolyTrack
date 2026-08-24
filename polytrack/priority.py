"""Keep long-running work off the efficiency cores.

Windows tags a process it considers background work -- one launched detached,
from a background shell, a scheduled task, a CI runner -- with background
quality of service, "EcoQoS". On a hybrid CPU that confines it to the
efficiency cores, and on a 12th-gen part those are roughly half the throughput
of a performance core at a lower clock.

This is worth a module because the symptom does not look like what it is. A
training run so tagged showed every python process pegging the four E-cores at
100% while the eight P-core threads sat at 2-24%, total CPU at 29%, clocks
boosting normally and no thermal or power cap in sight. Read as "the machine is
mostly idle, so the learner must be blocked on something", which is what it
looks like, it sends you hunting for a lock. The measured cost here was 3x:
achieved replay ratio 0.11 against 0.33, and 38 transitions/second against 64.

``SetProcessInformation`` with ``ProcessPowerThrottling`` is the documented way
to opt out, and it is inherited by child processes -- so the learner calling it
covers the collectors it spawns. Raising the priority class also happens to lift
the tag, but it is a side effect of a different mechanism and it makes the rest
of the machine sluggish, which is a poor trade on a laptop somebody is using.
"""

from __future__ import annotations

import sys

#: PROCESS_INFORMATION_CLASS::ProcessPowerThrottling.
_PROCESS_POWER_THROTTLING = 4
_PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
#: The only flag defined for this structure: CPU speed.
_EXECUTION_SPEED = 0x1


def lift_background_qos() -> bool:
    """Opt this process out of Windows' efficiency-core confinement.

    Returns whether the request was made and accepted. False on any other
    platform, and on Windows versions predating the API -- neither is an
    error, so this never raises: it is an optimisation, and a run that cannot
    have it should still start.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        class _State(ctypes.Structure):
            _fields_ = [
                ("Version", wintypes.ULONG),
                ("ControlMask", wintypes.ULONG),
                ("StateMask", wintypes.ULONG),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetProcessInformation.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD
        ]
        kernel32.SetProcessInformation.restype = wintypes.BOOL

        # ControlMask says which knob this call is addressing; StateMask says
        # what to set it to. Naming the knob and leaving its bit clear is
        # "manage execution speed explicitly, and do not throttle it" -- as
        # opposed to a zero ControlMask, which means "go back to letting the
        # system decide" and would leave the tag in place.
        state = _State(
            _PROCESS_POWER_THROTTLING_CURRENT_VERSION, _EXECUTION_SPEED, 0
        )
        return bool(kernel32.SetProcessInformation(
            kernel32.GetCurrentProcess(),
            _PROCESS_POWER_THROTTLING,
            ctypes.byref(state),
            ctypes.sizeof(state),
        ))
    except Exception:
        return False
