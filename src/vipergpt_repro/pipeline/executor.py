"""Program execution engine φ.

Takes a generated program string defining `execute_command(...)`, compiles it in a
restricted namespace exposing only the ViperGPT API, and runs it on an input.

Safety: ViperGPT executes LLM-generated code. We compile in a namespace that does
NOT expose file/network/os builtins. This is defence-in-depth, not a full sandbox;
on a cluster, still run inside a container/unprivileged account. `execute_code`
in config gates whether batch mode runs programs at all.
"""
from __future__ import annotations

import logging
import signal
from dataclasses import dataclass
from typing import Any

from vipergpt_repro.pipeline import image_patch as ip
from vipergpt_repro.pipeline.video_segment import VideoSegment

logger = logging.getLogger(__name__)

# Builtins the generated program is allowed to use.
_SAFE_BUILTINS = {
    "len": len, "range": range, "min": min, "max": max, "sum": sum, "abs": abs,
    "sorted": sorted, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "list": list, "dict": dict, "set": set, "tuple": tuple, "str": str, "int": int,
    "float": float, "bool": bool, "round": round, "any": any, "all": all,
    "reversed": reversed, "print": print,
}


class ProgramError(Exception):
    pass


class _Timeout(Exception):
    pass


def _timeout_handler(signum, frame):  # pragma: no cover - signal path
    raise _Timeout()


@dataclass
class ExecResult:
    result: Any
    error: str | None = None
    program: str = ""


def build_namespace(bus) -> dict:
    """API namespace exposed to generated programs."""
    import math

    def ImagePatchFactory(image, *a, **kw):
        return ip.ImagePatch(image, *a, bus=bus, **kw)

    def VideoSegmentFactory(video, *a, **kw):
        return VideoSegment(video, *a, bus=bus, **kw)

    return {
        "__builtins__": _SAFE_BUILTINS,
        "math": math,
        "ImagePatch": ImagePatchFactory,
        "VideoSegment": VideoSegmentFactory,
        "best_image_match": ip.best_image_match,
        "distance": ip.distance,
        "bool_to_yesno": ip.bool_to_yesno,
        "coerce_to_numeric": ip.coerce_to_numeric,
        "List": list,
    }


def execute_program(program: str, image, bus, timeout_s: int = 120, **kwargs) -> ExecResult:
    """Compile and run a program. Returns ExecResult (never raises for program errors)."""
    ns = build_namespace(bus)
    use_alarm = hasattr(signal, "SIGALRM")
    try:
        compiled = compile(program, "<generated_program>", "exec")
        exec(compiled, ns)  # noqa: S102 - intentional; restricted namespace
        if "execute_command" not in ns:
            return ExecResult(None, "program did not define execute_command", program)

        if use_alarm:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout_s)
        try:
            result = ns["execute_command"](image, **kwargs)
        finally:
            if use_alarm:
                signal.alarm(0)
        return ExecResult(result, None, program)
    except _Timeout:
        return ExecResult(None, f"timeout after {timeout_s}s", program)
    except Exception as e:  # program errors are data, not crashes
        logger.debug("Program execution error: %s", e)
        return ExecResult(None, f"{type(e).__name__}: {e}", program)
