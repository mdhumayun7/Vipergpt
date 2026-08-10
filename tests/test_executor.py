"""Executor + ImagePatch tests using the mock ModuleBus (no GPU / weights)."""
import numpy as np

from vipergpt_repro.pipeline.executor import execute_program
from vipergpt_repro.pipeline.vision_bus import ModuleBus


def _img():
    return np.zeros((3, 32, 48), dtype="float32")


def test_simple_program_runs():
    bus = ModuleBus(mock=True)
    prog = (
        "def execute_command(image):\n"
        "    image_patch = ImagePatch(image)\n"
        "    return image_patch.simple_query('What is this?')\n"
    )
    res = execute_program(prog, _img(), bus)
    assert res.error is None
    assert res.result == "mock-answer"


def test_find_and_count_program():
    bus = ModuleBus(mock=True)
    prog = (
        "def execute_command(image):\n"
        "    image_patch = ImagePatch(image)\n"
        "    return len(image_patch.find('foo'))\n"
    )
    res = execute_program(prog, _img(), bus)
    assert res.error is None
    assert res.result == 1  # mock find returns one center box


def test_missing_execute_command_is_reported():
    bus = ModuleBus(mock=True)
    res = execute_program("x = 1\n", _img(), bus)
    assert res.error is not None


def test_program_exception_is_captured_not_raised():
    bus = ModuleBus(mock=True)
    prog = "def execute_command(image):\n    return 1/0\n"
    res = execute_program(prog, _img(), bus)
    assert res.result is None
    assert "ZeroDivisionError" in res.error


def test_filesystem_access_blocked():
    bus = ModuleBus(mock=True)
    prog = "def execute_command(image):\n    return open('/etc/passwd').read()\n"
    res = execute_program(prog, _img(), bus)
    # open is not in the safe builtins -> NameError captured
    assert res.result is None
    assert res.error is not None
