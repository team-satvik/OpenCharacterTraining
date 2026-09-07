"""
filesystem paths, read from the environment

upstream expects each user to write this file by hand (it is gitignored there),
so a fresh clone cannot import character.* at all. we track it instead.
"""

import os
from pathlib import Path


def _required(var: str) -> str:
    value = os.environ.get(var)
    if not value:
        raise RuntimeError(f"{var} is not set: point it at a directory before importing character.*")
    return value


MODEL_PATH = _required("OCT_MODEL_PATH")
DATA_PATH = _required("OCT_DATA_PATH")
LORA_PATH = _required("OCT_LORA_PATH")
# constitutions ship with the repo, so this one has a sensible default
CONSTITUTION_PATH = os.environ.get(
    "OCT_CONSTITUTION_PATH",
    str(Path(__file__).resolve().parent.parent / "constitutions"),
)
