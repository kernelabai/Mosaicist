"""The knob space a candidate exposes, and how a setting reaches the capture process.

A candidate module declares `KNOBS` and reads its setting with `from_env()`::

    from mosaicist.converge.knobs import from_env

    KNOBS = {"stages": [1, 2, 4], "block_k": [64, 128, 256]}

    def build():
        k = from_env(defaults={"stages": 2, "block_k": 128})
        ...

Captures run in their own process, so a setting travels as JSON in the environment
rather than as an argument. That also means a candidate is reproducible from its bundle:
the setting is recorded alongside it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterator

ENV_VAR = "MOSAICIST_KNOBS"

Setting = dict[str, Any]


def from_env(defaults: Setting | None = None) -> Setting:
    """The setting for this capture, merged over `defaults`."""
    out = dict(defaults or {})
    raw = os.environ.get(ENV_VAR)
    if raw:
        out.update(json.loads(raw))
    return out


def to_env(setting: Setting) -> dict[str, str]:
    return {ENV_VAR: json.dumps(setting, sort_keys=True)}


def key(setting: Setting) -> str:
    return json.dumps(setting, sort_keys=True)


@dataclass
class KnobSpace:
    """Named knobs and the values worth trying, in preference order."""

    knobs: dict[str, list[Any]] = field(default_factory=dict)

    @classmethod
    def from_module(cls, module) -> "KnobSpace":
        return cls(knobs=dict(getattr(module, "KNOBS", {}) or {}))

    def default(self) -> Setting:
        return {k: v[0] for k, v in self.knobs.items() if v}

    def neighbours(self, setting: Setting, only: list[str] | None = None) -> Iterator[Setting]:
        """Settings one knob away from `setting`.

        One knob at a time is the whole point: every accepted step has to be
        attributable to a specific change, or a speedup cannot be explained.
        """
        for name, values in self.knobs.items():
            if only is not None and name not in only:
                continue
            for v in values:
                if setting.get(name) != v:
                    n = dict(setting)
                    n[name] = v
                    yield n

    def clamp(self, setting: Setting) -> Setting:
        """Drop knobs this space does not declare, so a stale setting stays usable."""
        return {k: v for k, v in setting.items() if k in self.knobs}
