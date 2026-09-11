"""Guard bands around output buffers to catch out-of-bounds writes.

Workers allocate each output inside a larger device buffer pre-filled with a
sentinel byte, hand the kernel a view of the middle, and after the run copy
the raw bytes back. A write that lands outside the view but leaves every
in-bounds value correct still shows up here.
"""

from __future__ import annotations

import numpy as np

SENTINEL = 0xA5


def layout(payload_nbytes: int, pad_nbytes: int = 4096, align: int = 256) -> tuple[int, int]:
    """(total_nbytes, payload_offset) for a guarded allocation."""
    offset = -(-pad_nbytes // align) * align
    return offset + payload_nbytes + pad_nbytes, offset


def fill(nbytes: int) -> np.ndarray:
    return np.full(nbytes, SENTINEL, dtype=np.uint8)


def violations(raw: np.ndarray, payload_offset: int, payload_nbytes: int, limit: int = 16) -> list[tuple[int, int]]:
    """(byte offset relative to the payload start, value) for corrupted guard bytes."""
    raw = np.asarray(raw, dtype=np.uint8).reshape(-1)
    guard = np.ones(raw.size, dtype=bool)
    guard[payload_offset : payload_offset + payload_nbytes] = False
    bad = np.flatnonzero(guard & (raw != SENTINEL))
    return [(int(i) - payload_offset, int(raw[i])) for i in bad[:limit]]
