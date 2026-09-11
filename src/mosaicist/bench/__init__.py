"""Benchmark statistics (device-time capture lives in the GPU workers)."""

from .stats import MIN_NOISE, Timing, noise_floor, summarize, verdict

__all__ = ["MIN_NOISE", "Timing", "noise_floor", "summarize", "verdict"]
