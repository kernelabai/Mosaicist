"""Mosaicist: converge a Pallas Mosaic GPU kernel on a CuTeDSL reference.

See DESIGN.md for the full design. Implemented so far (GPU-independent):

- ``mosaicist.ptx``      PTX parsing, CFG/loops, fingerprints (L0-L5), alignment, diff
- ``mosaicist.diagnose`` discrepancy rows -> ranked, tagged fixes
- ``mosaicist.verify``   numerics gate (oracle-relative ULP quantiles, bitwise signal)
- ``mosaicist.bench``    timing statistics, noise floor
- ``mosaicist.converge`` acceptance rule and candidate beam
"""

__version__ = "0.0.1"
