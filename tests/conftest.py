from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES


@pytest.fixture
def ref_ptx() -> str:
    return (FIXTURES / "hopper_gemm_ref.ptx").read_text()


@pytest.fixture
def cand_ptx() -> str:
    return (FIXTURES / "hopper_gemm_v0.ptx").read_text()
