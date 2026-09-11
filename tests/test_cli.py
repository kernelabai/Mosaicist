import json

import numpy as np

from mosaicist.bundle import Bundle, Launch
from mosaicist.cli import main


def test_diff_command(fixtures, capsys):
    rc = main(["diff", str(fixtures / "hopper_gemm_ref.ptx"), str(fixtures / "hopper_gemm_v0.ptx"),
               "--ref-log", str(fixtures / "hopper_gemm_ref.ptxas.log"),
               "--cand-log", str(fixtures / "hopper_gemm_v0.ptxas.log")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "fingerprint distance D =" in out
    assert "1. P1 skeleton" in out and "num_compute_wgs=2" in out


def test_diff_json_and_fingerprint(fixtures, capsys):
    main(["diff", str(fixtures / "hopper_gemm_ref.ptx"), str(fixtures / "hopper_gemm_v0.ptx"), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert {"distance", "discrepancies", "fixes", "loop_pairs"} <= set(data)
    main(["fingerprint", str(fixtures / "hopper_gemm_ref.ptx")])
    out = capsys.readouterr().out
    assert "kernel_gemm_ws" in out and "role=mma" in out


def test_bundle_roundtrip_feeds_launch_into_l0(fixtures, tmp_path):
    (tmp_path / "k.ptx").write_text((fixtures / "hopper_gemm_ref.ptx").read_text())
    b = Bundle(kind="reference", name="gemm", arch="sm_90a", ptx="k.ptx",
               launch=Launch(grid=[132, 1, 1], block=[384, 1, 1], cluster=[2, 1, 1], dynamic_smem=196608))
    b.save(tmp_path)
    loaded = Bundle.load(tmp_path)
    assert loaded.launch.grid == [132, 1, 1]
    sk = loaded.fingerprint().skeleton
    assert sk["grid"] == [132, 1, 1] and sk["dynamic_smem"] == 196608


def test_check_command(tmp_path, capsys):
    oracle = np.linspace(-4, 4, 257)
    ref = oracle.astype(np.float16).astype(np.float64)
    for name, arr in (("ref", ref), ("cand", ref), ("oracle", oracle)):
        np.save(tmp_path / f"{name}.npy", arr)
    rc = main(["check", str(tmp_path / "ref.npy"), str(tmp_path / "cand.npy"), str(tmp_path / "oracle.npy"),
               "--fmt", "f16"])
    assert rc == 0 and capsys.readouterr().out.startswith("PASS")
