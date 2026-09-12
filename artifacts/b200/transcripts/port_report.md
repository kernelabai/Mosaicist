# Port of ref_flashinfer_gemm

- reference: `ref_flashinfer_gemm` on sm_100a, 13.4 us
- tile {'m': 512, 'n': 2048}, mma tcgen05 (mxf4nvf4)
- deferred from the reference at v0: nothing

Best candidate 18.4 us (1.38x reference), fingerprint distance 0.316, setting `{"block_k": 512, "collective": false, "stages": 1, "warp_split": false}`.

```
reference 13.4 us, noise floor 1.1%

   0 accept      28.6 us  D=0.345  numerics ok  {"block_k": 128, "collective": false, "stages": 1, "warp_split": false}
          next: [P1 rewrite] 1 warpgroup per CTA; reference runs 2
   1 accept      21.7 us  D=0.343  numerics ok  {"block_k": 256, "collective": false, "stages": 1, "warp_split": false}
          next: [P1 rewrite] 1 warpgroup per CTA; reference runs 2
   2 accept      18.4 us  D=0.316  numerics ok  {"block_k": 512, "collective": false, "stages": 1, "warp_split": false}
          next: [P1 rewrite] 1 warpgroup per CTA; reference runs 2
   3   --        26.9 us  D=0.316  numerics ok  {"block_k": 512, "collective": false, "stages": 2, "warp_split": false}
          next: [P1 rewrite] 1 warpgroup per CTA; reference runs 2
   4  ----   capture failed: Command '['/home/ubuntu/venv-jax/bin/python', '-m', 'mosaicist.cli', 'capture', 'cand_pallas_gemm', '--compiler', 'palla  {"block_k": 512, "collective": false, "stages": 3, "warp_split": false}
   5  ----   capture failed: Command '['/home/ubuntu/venv-jax/bin/python', '-m', 'mosaicist.cli', 'capture', 'cand_pallas_gemm', '--compiler', 'palla  {"block_k": 512, "collective": false, "stages": 4, "warp_split": false}

best 18.4 us (1.38x reference), D=0.316, setting {"block_k": 512, "collective": false, "stages": 1, "warp_split": false}

structural steps taken:
  - structural: collective -> True, named by collective
did not reach the reference within its noise floor

remaining differences with no knob to turn:
  - P1 rewrite: 1 warpgroup per CTA; reference runs 2
  - P2 investigate: TMA box rank differs
  - P5 investigate: Producer loop op order differs
  - P5 investigate: Mainloop op order differs
  - P5 rewrite: stmatrix use differs
```

## Gaps

Differences with no knob left to turn. These are the candidates for
an upstream report:

- P1 rewrite: 1 warpgroup per CTA; reference runs 2
- P2 investigate: TMA box rank differs
- P5 investigate: Producer loop op order differs
- P5 investigate: Mainloop op order differs
- P5 rewrite: stmatrix use differs
