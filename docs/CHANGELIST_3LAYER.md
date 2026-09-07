# Change List — 3-Layer Hi-DyPa (8+4+4) implementation

Everything below is **additive**. No existing file was modified, so the naive /
grouped / hi_dypa / segment schemes and all results already collected behave
exactly as before.

---

## 1. Files added

| File | Lines | Purpose |
|---|---|---|
| `src/hierarchy.py` | ~560 | Spec, codebooks, mixed-radix index, coarse-to-fine decode, precomputed decode tables. **stdlib only** — no torch, no pandas |
| `src/hierarchical_watermark.py` | ~190 | `HierarchicalMultiUserWatermarker`, subclass of the existing `NaiveMultiUserWatermarker` |
| `tests/test_hierarchy.py` | ~300 | 25 tests, CPU-only, no model. Runs under pytest **or** standalone |
| `evaluation_scripts/benchmark_tracing_time.py` | ~430 | The tracing-time / tracing-accuracy benchmark |
| `slurm_scripts/run_tracing_time_benchmark_hpc.sh` | ~140 | OzSTAR job (CPU partition, no GPU) |
| `config/hierarchies/l16_8_4_4.json` | — | **The recommended config**: 8+4+4, d=(4,2,2), fanouts (16,8,8), capacity 1024 |
| `config/hierarchies/l16_8_8_optionC.json` | — | 2-layer 8+8, d=(4,2) — comparison |
| `config/hierarchies/l16_8_8_legacy.json` | — | 2-layer 8+8, d=(2,1) — reproduces today's Hi-DyPa layout |
| `config/hierarchies/l16_flat.json` | — | 1 level of 16 bits — the naive layout |
| `config/hierarchies/l12_6_3_3.json` | — | L=12 fallback if calibration rejects L=16 |

## 2. Files modified

**None.** `src/watermark.py`, `src/fingerprinting.py`, `src/segment_watermark.py`,
`src/reedsolomon.py` and every evaluation script are untouched.

---

## 3. Settings chosen (and why)

| Setting | Value | Reason |
|---|---|---|
| Split | **8 + 4 + 4** | Exhaustive search over all splits of 16 across 3 levels and all distance assignments up to 6, scored on `P(exact identification)` |
| Distances | **(4, 2, 2)** | `d=4` at the top is the max achievable for 16 codewords in 8 bits (`A2(8,4)=16`); `d=2` is the max for 8 codewords in 4 bits |
| Fanouts | **(16, 8, 8)** | Capacity 1024 ≥ 1000 users, all 16 containers occupied |
| Layer-1 code | extended Hamming `[8,4,4]` | Explicit committed table, not greedy search — removes the risk of greedy under-delivering |
| Layer-2/3 code | even parity `[4,3,2]` | Same `(i << 1) \| parity(i)` rule as the existing `d=2` group codes |
| Bit order | **MSB-first** everywhere | Matches the existing Hi-DyPa lazy `d=2` path (which all your current results used) |
| `beam_width` | `0` (keep all ties) | Reproduces the current two-stage trace when depth = 2 |
| `margin` | `0` | Same |
| `use_tables` | `True` | O(1) decode; falls back to a bitwise scan for partially-filled nodes so results are identical |
| Erasure symbols | `{⊥, *, ?}` | Single shared constant — the existing code disagreed between `('⊥','*')` and `('⊥','*','?')` |
| Assignment | `sequential` | `row -> divmod` mixed radix; reduces exactly to `group_id = row // users_per_group` at depth 2 |
| Capacity overflow | warn + truncate | Same behaviour as the existing `load_users` |
| Over-requested fanout | **raises** | The existing code only warned and silently shrank capacity |

---

## 4. Corrections to the earlier design documents

Three things I stated earlier turned out to be wrong once measured or once the
real Segment-WM code was available. All three matter.

**(a) Segment-WM's RS is not `t = 0`.**
I claimed that at 16 bits RS is cornered into `RS(4,3)` with zero correction. That
assumed the RS codeword must fit in 16 channel slots. Your `segment_watermark.py`
sets `DEFAULT_RS_PARAMS[16] = (6, 4, 4)`: **6 segments × 4 bits = 24 channel
symbols carrying a 16-bit payload**, so `nsym = 2` and `t = 1` — it corrects one
symbol error. The correct framing is that Segment-WM spends 50 % more channel
than Hi-DyPa's 16 bit-positions and still trades worse; not that its code is
degenerate. `DESIGN_TRACING_VS_SEGMENT.md` §1 overstates this and should be read
with this correction.

**(b) Depth improves accuracy here; my earlier prediction was backwards.**
`MULTI_LEVEL_EXTENSION.md` §8 predicted depth costs accuracy at fixed `L`. The
search shows the opposite whenever capacity is in surplus, because each extra
level adds a parity check. Corrected statement is in `DESIGN_3LAYER.md` §2.

**(c) A fully-erased top layer fans out to 15 containers, not 16.**
Container 15 holds only 40 of 1000 users, so it has 5 teams rather than 8. When
the whole layer-1 segment is erased, container 15 is correctly eliminated at
layer 2 because the observed team codeword does not exist under it. This is the
ragged-tree path working, and it is now pinned by
`test_full_erasure_of_top_level_fans_out`.

---

## 5. Verification

`python tests/test_hierarchy.py` → **25/25 pass**, ~2 s, no GPU, no model.

Covers: spec validation, codebook distances and capacities, the extended Hamming
table, the legacy even-parity rule, path/row round-trips, ragged children,
distance profile (4 across containers / 2 within), clean decode of all 1000
users, single-flip correction at layer 1, 3-erasure correction at layer 1,
collusion containment at both levels, table-vs-scan equivalence over 4000 random
corruptions, and table entries vs brute force.

---

## 6. Measured results (local run, N = 1000, L = 16)

`evaluation/tracing_time/results_local.json`, 1000 trials × 3 runs, median.

| Scheme | µs/trace @ p=0 | µs/trace @ p=0.10 | exact id @ p=0.10 |
|---|---|---|---|
| `naive` (flat 16-bit) | 429.7 | 412.9 | 38.0 % |
| `hi_dypa_2layer` (8+8, d=2,1) | 47.9 | 51.9 | 45.2 % |
| `hier_2layer_optC` (8+8, d=4,2) | 18.3 | 23.3 | 84.7 % |
| `hier_3layer_scan` (8+4+4) | 12.5 | 17.2 | **90.4 %** |
| **`hier_3layer_tables` (8+4+4)** | **8.1** | **9.9** | **90.4 %** |
| `segment_rs_ml` | 904.0 | 287.1 | 8.3 % |
| `segment_rs_synd` | 780.4 | 288.3 | 34.5 % |

Speedup of `hier_3layer_tables` at N = 1000:

| | p=0.00 | p=0.05 | p=0.10 | p=0.20 |
|---|---|---|---|---|
| vs `naive` | ×53.1 | ×43.4 | ×41.7 | ×42.4 |
| vs `hi_dypa_2layer` | ×5.9 | ×5.1 | ×5.2 | ×6.8 |
| vs `hier_2layer_optC` | ×2.3 | ×2.5 | ×2.4 | ×2.3 |
| vs `segment_rs_ml` | **×111.7** | ×41.9 | ×29.0 | ×26.6 |
| vs `segment_rs_synd` | ×96.4 | ×54.3 | ×29.1 | ×6.6 |

Both goals met: the 3-layer scheme is the fastest tracer **and** the most
accurate, against every baseline at every erasure rate tested.

### Caveats to carry into the write-up

1. **This is the tracing stage only.** Detection (the `2L` zero-bit passes) is
   excluded; it dominates end-to-end wall clock and is identical across the
   L-bit schemes. Claim a decode-stage speedup, not an end-to-end one.
2. **The accuracy columns use a synthetic channel**, not a measured one. L-bit
   schemes get independent per-bit erasures; Segment-WM gets the matched
   per-symbol rate `1 − (1−p)⁴`. These are different channels, so the accuracy
   comparison across scheme families is indicative until calibration supplies
   the real per-bit `p` and the real segment error profile.
3. **`segment_rs_synd` gets faster as `p` grows** (780 → 53 µs) because the
   syndrome decoder bails out early more often — it is fast precisely when it
   is failing. Read its time column together with its accuracy column.
4. **Pure-Python timings.** Absolute µs are interpreter-bound; the ratios are
   the meaningful quantity. All schemes are measured in the same interpreter.
5. **L=16 is still unvalidated on a real model.** `z ≈ 54/L` puts it near 3.4
   against a 4.0 threshold. Calibration remains the gate on the whole config.

---

## 7. Not implemented (deliberately)

- `helper_scripts/calibrate_lbits.py` — needs a GPU run; specified in
  `DESIGN_3LAYER.md` §8 but not written.
- Soft-decision (LLR) rescoring tier.
- `main_multiuser.py --scheme hierarchical` CLI wiring.
- Migration of the evaluation scripts onto the shared decoder (Phase 0 of
  `MULTI_LEVEL_EXTENSION.md`); the 5 duplicated `decode_hi_dypa_user` copies are
  still there.


---

## 8. Added for the GPU run (A100 / GPT-2)

| File | Purpose |
|---|---|
| `evaluation_scripts/evaluate_hierarchical_gpt2.py` | End-to-end: embed -> generate -> detect -> trace, on the **real** channel. `--mode calibrate` sweeps L x max_new_tokens; `--mode full` compares all schemes |
| `slurm_scripts/run_hierarchical_gpt2_a100_hpc.sh` | A100 job (`milan-gpu`, `--gres=gpu:1`, 12 h): tests -> calibration -> L=16 comparison -> L=12 fallback |

### GPT-2's context is the binding constraint

GPT-2 has `n_positions = 1024`, so prompt + generation cannot exceed 1024 tokens
and the block count is capped near 960. Since per-bit `z ~ sqrt(blocks) / L`,
that ceiling -- not the hierarchy -- decides whether L=16 is usable. The scripts
clamp `max_new_tokens` to `1024 - 64` on GPT-2 automatically.

This is why step 1 of the A100 job is calibration and step 3 runs the L=12
fallback unconditionally: whichever way L=16 lands, there is a reportable
configuration at the end of the job.

### Bug found and fixed while validating

Printing a recovered codeword crashes with `UnicodeEncodeError` on any stdout
that is not UTF-8, because U+22A5 is emitted by existing `print()` calls in
`src/watermark.py` (line 1139 among others). On a container without a UTF-8
locale this would have killed the job partway through. Both evaluation scripts
now force UTF-8 on stdout/stderr at startup, and both SLURM scripts export
`PYTHONIOENCODING=utf-8` and `LC_ALL=C.UTF-8`. `src/watermark.py` itself is
still untouched.

### Measured: the shipped 2-layer tracer is the slow one

An offline dry run (stubbed model, 200 trials, 10 % erasure) timed the tracing
call of each scheme as it currently ships:

| Scheme | µs/trace | exact id | per-level prefix accuracy | containment |
|---|---|---|---|---|
| `hier:l16_8_4_4` (ours) | **15.9** | **89.0 %** | 100 % / 94 % / 89 % | 5.4 users |
| `hier:l16_8_8_optionC` | 32.1 | 85.5 % | 100 % / 86 % | 10.1 users |
| `naive` (as shipped) | 3,464 | 38.5 % | — | — |
| `hi_dypa_2layer` (as shipped) | **109,746** | 41.5 % | — | — |

The shipped `HiDyPaMultiUserWatermarker.trace_from_codeword` costs **110 ms per
trace** -- about 6,900x our decoder -- because `get_codeword_for_user` runs an
`O(N)` pandas scan (`watermark.py:1032`) once per candidate inside the tracing
loop (`watermark.py:1214`). The synthetic CPU benchmark's `hi_dypa_2layer` row
(~48 µs) measures the *algorithm* on the same codeword layout; this row measures
the *implementation*. Report whichever you mean, and say which.
