# Design — Beating Segment-WM on Tracing Time *and* Tracing Accuracy at 16 bits

Target: at a matched 16-bit payload and matched identity count (≥1000), Hi-DyPa should trace both
**faster** and **more accurately** than the Segment-WM / Reed–Solomon baseline in
[reedsolomon.py](../src/reedsolomon.py).

Both wins come from the same root cause: **splitting L into short segments**. Short segments allow
(a) bit-level distance instead of symbol-level, and (b) complete precomputed decode tables. Neither
is available to a flat 16-bit RS code.

All numbers below were computed, not asserted.

---

## 1. The matched baseline — and why RS is cornered at 16 bits

Your `ReedSolomon` requires `0 < k < n <= 2^m - 1`, and the embedded codeword must fit the payload
budget `n·m = 16`. Enumerating every legal parameter set:

| m | n | k | nsym | t | identities | correction |
|---|---|---|---|---|---|---|
| 4 | 4 | 1 | 3 | 1 | 16 | 1 symbol |
| 4 | 4 | 2 | 2 | 1 | 256 | 1 symbol |
| **4** | **4** | **3** | **1** | **0** | **4096** | **none** |
| 8 | 2 | 1 | 1 | 0 | 256 | none |

You have 1000 users. **The only parameter set reaching ≥1000 identities is RS(4,3) over GF(2⁴),
which has `t = 0` — it corrects nothing.** It can detect a corrupted symbol but not repair one.

Two consequences follow immediately, and they are the entire basis of this design:

1. **Accuracy:** at this operating point Segment-WM has zero correction capability, while option C
   corrects one bit flip and up to three erasures in the container segment.
2. **Time:** with `t = 0` the algebraic decoder is useless, so Segment-WM must fall back to
   nearest-codeword ML search — exactly what `ReedSolomonCodebook` implements — which is **O(N)**.

This is not a strawman baseline. It is what your own code is forced into by the payload budget, and
the `ReedSolomonCodebook` docstring already states that the exhaustive decoder was chosen precisely
because it is *stronger* than the syndrome decoder.

**Matched configuration for all experiments:**

```python
# Segment-WM baseline
rs   = ReedSolomon(n=4, k=3, m=4)              # 16-bit codeword, 4096 payloads
book = ReedSolomonCodebook(rs, num_payloads=1000)   # restricted to your 1000 users

# Hi-DyPa option C
spec = HierarchySpec(levels=(LevelSpec("container", 8, 16, 4),
                             LevelSpec("user",      8, 64, 2)))   # 1024 capacity
```

Same 16 bits embedded, same 1000 identities, same channel. Nothing is tilted.

---

## 2. Accuracy design

### 2.1 The channel is bit-wise and erasure-dominant — RS is built for the opposite

`LBitLogitProcessor` assigns bit positions to high-entropy blocks through a **random permutation**,
so bit errors and erasures are independent and uniformly spread across the L positions. They are not
bursty.

RS groups bits into 4-bit symbols. A symbol is destroyed if *any one* of its 4 bits is erased, so an
independent per-bit erasure rate `p` becomes a per-symbol rate `q = 1 − (1−p)⁴`:

| per-bit `p` | per-symbol `q` | amplification |
|---|---|---|
| 0.02 | 0.078 | ×3.9 |
| 0.05 | 0.185 | ×3.7 |
| 0.10 | 0.344 | ×3.4 |
| 0.20 | 0.590 | ×3.0 |

**RS pays roughly a 3–4× erasure penalty purely from symbol grouping**, and then has only
`nsym = 1` symbol of redundancy to absorb it. Symbol-oriented coding is the right tool for burst
channels; this channel is the opposite.

### 2.2 Predicted failure rates (analytic — to be validated empirically)

Under independent per-bit erasure rate `p`:

| `p` | Segment RS(4,3) fails | Option C — container lost | Option C — user ambiguous |
|---|---|---|---|
| 0.02 | 3.3 % | **0.00 %** | 1.0 % |
| 0.05 | 15.9 % | **0.04 %** | 5.7 % |
| 0.10 | 42.6 % | **0.50 %** | 18.7 % |
| 0.15 | 65.4 % | **2.14 %** | 34.3 % |
| 0.20 | 81.0 % | **5.63 %** | 49.7 % |
| 0.30 | 95.5 % | **19.41 %** | 74.5 % |

Two things to note when writing this up:

- The **container** column is the fair comparison against "RS fails" — both mean *the tracer has
  lost the identity entirely*. Option C is 1–2 orders of magnitude better across the whole range.
- The **user ambiguous** column is not a failure in the same sense: the container is still correct,
  so you have narrowed 1000 users to ~64 and can report a containment set. Segment-WM has no
  equivalent partial answer — it returns a wrong payload or nothing.

These are model predictions from the analytic channel, and they must be labelled as such until the
measured `p` from calibration replaces them.

### 2.3 Accuracy tier 2 — soft-decision rescoring

`LBitWatermarker.detect` currently discards a lot of information: it collapses `(z_i0, z_i1)` into a
hard symbol in `{0, 1, ⊥, *}`. Keeping the magnitudes gives a per-bit reliability

```
llr_i = z_i1 − z_i0            # sign = decided bit, magnitude = confidence
```

and turns Hamming distance into weighted distance `Σ |llr_i| · [bit_i ≠ codeword_i]`.

This is applied **only to candidates that survive the hard-decision stage**, so it costs almost
nothing (typically <100 candidates) and it resolves exactly the ties that the `⊥`/`*` symbols
create. It composes with the hierarchy: accumulate weighted distance per level as before.

Give the same treatment to the RS baseline where it can accept it, so the comparison stays fair —
`ReedSolomonCodebook.decode` can rank by weighted distance just as easily.

---

## 3. Tracing-time design

Three tiers, each strictly faster than the last. Implement all three; report all three.

### Tier 1 — bitwise integers instead of strings (constant-factor)

The current decode compares codewords **character by character** on Python strings
([watermark.py:1217](../src/watermark.py#L1217)). Replace with integer masks:

```python
# per level, precomputed once
codeword_int[i]                      # int, `bits` wide
# per trace
value_mask   : int   # recovered bits, erased positions forced to 0
erasure_mask : int   # 1 where the symbol was ⊥ or *

distance = ((value_mask ^ codeword_int[i]) & ~erasure_mask).bit_count()
```

Two integer ops and a popcount per candidate, versus 8 character comparisons. `int.bit_count()` is
available on Python 3.10+.

### Tier 2 — hierarchical pruning (sublinear in N)

Already the design: 16 layer-1 candidates, then 64 within each surviving container. **80 popcounts**
versus 1000 codeword comparisons for a flat scan.

### Tier 3 — complete precomputed decode tables (constant in N) ← the headline

Because each layer is only 8 bits, the **entire received-word space is enumerable**. Index a table
by `(value_mask, erasure_mask)`, both 8-bit:

```
table size = 2^8 x 2^8 = 65,536 entries per layer
```

Each entry stores the answer for that received segment:

```python
@dataclass
class LevelDecodeEntry:
    best_distance: int
    candidate_mask: int      # bitmask over the level's fanout (16 or 64 bits) of tied children
    n_ties: int
```

Decoding a trace becomes:

```python
e1 = LAYER1_TABLE[(v1 << 8) | m1]        # one array lookup  -> container candidate mask
e2 = LAYER2_TABLE[(v2 << 8) | m2]        # one array lookup  -> user index candidate mask
# combine: for each container in e1.candidate_mask, intersect e2.candidate_mask with occupied children
```

**Two array lookups plus a bitmask intersection — O(1), independent of N.**

Memory: 65,536 entries × 2 layers ≈ 1 MB. Build cost: 65,536 × 16 + 65,536 × 64 ≈ 5.2 M distance
computations, a few seconds in NumPy, computed once at load and cacheable to disk.

### Why Segment-WM cannot do this

The same trick applied to a flat 16-bit code needs

```
2^16 x 2^16 = 4,294,967,296 entries
```

which is infeasible. **Table decoding is only possible because the hierarchy splits L into short
independent segments.** That is the argument, and it is why this is a property of the scheme rather
than an implementation trick you could hand to the baseline.

---

## 4. Combined comparison

Decode-stage work per trace, N = 1000, 16-bit payload:

| Decoder | Work per trace | Scales with N? | Corrects |
|---|---|---|---|
| Segment RS syndrome | ~30 GF ops | no | **nothing** (`t=0`) |
| Segment RS ML search (`ReedSolomonCodebook`) | **4,000** symbol compares | **linear** | nearest-codeword only |
| Hi-DyPa, current string compare | 640 char compares | sublinear | 1 bit @ L1 |
| Hi-DyPa, tier 1 bitwise | 80 popcounts | sublinear | 1 bit @ L1 |
| **Hi-DyPa, tier 3 tables** | **2 array lookups** | **constant** | 1 bit @ L1 |

So the claim is: **~2000× less decode work, and 1–2 orders of magnitude lower identity-loss rate, at
identical payload and identical capacity.**

---

## 5. Two-tier decoder (the thing to actually build)

```python
def trace(recovered_hard, llr=None):
    e1 = LAYER1_TABLE[index(recovered_hard[0:8])]      # O(1)
    e2 = LAYER2_TABLE[index(recovered_hard[8:16])]     # O(1)

    candidates = combine(e1, e2, occupied_children)     # usually exactly 1
    if len(candidates) == 1:
        return candidates[0]                            # fast path

    if llr is not None:                                 # soft rescoring, only on ties
        return argmin_weighted_distance(candidates, llr)
    return candidates                                   # ambiguous -> containment_path
```

O(1) in the common case, accurate in the hard case, and it always has a meaningful partial answer
(`containment_path`) to fall back on. The RS baseline has none of these three properties.

---

## 6. Measurement protocol (must be fair, or the result is worthless)

**Report all three RS decoders**, not just the slow one:

1. `ReedSolomon.decode` (syndrome) — note that at `t=0` it corrects nothing, so its speed is
   irrelevant to accuracy. Report it anyway so no reviewer thinks it was hidden.
2. `ReedSolomonCodebook.decode` (ML) — the only usable RS decoder at this operating point.
3. Hi-DyPa tiers 1 / 2 / 3.

**Separate the stages.** End-to-end tracing is dominated by the `2L` zero-bit detection passes over
the token sequence, which are identical for both schemes and take milliseconds to seconds. Report:

- `decode_time_us` — the codeword→identity stage only, where the difference lives
- `detect_time_ms` — shared, reported once for context
- the ratio, so a reader can see decode is a small slice of end-to-end

Do **not** claim an end-to-end speedup. Claim a decode-stage speedup and say plainly that detection
dominates wall clock today. The decode result matters because it is what scales with N.

**Sweep two axes:**

- `N ∈ {10², 10³, 10⁴}` — shows the linear-vs-constant divergence. (N>1024 needs a 3rd layer or
  larger fanouts; RS needs a larger payload, so note where the comparison stops being matched.)
- erasure rate `p ∈ {0.02 … 0.30}` — validates the §2.2 table, and shows the RS ML fallback rate.

**Timing hygiene:** `timeit` with warm-up, ≥10⁴ repetitions, table build excluded and reported
separately as an amortised one-off, same machine, no GPU involvement (this stage is pure CPU).

**Use the offline channel model.** Decode accuracy and speed can both be measured against
synthetically corrupted codewords at a chosen `p`, with `p` calibrated from a modest number of real
generations. That gives 10⁵ trials per cell instead of 300, with no GPU time.

---

## 7. Honest caveats

1. **Detection, not decoding, dominates end-to-end time.** State it explicitly; claim the decode
   stage only.
2. **RS looks bad here because of the 16-bit budget, not because RS is a bad code.** Given a longer
   payload, RS(15,11) over GF(2⁴) would correct 2 symbols comfortably. Say so — the finding is
   "symbol-level MDS coding is a poor fit at short watermark payloads with independent bit erasures",
   which is a sharper and more defensible claim than "RS is worse".
3. **Table decoding scales only while segments stay short.** At 8 bits/level it is 64 K entries; at
   12 bits/level it is 16.7 M; at 16 it is infeasible. This is a property of *hierarchical* codes,
   and worth stating as such.
4. **The `d=2` layer-2 code still corrects nothing.** The accuracy win is concentrated at layer 1.
   Be precise: Hi-DyPa protects *container* identity strongly and *user* identity weakly.
5. **Soft-decision helps both schemes.** Apply it to the baseline too, or the comparison is unfair.

---

## 8. Tests

```python
def test_table_matches_bruteforce():
    # for all 65,536 (value, erasure) pairs, table entry == exhaustive nearest-codeword
    for v in range(256):
        for m in range(256):
            assert LAYER1_TABLE[(v << 8) | m] == brute_force_layer1(v, m)

def test_table_decode_matches_tier1():
    # tier 3 must be observationally identical to tier 1/2, only faster
    for _ in range(10_000):
        recovered = random_corrupted_codeword()
        assert decode_tables(recovered) == decode_bitwise(recovered)

def test_rs_baseline_t_is_zero():
    rs = ReedSolomon(n=4, k=3, m=4)
    assert rs.t == 0 and rs.nsym == 1          # documents why ML search is required

def test_matched_capacity():
    assert OPTION_C.capacity() == 1024
    assert 2 ** (3 * 4) == 4096                # RS(4,3) headroom, restricted to 1000

def test_soft_rescoring_only_on_ties():
    # fast path must not invoke the LLR path when the table returns a unique answer
```

---

## 9. Build order

| Step | Deliverable | Why first |
|---|---|---|
| 1 | Channel calibration (`calibrate_lbits.py`) | gives the real `p`; also decides L=16 vs L=12 |
| 2 | Tier 1 bitwise decode + option C codebooks | correctness baseline, no tables yet |
| 3 | Tier 3 tables + `test_table_matches_bruteforce` | the O(1) claim, provably identical to tier 1 |
| 4 | RS baseline harness at matched parameters | the comparison |
| 5 | Offline Monte-Carlo over `p` and `N` | the two headline figures |
| 6 | Soft-decision tier, applied to **both** schemes | the accuracy ceiling |

Steps 2–6 need no GPU. Only step 1 does.
