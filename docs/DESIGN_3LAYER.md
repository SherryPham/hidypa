# Design — 3-Layer Hi-DyPa at 16 bits (recommended configuration)

Supersedes [DESIGN_OPTION_C.md](DESIGN_OPTION_C.md) as the recommended build target.
Goal: beat Segment-WM / Reed–Solomon on **both** tracing time and tracing accuracy at a matched
16-bit payload and matched identity count.

Every number below was computed and verified, not asserted.

---

## 1. Recommended configuration

| | Layer 1 `container` | Layer 2 `team` | Layer 3 `user` |
|---|---|---|---|
| bits | 8 | 4 | 4 |
| fanout | 16 | 8 | 8 |
| min distance | 4 | 2 | 2 |
| construction | extended Hamming `[8,4,4]` | even parity `[4,3,2]` | even parity `[4,3,2]` |
| corrects | 1 flip, or 3 erasures, or 1 flip + 1 erasure | 1 erasure | 1 erasure |

**Capacity = 16 × 8 × 8 = 1024** ≥ 1000 users ✓ · **L = 8 + 4 + 4 = 16** ✓

Selected by exhaustive search over every `(b₁,b₂,b₃)` summing to 16 and every `(d₁,d₂,d₃)` up to 6,
with fanouts set to what a lexicode construction can actually build (not theoretical `A2` bounds),
scored on `P(exact identification)` under an independent per-bit erasure channel. `(8,4,4)` with
`d=(4,2,2)` was the top-ranked 3-layer configuration at both `p=0.10` and `p=0.20`, and within 0.2
percentage points of the top at `p=0.05` while having strictly better container containment.

Verified codebooks:

```
Layer 1  extended Hamming [8,4,4] : 16 words, d = 4   (table in DESIGN_OPTION_C.md §2)
Layer 2  even parity      [4,3,2] :  8 words, d = 2
Layer 3  even parity      [4,3,2] :  8 words, d = 2
         ['0000','0011','0101','0110','1001','1010','1100','1111']
```

---

## 2. Why 3 layers beats 2 — and a correction to the earlier spec

[MULTI_LEVEL_EXTENSION.md](MULTI_LEVEL_EXTENSION.md) §8 predicted that depth *costs* accuracy at
fixed `L`. **That prediction was wrong in this regime, and the search shows why.**

Each layer carries its own parity check. Splitting one 8-bit `d=2` layer into two 4-bit `d=2` layers
spends **two** parity bits on those 8 positions instead of one:

| | information bits | parity bits | dead bits | identities |
|---|---|---|---|---|
| Option C, layer 2 (8 bits, fanout 64) | 6 | 1 | **1** | 64 |
| 3-layer, layers 2+3 (4+4 bits, fanout 8×8) | 6 | **2** | 0 | 64 |

Same 8 bits, same 64 identities — but option C's dead bit (§3 of that doc: the constant leading zero
forced by `fanout=64`) **becomes a second, real parity check**. This is not a trade-off; the 3-layer
split strictly dominates option C at equal capacity and equal payload.

The corrected general statement: **depth costs capacity, and when capacity is in surplus, that is
exactly how you buy accuracy.** Depth only hurts when you are capacity-bound. You need 1000
identities out of 65,536 addressable at 16 bits, so you are nowhere near bound — which is why
depth pays here and why it might not in a config with tight capacity.

---

## 3. Accuracy

Per-bit erasure rate `p`, guarantee model (`layer resolves uniquely ⟺ erasures ≤ d−1`):

| `p` | Segment RS(4,3) | Option C (8,8) | **3-layer (8,4,4)** | 3-layer container correct |
|---|---|---|---|---|
| 0.02 | 96.7 % | 99.0 % | **99.5 %** | 99.999 % |
| 0.05 | 84.1 % | 94.2 % | **97.2 %** | 99.963 % |
| 0.10 | 57.4 % | 80.9 % | **89.4 %** | 99.498 % |
| 0.15 | 34.6 % | 64.3 % | **77.6 %** | 97.865 % |
| 0.20 | 19.0 % | 47.5 % | **63.3 %** | 94.372 % |
| 0.30 | 4.5 % | 20.6 % | **34.2 %** | 80.590 % |

At `p = 0.10` the 3-layer scheme identifies exactly **89.4 %** of the time against RS's **57.4 %**,
and still names the correct container **99.5 %** of the time.

The RS column carries the ×3–4 symbol-erasure amplification explained in
[DESIGN_TRACING_VS_SEGMENT.md](DESIGN_TRACING_VS_SEGMENT.md) §2.1: RS groups bits into 4-bit
symbols, and this channel erases bits independently, so one erased bit destroys a whole symbol
against only `nsym = 1` symbol of redundancy.

### Distance profile (verified over all 1000 users)

| users diverge at | minimum distance |
|---|---|
| layer 1 (different container) | **4** |
| layer 2 (same container, different team) | 2 |
| layer 3 (same team, different user) | 2 |
| overall code | 2 |

Protection is concentrated where damage is largest: mistaking a container costs 64 users, a team
costs 8, a user costs 1.

---

## 4. Tracing time

| Decoder | Work per trace | Scales with N? |
|---|---|---|
| Segment RS ML search | **4,000** symbol compares | **linear** |
| Option C (8,8), popcount | 80 popcounts | sublinear |
| **3-layer (8,4,4), popcount** | **32 popcounts** | sublinear |
| Option C (8,8), tables | 2 lookups · 131,072 entries | constant |
| **3-layer (8,4,4), tables** | **3 lookups · 66,048 entries** | **constant** |

The 3-layer split is faster on *both* paths and needs **half the table memory**, because table size
is `4^b` per layer: `4⁸ + 4⁴ + 4⁴ = 66,048` versus `4⁸ + 4⁸ = 131,072`.

RS cannot use table decoding at all — a flat 16-bit table needs `4¹⁶ ≈ 4.3 billion` entries.
Table decoding is available *only* because the hierarchy splits `L` into short segments, which is
what makes this a property of the scheme rather than an implementation trick.

---

## 5. Assignment (1000 users, sequential)

Mixed radix with fanouts `(16, 8, 8)`:

```
row -> (row // 64, (row // 8) % 8, row % 8)
```

Verified: **1000 / 1000 codewords unique**, all 16 containers occupied, container 15 holding 40 users
(teams 0–4).

| row | path | codeword `L1 \| L2 \| L3` |
|---|---|---|
| 0 | (0, 0, 0) | `00000000` `0000` `0000` |
| 8 | (0, 1, 0) | `00000000` `0011` `0000` |
| 63 | (0, 7, 7) | `00000000` `1111` `1111` |
| 64 | (1, 0, 0) | `00010111` `0000` `0000` |
| 500 | (7, 6, 4) | `01110100` `1100` `1001` |
| 999 | (15, 4, 7) | `11111111` `1001` `1111` |

`children()` must return only occupied nodes — container 15 has 5 teams, not 8, and its last team
has fewer than 8 users.

---

## 6. Extra benefit: finer collusion containment

Three levels give three granularities of partial answer instead of two:

| deepest unambiguous level | containment set |
|---|---|
| layer 1 only | 64 users |
| layers 1–2 | **8 users** |
| layers 1–3 | 1 user (exact) |

Colluders who share a team are localised to 8 candidates out of 1000. Option C's coarsest useful
answer is 64; Segment-WM has no equivalent partial answer at all — it returns a wrong payload or
fails. This makes the collusion experiment materially stronger.

---

## 7. Code impact relative to the option C design

Almost nothing changes — this is a config swap plus one trivial codebook:

```json
{
  "name": "l16_8_4_4",
  "L": 16,
  "levels": [
    { "name": "container", "bits": 8, "fanout": 16, "min_distance": 4 },
    { "name": "team",      "bits": 4, "fanout":  8, "min_distance": 2 },
    { "name": "user",      "bits": 4, "fanout":  8, "min_distance": 2 }
  ],
  "assignment": "sequential"
}
```

- `EvenParityCodebook` already handles `(bits=4, d=2)` — same `(i << 1) | parity(i)` rule, no new code.
- `LinearCodebook` with `EXTENDED_HAMMING_8_4_4` is unchanged.
- `HierarchyIndex` mixed-radix and `decode_path` are depth-agnostic by construction.
- The decode-table builder loops over levels; `4⁴ = 256`-entry tables cost nothing to build.

This is the payoff from designing for general `D` in the first place: moving from 2 layers to 3 is a
JSON edit, not a rewrite.

---

## 8. What still has to be checked

1. **Calibration is still the gate.** Every accuracy number above is a function of `p`, and `p`
   depends on whether L=16 is even carryable — `z ≈ 54/L` puts it near 3.4 against a 4.0 threshold.
   Run `helper_scripts/calibrate_lbits.py` first. If L=16 forces `p > 0.2`, consider `L=12` with
   `(6,3,3)` and re-run this search at the measured `p`.
2. **The guarantee model is conservative.** It counts a layer as failed whenever erasures exceed
   `d−1`, though real codes often still resolve uniquely. Expect measured accuracy ≥ these numbers.
   Validate with Monte-Carlo over the actual codebooks before publishing.
3. **Flips vs erasures.** The table above models erasures only. The channel also produces `*`
   (collusion) and occasional flips. Layer 1 corrects 1 flip; layers 2 and 3 correct none. Re-run the
   search with a mixed flip/erasure channel once calibration gives the real mix.
4. **Soft-decision rescoring** (`llr = z_i1 − z_i0`) applies unchanged and should lift every row —
   apply it to the RS baseline too, or the comparison is unfair.

---

## 9. Build order (unchanged from the option C plan)

| Step | Deliverable |
|---|---|
| 1 | `calibrate_lbits.py` → measured `p`, decides L=16 vs L=12 |
| 2 | `hierarchy.py`: spec + codebooks + mixed-radix index (depth-agnostic) |
| 3 | `decode_path` tier 1 (bitwise popcount) + tests |
| 4 | Decode tables (tier 3) + `test_table_matches_bruteforce` |
| 5 | RS baseline harness at matched parameters |
| 6 | Monte-Carlo over `p` and `N` → the two headline figures |
| 7 | Soft-decision tier, applied to both schemes |

Only step 1 needs a GPU.
