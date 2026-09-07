# Option C — Design Specification

**2 layers × 8 bits, L = 16, distance-optimal.**
Companion to [DESIGN_2LAYER_8_8.md](DESIGN_2LAYER_8_8.md), which covers the shared machinery
(`HierarchySpec`, `Codebook`, `HierarchyIndex`, `decode_path`). This document fixes every concrete
value for option C and lists the golden numbers the tests must assert.

All tables below were computed and verified, not asserted.

---

## 1. Parameters

| | Layer 1 (`container`) | Layer 2 (`user`) |
|---|---|---|
| bits | 8 | 8 |
| fanout | 16 | 64 |
| min distance | 4 | 2 |
| construction | extended Hamming `[8,4,4]` | even parity `[8,7,2]`, first 64 words |
| corrects | `2t + e ≤ 3` → 1 flip, or 3 erasures, or 1 flip + 1 erasure | `2t + e ≤ 1` → 1 erasure, 0 flips |

**Capacity = 16 × 64 = 1024** (≥ 1000 users ✓)

### Both layers are distance-optimal for their fanout

Using known values of `A2(8,d)` = `{1: 256, 2: 128, 3: 20, 4: 16, 5: 4}`:

- Layer 1 needs 16 codewords → the largest `d` with `A2(8,d) ≥ 16` is **d = 4**.
- Layer 2 needs 64 codewords → the largest `d` with `A2(8,d) ≥ 64` is **d = 2**.

So option C is not a compromise at either level — it is the maximum achievable minimum distance
given 8 bits and these fanouts. `d=2` at layer 2 is forced (`A2(8,3) = 20 < 64`), not chosen.

---

## 2. Layer 1 codebook — extended Hamming `[8,4,4]`

Systematic: message `m0..m3` (MSB-first) followed by four parity bits.

```
p1 = m0 ^ m1 ^ m2
p2 =      m1 ^ m2 ^ m3
p3 = m0 ^ m1      ^ m3
p4 = m0 ^ m1 ^ m2 ^ m3 ^ p1 ^ p2 ^ p3      (overall parity)

codeword = m0 m1 m2 m3 p1 p2 p3 p4
```

**Verified:** 16 unique codewords, minimum pairwise distance **exactly 4**, weight spectrum
`{0, 4, 8}` (as expected — this code is self-dual).

| container | codeword | | container | codeword |
|---|---|---|---|---|
| 0 | `00000000` | | 8 | `10001011` |
| 1 | `00010111` | | 9 | `10011100` |
| 2 | `00101101` | | 10 | `10100110` |
| 3 | `00111010` | | 11 | `10110001` |
| 4 | `01001110` | | 12 | `11000101` |
| 5 | `01011001` | | 13 | `11010010` |
| 6 | `01100011` | | 14 | `11101000` |
| 7 | `01110100` | | 15 | `11111111` |

This table is small and fixed — **commit it as a golden fixture** and have `LinearCodebook` assert
against it, rather than trusting the generator matrix to be transcribed correctly.

---

## 3. Layer 2 codebook — even parity, first 64 words

```
codeword(i) = format((i << 1) | parity(i), '08b')      for i in 0..63
```

Identical rule to the existing `_generate_single_group_codeword_int` in
[watermark.py:706](../src/watermark.py#L706), so the even-parity path stays shared with the legacy
2-level code.

**Verified:** 64 unique codewords, minimum distance **exactly 2**, all even weight.

```
i=0  -> 00000000      i=4  -> 00001001
i=1  -> 00000011      i=5  -> 00001010
i=2  -> 00000101      i=6  -> 00001100
i=3  -> 00000110      i=7  -> 00001111
...
i=62 -> 01111101      i=63 -> 01111110
```

### Known wart: layer 2's leading bit is constant

Because `i < 64`, `(i << 1) | parity` is always `< 128`, so **bit 0 of every layer-2 codeword is
`0`**. That position carries no information: it matches every candidate equally, so it neither helps
nor hurts decoding — it is simply a wasted payload bit.

It is wasted because 64 codewords at `d=2` only need 7 bits (`A2(7,2) = 64` exactly). Three ways to
handle it:

| Variant | Layout | Capacity | Containers used at N=1000 | Notes |
|---|---|---|---|---|
| **C1 (recommended)** | 8+8, fanout (16, 64) | 1024 | **16 of 16** | as specified here; one dead bit |
| C2 | 8+8, fanout (16, 128) | 2048 | 8 of 16 | no dead bit, more headroom, but under-exercises layer 1 |
| C3 | 8+**7**, fanout (16, 64) | 1024 | 16 of 16 | nothing wasted, and `L=15` slightly eases the z-score problem |

**Recommend C1 for the first implementation**: it matches your stated "two layers of 8 bits", and it
is the only variant where all 16 containers are occupied, which keeps the layer-1 decoding problem
non-trivial. Revisit after calibration — if L=16 turns out marginal, **C3 is a free bit back**.

---

## 4. Assignment (sequential, 1000 users)

```
row -> (container, index) = divmod(row, 64)
```

| | value |
|---|---|
| containers occupied | **16 of 16** |
| users in containers 0–14 | 64 each |
| users in container 15 | **40** (rows 960–999) |
| unique codewords | **1000 / 1000** (verified) |

`children((15,))` must therefore return 40 entries, not 64 — this is why `HierarchyIndex.children()`
returns only occupied children.

Worked examples (verified):

| row | container | index | codeword (`layer1 \| layer2`) |
|---|---|---|---|
| 0 | 0 | 0 | `00000000` `00000000` |
| 63 | 0 | 63 | `00000000` `01111110` |
| 64 | 1 | 0 | `00010111` `00000000` |
| 500 | 7 | 52 | `01110100` `01101001` |
| 999 | 15 | 39 | `11111111` `01001110` |

---

## 5. Distance profile of the full 16-bit code

Verified over all 1000 users:

| pair type | minimum distance |
|---|---|
| same container | **2** |
| different containers | **4** |
| overall code | **2** |

This asymmetry *is* the design. Confusing two users inside a container costs one user; confusing two
containers costs 64. So the protection is concentrated where the damage is:

- A single bit flip anywhere in the layer-1 segment is **corrected** — you stay in the right container.
- A single bit flip in the layer-2 segment may misidentify the user, but never the container.

Compare with your current L=8 naive scheme, where the code uses all 256 words at `d=1` and any
single flip silently changes identity.

---

## 6. Decoding

Uses `decode_path` from the shared design with defaults `beam_width=0, margin=0`.

```
segments = recovered[0:8], recovered[8:16]

Stage 1 — container
    for each of the 16 layer-1 codewords: distance over non-erased positions
    keep all containers at minimum distance
Stage 2 — user
    for each surviving container, for each OCCUPIED child index:
        cumulative distance = stage-1 distance + layer-2 distance
    keep all at minimum cumulative distance

ties -> containment_path = longest shared prefix
```

**Cost:** 16 comparisons at layer 1 + 64 per surviving container. Typical trace ≈ **80 comparisons
vs 1000** for a flat scan over the same user set.

### What decoding recovers (`2t + e ≤ d − 1`)

| damage to layer-1 segment | outcome |
|---|---|
| 0–3 erasures | container recovered exactly |
| 1 flip | container recovered exactly |
| 1 flip + 1 erasure | container recovered exactly |
| 2 flips | **ambiguous** — ties, `containment_path` empty, decoder reports failure rather than guessing |
| 4+ erasures | fans out; layer 2 still scored across all candidates |

| damage to layer-2 segment | outcome |
|---|---|
| 1 erasure | user recovered exactly |
| 1 flip | may land on a neighbouring user; container still correct |

The "2 flips → report failure" behaviour matters: with `d=4` the decoder can *tell* it is beyond its
correction radius, which the current `d=1`/`d=2` design cannot. Surface that as
`per_level_margin[0] == 0` (a tie at layer 1) rather than silently returning the first candidate.

---

## 7. Config file

`config/hierarchies/l16_8_8_optionC.json`:

```json
{
  "name": "optionC_l16_8_8",
  "L": 16,
  "levels": [
    { "name": "container", "bits": 8, "fanout": 16, "min_distance": 4 },
    { "name": "user",      "bits": 8, "fanout": 64, "min_distance": 2 }
  ],
  "assignment": "sequential"
}
```

CLI equivalent: `--levels "container:8:16:4,user:8:64:2"`

```
python -m src.main_multiuser generate "Explain photosynthesis." \
    --scheme hierarchical \
    --hierarchy-config config/hierarchies/l16_8_8_optionC.json \
    --l-bits 16 --user-id 500 --model gpt2 --max-new-tokens 2048
```

Expected trace output for user 500:

```
Recovered: 01110100 | 01101001
Traced path: container=7 -> user=52   (User ID 500, cumulative distance 0)
Level margins: container +4, user +2
```

---

## 8. Code changes specific to option C

Everything is shared machinery except one new class:

```python
# src/hierarchy.py

EXTENDED_HAMMING_8_4_4 = [        # golden fixture, §2
    "00000000", "00010111", "00101101", "00111010",
    "01001110", "01011001", "01100011", "01110100",
    "10001011", "10011100", "10100110", "10110001",
    "11000101", "11010010", "11101000", "11111111",
]

class LinearCodebook(Codebook):
    """Codebook from an explicit table or GF(2) generator matrix."""
    def __init__(self, words: list[str], bits: int, min_distance: int, fanout: int)
```

`make_codebook` dispatch gains one case:

```python
if (bits, min_distance) == (8, 4):
    return LinearCodebook(EXTENDED_HAMMING_8_4_4, 8, 4, fanout)
```

The generic `GreedyCodebook` stays as the fallback for other `(bits, d)` pairs, but option C never
reaches it — so **option C does not depend on greedy search succeeding**, which removes the risk
that greedy under-delivers and silently shrinks capacity.

---

## 9. Tests for option C (exact expected values)

```python
def test_layer1_is_extended_hamming():
    cb = make_codebook(bits=8, min_distance=4, fanout=16)
    assert len(cb) == 16
    assert cb.achieved_min_distance() == 4
    assert [cb[i] for i in range(16)] == EXTENDED_HAMMING_8_4_4
    assert sorted({w.count("1") for w in EXTENDED_HAMMING_8_4_4}) == [0, 4, 8]

def test_layer2_even_parity_64():
    cb = make_codebook(bits=8, min_distance=2, fanout=64)
    assert len(cb) == 64
    assert cb.achieved_min_distance() == 2
    assert all(w.count("1") % 2 == 0 for w in (cb[i] for i in range(64)))
    assert cb[0] == "00000000" and cb[63] == "01111110"

def test_option_c_capacity_and_assignment():
    idx = HierarchyIndex(OPTION_C, num_users=1000)
    assert OPTION_C.capacity() == 1024
    assert idx.path_of_row(500) == (7, 52)
    assert idx.path_of_row(999) == (15, 39)
    assert len(idx.children((15,))) == 40          # partially filled container
    assert len({idx.codeword_of_row(r) for r in range(1000)}) == 1000

def test_option_c_distance_profile():
    # same container -> 2, different containers -> 4, overall -> 2
    ...

def test_layer1_corrects_one_flip():
    cw = idx.codeword_of_row(500)                  # container 7
    for pos in range(8):                           # flip each layer-1 bit in turn
        damaged = flip(cw, pos)
        assert decode_path(damaged, idx).path[0] == 7

def test_layer1_corrects_three_erasures():
    for positions in combinations(range(8), 3):
        damaged = erase(cw, positions)
        assert decode_path(damaged, idx).path[0] == 7

def test_layer1_two_flips_reports_ambiguity():
    damaged = flip(flip(cw, 0), 1)
    result = decode_path(damaged, idx)
    assert result.per_level_margin[0] == 0         # tie, not a silent wrong answer

def test_containment_on_collusion():
    # two users in container 7 -> layer-2 positions become '*', layer 1 clean
    assert decode_path(colluded, idx).containment_path == (7,)
```

---

## 10. Still open

1. **Calibration first.** L=16 sits near `z ≈ 3.4` against a 4.0 threshold; `helper_scripts/calibrate_lbits.py`
   decides the token budget (or sends you to L=12). None of the code above changes if that happens —
   only the config file does. This is the one step that must precede implementation.
2. **C1 vs C3** (8+8 with a dead bit, vs 8+7 at L=15). I would ship C1 now and reconsider once
   calibration numbers exist, since C3's saved bit directly helps the z-score.
3. **Users file.** 1000 users against capacity 1024 leaves 24 spare. Fine as is; if the file grows,
   switch layer 2's fanout to 128 (variant C2) rather than changing the layer widths.
