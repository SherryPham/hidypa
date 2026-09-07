# Initial Code Design — 2-Layer Hierarchy, 8 + 8 bits (L = 16)

**Status:** design for review. No implementation yet.
**Config:** `D = 2`, `b_1 = 8` (container), `b_2 = 8` (user), `L = 16`.
Written general (D levels) but only D=2 is wired up and tested in this phase.

---

## 0. Read this first — L=16 is the main risk

Per-bit z-score scales as `z ∝ sqrt(blocks) / L`, because `LBitLogitProcessor` spreads a
permutation of the `L` bit positions across the high-entropy blocks: bit `i` is actually biased at
only ~`blocks/L` positions, while `detect()` normalises over all `blocks`.

```
total_score ≈ (blocks / L) · c        denominator = sqrt(blocks)
z           ≈ c · sqrt(blocks) / L
```

Your measured rule `z ≈ 54 / L` (at the token budget used for the L=8 tables) implies:

| L | expected z | vs threshold 4.0 |
|---|---|---|
| 8  | ~6.8 | comfortable |
| 13 | ~4.2 | marginal |
| **16** | **~3.4** | **below threshold — bits decode as `⊥`** |

To hold `z` constant, `blocks` (hence tokens) must scale as `L²`. Going 8 → 16 bits needs **~4×
the tokens**: `max_new_tokens` 512 → ~2048.

**Therefore step 1 of implementation is a calibration script, not the hierarchy** (§7). If L=16 at
2048 tokens still gives a high erasure rate on your model, the honest options are: raise tokens
further, lower `z_threshold` and accept more `*`, or drop to 2 layers × 6 bits (L=12). The
hierarchy code is identical either way — only the config changes.

---

## 1. Parameter choice for 8 + 8

`capacity(b, d)` = number of usable codewords:

| d | construction | capacity at b=8 | corrects (flips) | corrects (erasures) |
|---|---|---|---|---|
| 1 | identity (all words) | 256 | 0 | 0 |
| 2 | even parity `[8,7,2]` | 128 | 0 | 1 |
| 4 | extended Hamming `[8,4,4]` | 16 | 1 | 3 |

`t_flip = floor((d-1)/2)`, `erasures = d-1`.

You have **1000 users** but `L=16` addresses up to 65 536 — roughly 6 bits of pure slack. Spending
that slack on minimum distance is the whole point of having it, and it directly fixes the weakness
raised earlier (even parity corrects *zero* bit flips).

| Option | L1 (d, fanout) | L2 (d, fanout) | capacity | notes |
|---|---|---|---|---|
| A — capacity-first | (2, 128) | (1, 256) | 32 768 | naive generalisation of today's defaults; no correction anywhere |
| B — balanced | (2, 128) | (2, 128) | 16 384 | 1 erasure correctable per level |
| **C — recommended** | **(4, 16)** | **(2, 64)** | **1 024** | top level corrects 1 flip / 3 erasures; all 16 containers occupied by 1000 users (~63 each) |

**Recommendation: option C.** It puts the redundancy where errors are most expensive — an error in
the level-1 segment sends you to the wrong container and loses every user beneath it, whereas a
level-2 error costs one user. C is also the only option that actually exercises the hierarchy: with
option A's fanout of 256, your 1000 users occupy just 4 containers.

Make this a config field, not a constant — A and B are then one-line experiment variants.

---

## 2. Module layout

```
src/hierarchy.py                   NEW   pure logic, stdlib only (no torch, no pandas)
src/hierarchical_watermark.py      NEW   ties hierarchy.py to LBitWatermarker
config/hierarchies/l16_8_8.json    NEW   the config below
tests/test_hierarchy.py            NEW   pytest, CPU-only, runs in seconds
helper_scripts/calibrate_lbits.py  NEW   §7 — run BEFORE any sweep
src/main_multiuser.py              EDIT  --scheme hierarchical, --hierarchy-config, --levels
src/watermark.py                   EDIT  HiDyPa reimplemented on hierarchy.py (behaviour unchanged)
```

`hierarchy.py` stays free of torch/pandas deliberately: the entire codebook + decode layer is then
testable on CPU in seconds with no model download, which is where the recurring decode bugs live.

---

## 3. `src/hierarchy.py`

### 3.1 Constants and specs

```python
ERASURE_SYMBOLS = frozenset({"⊥", "*", "?"})   # single source of truth

@dataclass(frozen=True)
class LevelSpec:
    name: str
    bits: int
    fanout: int
    min_distance: int

@dataclass(frozen=True)
class HierarchySpec:
    levels: tuple[LevelSpec, ...]
    assignment: str = "sequential"

    @property
    def L(self) -> int: ...                 # sum of bits
    @property
    def depth(self) -> int: ...
    @property
    def offsets(self) -> tuple[tuple[int, int], ...]: ...   # ((0,8), (8,16)) here
    def capacity(self) -> int: ...          # product of fanouts
    def validate(self) -> None: ...         # raises ValueError, never silently repairs

    @classmethod
    def from_json(cls, path: str) -> "HierarchySpec": ...
    @classmethod
    def from_cli(cls, spec: str) -> "HierarchySpec": ...   # "container:8:16:4,user:8:64:2"
    @classmethod
    def legacy_two_level(cls, group_bits, user_bits,
                         max_groups=None, users_per_group=None,
                         min_distance=2) -> "HierarchySpec": ...
    def to_dict(self) -> dict: ...          # provenance block in result JSON
```

`validate()` checks: `depth >= 1`; every `bits >= 1`; `1 <= min_distance <= bits`;
`1 <= fanout <= capacity(bits, min_distance)`. It does **not** check `L` against the watermarker —
that happens in `HierarchicalMultiUserWatermarker.__init__` so the error can name both numbers.

### 3.2 Codebooks

```python
class Codebook(ABC):
    bits: int; min_distance: int; fanout: int
    def __getitem__(self, index: int) -> str: ...   # binary string, length == bits, MSB-first
    def __len__(self) -> int: ...
    def achieved_min_distance(self) -> int: ...     # test hook; brute force over all pairs

class IdentityCodebook(Codebook):       # d == 1
    # format(index, f"0{bits}b")

class EvenParityCodebook(Codebook):     # d == 2
    # cw = (index << 1) | parity(index);  format(cw, f"0{bits}b")
    # MUST reproduce watermark.py::_generate_single_group_codeword_int exactly

class LinearCodebook(Codebook):         # d >= 3 with a known generator matrix
    # index -> message bits -> GF(2) matrix multiply -> codeword
    # ships with EXTENDED_HAMMING_8_4_4 for (bits=8, d=4)

class GreedyCodebook(Codebook):         # generic (bits, d) fallback
    # max-min greedy ported from FingerprintingCode._generate_bch_codewords,
    # memoised per (bits, d) and generated once. At bits=8 that is 256
    # candidates -> microseconds, so no perf concern in this phase.

@lru_cache(maxsize=None)
def make_codebook(bits: int, min_distance: int, fanout: int) -> Codebook: ...
```

Dispatch: `d==1` → Identity; `d==2` → EvenParity; `(8,4)` → LinearCodebook with the extended Hamming
generator; otherwise → Greedy.

Two rules the tests must enforce:

1. **Canonical bit order is MSB-first everywhere.** The existing
   `FingerprintingCode._generate_bch_codewords` builds bit arrays LSB-first while the Hi-DyPa lazy
   `d=2` path uses `format(..., '0Gb')` (MSB-first). Distance is invariant under reversal so no past
   result is wrong, but the two emit different *strings* for the same index. Pick MSB-first (what all
   your existing runs used) and assert it.
2. **Greedy may under-deliver.** Greedy max-min is not guaranteed to reach `A2(b,d)`. `make_codebook`
   must raise if `len(codebook) < fanout` rather than silently returning fewer codewords — the
   current code only prints a warning, which would quietly shrink capacity.

### 3.3 Index (path ↔ row ↔ codeword)

```python
class HierarchyIndex:
    def __init__(self, spec: HierarchySpec, num_users: int): ...
    def path_of_row(self, row: int) -> tuple[int, ...]: ...   # mixed radix, leaf least-significant
    def row_of_path(self, path: tuple[int, ...]) -> int: ...
    def codeword_of_row(self, row: int) -> str: ...
    def codeword_of_path(self, path: tuple[int, ...]) -> str: ...
    def children(self, prefix: tuple[int, ...]) -> range: ...  # only OCCUPIED children
    def rows_under(self, prefix: tuple[int, ...]) -> range: ...
    def label(self, level: int, child_index: int) -> str: ...
```

Mixed radix, leaf least-significant:

```
row 500, fanouts (16, 64):  i_2 = 500 % 64 = 52 ;  i_1 = 500 // 64 = 7   ->  path (7, 52)
```

For `D=2` this is exactly `divmod(row, users_per_group)` — i.e. today's
`group_id = index // users_per_group`. That equivalence is a test, and it is why the 2-level path
stays bit-identical.

`children()` returns only occupied children, so a partially-filled last container never produces
phantom candidates (1000 users in 16×64 = 1024 slots leaves container 15 with 40 users).

**Performance:** `row` is the position in metadata sorted by `UserId` — matching current behaviour.
A precomputed `user_id -> row` dict replaces the `O(N)` pandas scan currently done per candidate
inside the trace loop.

### 3.4 Decode

```python
@dataclass
class DecodeResult:
    path: tuple[int, ...] | None
    row: int | None
    ties: list[tuple[int, ...]]
    per_level_distance: list[int]
    per_level_margin: list[int | None]    # runner-up minus best; None if unique candidate
    per_level_candidates: list[int]       # how many survived each level (diagnostics)
    cumulative_distance: int | None
    containment_path: tuple[int, ...]     # longest prefix common to all ties == predicted LCA
    containment_level: int
    candidates_evaluated: int             # cost metric

def decode_path(recovered: str, index: HierarchyIndex, *,
                beam_width: int = 0,      # 0 = keep every minimum-distance candidate
                margin: int = 0) -> DecodeResult: ...
```

Algorithm:

```python
segments = [recovered[s:e] for (s, e) in spec.offsets]     # 8 + 8
frontier = [((), 0)]

for level in range(depth):
    candidates = []
    for prefix, cum in frontier:
        for child in index.children(prefix):
            label = index.label(level, child)
            dist = sum(segments[level][j] != label[j]
                       for j in range(spec.levels[level].bits)
                       if segments[level][j] not in ERASURE_SYMBOLS)
            candidates.append((prefix + (child,), cum + dist))
    if not candidates:
        return DecodeResult(path=None, ...)
    best = min(c for _, c in candidates)
    kept = [x for x in candidates if x[1] <= best + margin]
    frontier = sorted(kept, key=itemgetter(1))[:beam_width] if beam_width else kept

ties = [p for p, c in frontier if c == min_cum]
containment_path = longest prefix shared by every tie
```

Notes:
- Erased positions are **skipped**, not counted as mismatches — same as today.
- A fully-erased level contributes distance 0 to every child, so the frontier fans out to all
  children. Correct semantics (no information), recorded via `per_level_candidates`.
- `beam_width=0, margin=0` (default) reproduces the current two-stage trace exactly for D=2.
  `beam_width=inf, margin=inf` is exhaustive nearest-codeword search — the accuracy upper bound.
- `containment_path` is the collusion-localisation output: when the leaf is ambiguous it names the
  deepest node that provably contains every candidate.

---

## 4. `src/hierarchical_watermark.py`

```python
class HierarchicalMultiUserWatermarker(NaiveMultiUserWatermarker):
    def __init__(self, lbit_watermarker: LBitWatermarker, spec: HierarchySpec):
        # raises if spec.L != lbit_watermarker.L, naming both numbers
    def load_users(self, users_file: str) -> pd.DataFrame: ...
    def get_codeword_for_user(self, user_id: int) -> str: ...
    def embed(self, master_key, user_id, prompt, **kw) -> str: ...   # inherited shape
    def trace(self, master_key, text, **kw) -> list[dict]: ...
    def trace_from_codeword(self, recovered: str) -> list[dict]: ...
    def decode(self, recovered: str) -> DecodeResult: ...            # richer, for eval scripts
```

`load_users` replaces the four overlapping truncation blocks currently in
`HiDyPaMultiUserWatermarker.load_users` with one check:

```python
if len(df) > spec.capacity():
    warn(f"users file has {len(df)} rows; spec capacity is {spec.capacity()}; truncating")
    df = df.head(spec.capacity())
```

`trace_from_codeword` returns today's dict plus new keys, so nothing downstream breaks:

```python
{"user_id": 500, "username": "500", "match_score_percent": 93.8,
 "group_id": 7,                       # == path[0], kept for backward compat
 "path": [7, 52], "path_names": {"container": 7, "user": 52},
 "per_level_distance": [0, 1], "cumulative_distance": 1,
 "containment_path": [7], "containment_level": 1}
```

`HiDyPaMultiUserWatermarker` is kept and reimplemented as a thin subclass building
`HierarchySpec.legacy_two_level(...)`, still exposing `group_bits`, `user_bits`, `_num_groups`,
`_users_per_group`, `group_to_users`, `_get_group_codeword_str()` — the attributes the evaluation
scripts read directly.

---

## 5. Config file

`config/hierarchies/l16_8_8.json` (option C):

```json
{
  "name": "l16_8_8_robust",
  "L": 16,
  "levels": [
    { "name": "container", "bits": 8, "fanout": 16, "min_distance": 4 },
    { "name": "user",      "bits": 8, "fanout": 64, "min_distance": 2 }
  ],
  "assignment": "sequential"
}
```

Also ship `l16_8_8_capacity.json` (option A) and `l16_8_8_balanced.json` (option B) so the
comparison is a flag change.

CLI equivalents:

```
--hierarchy-config config/hierarchies/l16_8_8.json
--levels "container:8:16:4,user:8:64:2"          # name:bits:fanout:min_distance
```

Example run:

```
python -m src.main_multiuser generate "Explain photosynthesis." \
    --scheme hierarchical --hierarchy-config config/hierarchies/l16_8_8.json \
    --l-bits 16 --user-id 500 --model gpt2 --max-new-tokens 2048

python -m src.main_multiuser trace demonstration/multiuser_output.txt \
    --scheme hierarchical --hierarchy-config config/hierarchies/l16_8_8.json \
    --l-bits 16 --model gpt2
```

Trace output:

```
Recovered: 01101001 | 00110100
Traced path: container=7 -> user=52   (User ID 500, cumulative distance 1)
Level margins: container +4, user +2
```

Ambiguous case:

```
Ambiguous at level 2. Confined to container=7 (63 users, was 1000).
```

---

## 6. Tests (`tests/test_hierarchy.py`, pytest, no model)

Spec / codebook
- `test_spec_validation` — `L != sum(bits)`, `fanout > capacity`, `d > bits` each raise
- `test_codebook_min_distance` — brute-force all pairs for `b in 2..10`, `d in 1..4`
- `test_codebook_capacity` — identity 256, even parity 128, extended Hamming 16 at b=8
- `test_codebook_raises_when_short` — greedy under-delivering raises, not warn-and-shrink
- `test_even_parity_matches_legacy` — equals `_generate_single_group_codeword_int` for all indices
- `test_bit_order_msb_first`

Index
- `test_path_row_roundtrip` — all 1000 rows, fanouts (16, 64)
- `test_mixed_radix_matches_divmod` — D=2 equals today's `divmod(row, users_per_group)`
- `test_children_only_occupied` — 1000 users in 16×64: container 15 has 40 children, not 64

Decode
- `test_decode_clean` — every user's own codeword decodes back to that user
- `test_decode_one_flip_level1` — d=4 container level corrects a single flip
- `test_decode_erasures` — up to 3 erasures in the container segment still resolve
- `test_decode_full_erasure_level1` — fans out to all containers, level 2 still scored
- `test_decode_matches_two_stage` — D=2, beam=0, margin=0 equals current `trace_from_codeword`
- `test_beam_equivalence` — unbounded beam equals exhaustive nearest-codeword search

Backward compatibility
- `test_legacy_codeword_identical` — for `L=8`, every `(G,U)` split, every user: new string == old
  string (golden JSON committed to the repo)

Collusion
- `test_containment_path` — build `*` strings from colluders sharing a container; assert
  `containment_path == (container,)` and `containment_level == 1`

---

## 7. `helper_scripts/calibrate_lbits.py` — run this FIRST

Purpose: find out whether L=16 is usable on your model before committing GPU time to a sweep.

```
python helper_scripts/calibrate_lbits.py \
    --model deepseek-llm-7b --l-bits 8,12,16 \
    --max-new-tokens 512,1024,2048 --num-prompts 20
```

For each `(L, tokens)` it embeds a known random codeword, detects, and reports:

| column | why |
|---|---|
| `blocks` | high-entropy positions found |
| `blocks_per_bit` = blocks / L | the quantity that drives z |
| `mean_z` per bit | compare against `z_threshold` |
| `erasure_rate` | fraction of `⊥` |
| `collision_rate` | fraction of `*` |
| `exact_recovery` | fraction of trials with all L bits correct |

Decision rule: pick the smallest `max_new_tokens` where `erasure_rate < 0.05` at `L=16`. If no
setting reaches it, fall back to **2 × 6 bits (L=12)** and note the reason — the hierarchy code does
not change, only the config.

This is ~20 generations per cell instead of 300, so it is cheap, and it produces a figure
(`blocks_per_bit` vs `exact_recovery`) worth including in the write-up.

---

## 8. Build order

| Step | Deliverable | Done when |
|---|---|---|
| 1 | `calibrate_lbits.py` + calibration run | you know the token budget for L=16, or have chosen L=12 |
| 2 | `hierarchy.py`: spec + codebooks + index | spec/codebook/index tests pass |
| 3 | `hierarchy.py`: `decode_path` | decode tests pass, including two-stage equivalence |
| 4 | `hierarchical_watermark.py` | golden backward-compat test passes; generate+trace works on gpt2 |
| 5 | `main_multiuser.py` wiring + configs | the §5 commands run end to end |
| 6 | Port `HiDyPaMultiUserWatermarker` onto it; delete the 5 duplicated `decode_hi_dypa_user` | existing eval scripts produce byte-identical output on a saved sample |

Steps 2–5 need no GPU. Step 1 is the only one that must run first, and it is the one that decides
whether L=16 survives.

---

## 9. Open questions

1. **Option A, B, or C** for the (d, fanout) pair? I recommend C.
2. **L=16 vs L=12** if calibration shows 2048 tokens is not enough — 2×6 keeps the two-layer
   structure at a payload your model can actually carry.
3. Do you want `main_multiuser.py --scheme hierarchical` now, or is the Python API enough until the
   evaluation scripts are migrated?
