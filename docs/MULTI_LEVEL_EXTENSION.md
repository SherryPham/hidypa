# Multi-Level Hierarchical Watermarking — Requirements & Implementation Instructions

**Status:** Draft for review — no code written yet.
**Scope:** Generalise Hi-DyPa from a fixed 2-level hierarchy (group → user) to a user-configurable
D-level hierarchy (e.g. org → division → team → user), with arbitrary depth, arbitrary bit
allocation per level, and arbitrary branching factors.

---

## 1. Goal

Today the codeword is hard-wired as exactly two segments:

```
codeword = group_code (G bits) ++ user_code (U bits),   G + U = L
```

After this extension the codeword becomes an arbitrary-depth concatenation:

```
codeword = c_1 (b_1 bits) ++ c_2 (b_2 bits) ++ ... ++ c_D (b_D bits),   sum(b_l) = L
```

where the number of levels `D`, the width `b_l`, the branching factor `f_l`, and the minimum
Hamming distance `d_l` of every level are chosen by the user in a config file or on the CLI.
Level `D` is the leaf level (individual users); levels `1..D-1` are nested containers.

The 2-level case must remain reachable and must produce **bit-identical codewords** to the current
implementation, so that all Table 2/3/4 results already collected stay valid.

---

## 2. What exists today (baseline to generalise)

| Concern | Where | Current behaviour |
|---|---|---|
| Group codebook | [watermark.py:706-745](../src/watermark.py#L706-L745) `_generate_single_group_codeword_int` | `d=2` even-parity, lazy: `cw = (group_id << 1) \| parity(group_id)`; capacity `2^(G-1)` |
| Group codebook (`d>2`) | [fingerprinting.py:54-185](../src/fingerprinting.py#L54-L185) `_generate_bch_codewords` | max-min greedy over all `2^L` words; **eager**, `O(2^L · k)` |
| User fingerprint | [fingerprinting.py:353-375](../src/fingerprinting.py#L353-L375) `generate_user_fingerprint` | plain binary index within group, `d=1`, capacity `2^U` |
| Assignment | [watermark.py:1001-1007](../src/watermark.py#L1001-L1007) | `group_id = row_index // users_per_group` (sequential, balanced) |
| Encode | [watermark.py:1016-1071](../src/watermark.py#L1016-L1071) `get_codeword_for_user` | `group_code + user_code`, validated against `L` |
| Decode | [watermark.py:1118-1247](../src/watermark.py#L1118-L1247) `trace_from_codeword` | Stage 1: nearest group codeword over group segment (keep all ties). Stage 2: nearest **full** codeword among users in tied groups |
| Decode (eval copies) | 5 duplicates of `decode_hi_dypa_user` in `evaluation_scripts/` | Same algorithm, drifted independently — this is where the decode bugs kept reappearing |
| CLI | [main_multiuser.py:79-82](../src/main_multiuser.py#L79-L82) | `--group-bits`, `--user-bits`, `--max-groups`, `--users-per-group`, `--min-distance {2}` |

**Key observation:** `(group_id, index_in_group) = divmod(row_index, users_per_group)` is exactly a
2-digit mixed-radix decomposition. The D-level generalisation is a D-digit mixed-radix
decomposition, so the existing behaviour falls out of the general one for free.

---

## 3. Target model

### 3.1 Notation

- `D` — number of levels (depth). `D = 1` degenerates to the naive flat scheme; `D = 2` is today's Hi-DyPa.
- `b_l` — bits assigned to level `l`, `l ∈ {1..D}`, with `Σ b_l = L`.
- `f_l` — branching factor of level `l` (number of distinct children a node at level `l-1` can have).
- `d_l` — minimum Hamming distance of the level-`l` codebook.
- `C_l` — the level-`l` codebook: an injective map `{0..f_l-1} → {0,1}^{b_l}` with pairwise distance `≥ d_l`.
- **Path** of a user: `p = (i_1, i_2, ..., i_D)`, `i_l ∈ {0..f_l-1}`.
- **Codeword**: `w(p) = C_1[i_1] ++ C_2[i_2] ++ ... ++ C_D[i_D]`.

### 3.2 Capacity

```
capacity(b, d) = 2^b            if d = 1   (identity codebook)
               = 2^(b-1)        if d = 2   (even-parity codebook)
               = A2(b, d)       if d ≥ 3   (greedy max-min; bounded by helper_scripts/compute_code_capacity.py)

Constraint per level:  f_l ≤ capacity(b_l, d_l)
Total user capacity:   N_max = Π_l f_l
```

### 3.3 Distance guarantee (what the hierarchy buys)

For two users `p ≠ q` whose paths first differ at level `k`:

```
d_H(w(p), w(q)) ≥ d_k
```

i.e. the separation between two users is governed by the **shallowest level at which they diverge**.
Users in different top-level containers are far apart; siblings are close. This is the property that
makes coarse-to-fine tracing work, and it is what we will measure.

### 3.4 Label independence

Default: **path-independent labels** — `C_l` is one shared codebook per level, so a node's label
depends only on its index among its siblings, not on its ancestors. This matches the current
implementation and keeps memory at `O(Σ f_l)` instead of `O(N)`.

Optional (flag `path_dependent_labels: true`, phase 5+): derive each node's label from
`HMAC(master_key, path_prefix)` restricted to the codebook, so identical sibling indices under
different parents get different labels. This hardens against *segment-splicing* collusion but costs a
key-dependent codebook (detector needs the key — which it already has). **Recommendation: implement
the flag and the plumbing, default it off, evaluate it separately.**

---

## 4. Configuration format

### 4.1 JSON config (primary interface)

`config/hierarchies/<name>.json`:

```json
{
  "name": "org3",
  "L": 12,
  "levels": [
    { "name": "region", "bits": 3, "fanout": 4,  "min_distance": 2 },
    { "name": "team",   "bits": 4, "fanout": 8,  "min_distance": 2 },
    { "name": "user",   "bits": 5, "fanout": 32, "min_distance": 1 }
  ],
  "assignment": "sequential",
  "path_dependent_labels": false
}
```

Field rules:

| Field | Required | Default | Validation |
|---|---|---|---|
| `L` | yes | — | `L == Σ bits`; error otherwise (do **not** silently pad) |
| `levels[].name` | no | `level{i}` | used in logs, output dirs, metric keys |
| `levels[].bits` | yes | — | `≥ 1` |
| `levels[].fanout` | no | `capacity(bits, min_distance)` | `1 ≤ fanout ≤ capacity(bits, min_distance)` |
| `levels[].min_distance` | no | `2` for `l < D`, `1` for `l = D` | `1 ≤ d ≤ bits` |
| `assignment` | no | `sequential` | `sequential` \| `explicit` |
| `path_dependent_labels` | no | `false` | bool |

### 4.2 CLI shorthand

```
--levels "region:3:4:2,team:4:8:2,user:5:32:1"      # name:bits:fanout:min_distance
--levels "3,4,5"                                     # bits only; fanout/d take defaults
--hierarchy-config config/hierarchies/org3.json      # file form (takes precedence)
```

### 4.3 Legacy mapping (must keep working)

```
--scheme hi_dypa --group-bits G --user-bits U [--max-groups M] [--users-per-group K]
```
translates internally to

```
levels = [ {bits: G, fanout: M  or 2^(G-1), min_distance: 2},
           {bits: U, fanout: K  or 2^U,     min_distance: 1} ]
```

`--max-groups` and `--users-per-group` are just `fanout` on levels 1 and 2. Emit a one-line
deprecation notice, keep the flags forever (all existing slurm scripts use them).

### 4.4 Explicit (non-balanced) hierarchies

When `assignment: "explicit"`, `users.csv` must carry either a `Path` column (`"2/5/3"`) or one column
per level named after `levels[].name`. This allows **ragged trees** — real org charts where team A has
3 people and team B has 30. Sequential assignment always yields a balanced tree; explicit does not.

`helper_scripts/generate_users.py` gains `--levels/--hierarchy-config` to emit these columns.

---

## 5. Code changes, file by file

### 5.1 New: `src/hierarchy.py` (pure logic, no torch, no model — must be unit-testable on CPU)

```python
ERASURE_SYMBOLS = frozenset({"⊥", "*", "?"})     # single source of truth

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
    path_dependent_labels: bool = False

    @property
    def L(self) -> int
    @property
    def depth(self) -> int
    @property
    def offsets(self) -> tuple[tuple[int, int], ...]   # (start, end) slice per level
    def capacity(self) -> int                          # Π fanout
    def validate(self) -> None                         # raises ValueError with actionable message

    @classmethod
    def from_dict(cls, d: dict) -> "HierarchySpec"
    @classmethod
    def from_json(cls, path: str) -> "HierarchySpec"
    @classmethod
    def from_cli(cls, s: str, L: int | None = None) -> "HierarchySpec"
    @classmethod
    def legacy_two_level(cls, group_bits, user_bits, max_groups=None,
                         users_per_group=None, min_distance=2) -> "HierarchySpec"
    def to_dict(self) -> dict                          # for result JSON provenance

class Codebook:
    """One level's codebook. Lazy + cached. __getitem__(index) -> str of length `bits`."""
    def __init__(self, bits: int, min_distance: int, fanout: int)
    def __getitem__(self, index: int) -> str
    def __len__(self) -> int
    def all(self) -> list[str]                          # materialise (small levels / viz only)
    def verify_min_distance(self) -> bool               # test hook

class HierarchyIndex:
    """Path <-> row-index <-> codeword. Owns the codebooks."""
    def __init__(self, spec: HierarchySpec, num_users: int, paths: list[tuple] | None = None)
    def path_of_index(self, row_index: int) -> tuple[int, ...]
    def index_of_path(self, path: tuple[int, ...]) -> int
    def codeword_of_path(self, path) -> str
    def children(self, prefix: tuple[int, ...]) -> list[int]      # existing children only
    def users_under(self, prefix: tuple[int, ...]) -> list[int]   # row indices
    def node_label(self, level: int, index: int, prefix=()) -> str

@dataclass
class DecodeResult:
    path: tuple[int, ...] | None
    row_index: int | None
    per_level_candidates: list[list[tuple[tuple, int]]]  # (path, cum_distance) kept at each level
    per_level_margin: list[int | None]     # runner-up distance − best distance, None if unique
    cumulative_distance: int | None
    ties: list[tuple[int, ...]]
    containment_path: tuple[int, ...]      # deepest unambiguous prefix = predicted LCA
    containment_level: int

def decode_path(recovered: str, index: HierarchyIndex, *,
                beam_width: int = 0,      # 0 = keep all ties only
                margin: int = 0,
                erasures: frozenset = ERASURE_SYMBOLS) -> DecodeResult
```

**Codebook constructions:**

| `d` | Construction | Indexing rule (must be exact) |
|---|---|---|
| 1 | identity | `format(index, f"0{bits}b")` |
| 2 | even parity | `format((index << 1) \| parity(index), f"0{bits}b")` — reproduces today's group codes exactly |
| ≥3 | greedy max-min | port from `FingerprintingCode._generate_bch_codewords`, but **memoised per (bits, d)** and generated once, not per call |

> **Bit-order bug to fix while porting:** `FingerprintingCode._generate_bch_codewords` builds bit
> arrays LSB-first (`[(i >> b) & 1 for b in range(L)]`) while the Hi-DyPa lazy path uses MSB-first
> (`format(cw, '0Gb')`). Distance is invariant under bit reversal so no result is wrong, but the two
> paths emit *different strings* for the same index. `Codebook` must pick one canonical order
> (MSB-first, matching the `d=2` lazy path that all current experiments used) and assert it in tests.

### 5.2 New: `src/hierarchical_watermark.py` (or a new class in `src/watermark.py`)

```python
class HierarchicalMultiUserWatermarker(NaiveMultiUserWatermarker):
    def __init__(self, lbit_watermarker: LBitWatermarker, spec: HierarchySpec)
    def load_users(self, users_file: str) -> pd.DataFrame
    def get_codeword_for_user(self, user_id: int) -> str
    def embed(self, master_key, user_id, prompt, **kw) -> str
    def trace(self, master_key, text, **kw) -> list[dict]
    def trace_from_codeword(self, recovered: str) -> list[dict]
    def decode(self, recovered: str) -> DecodeResult      # richer, used by eval scripts
```

`trace_from_codeword` returns the existing dict shape **plus** new keys, so nothing downstream breaks:

```python
{
  "user_id": 42, "username": "42", "match_score_percent": 93.7,
  "group_id": 5,                      # == path[0], kept for backward compat
  "path": [5, 2, 10],
  "path_names": {"region": 5, "team": 2, "user": 10},
  "per_level_distance": [0, 1, 0],
  "cumulative_distance": 1,
  "containment_path": [5, 2],         # predicted LCA when the leaf is ambiguous
  "containment_level": 2
}
```

`HiDyPaMultiUserWatermarker` is **retained** and reimplemented as a thin subclass that builds the
legacy 2-level spec. It must keep exposing the attributes the eval scripts read directly:
`group_bits`, `user_bits`, `_num_groups`, `_users_per_group`, `group_to_users`,
`_get_group_codeword_str()`, `user_metadata`, `lbw`, `N`.

### 5.3 Rewritten: the five duplicated `decode_hi_dypa_user` copies

Delete the private copies in
`evaluate_hi_dypa_detection.py`, `evaluate_hi_dypa_robustness.py`, `evaluate_rewrite_attack.py`,
`evaluate_paraphrasing_attack.py`, `evaluate_synonym_attack.py` (and any in
`compare_collusion_resistance.py` / `evaluate_framing_attack.py`), and replace with a single import
from `src/hierarchy.py`. Keep a compatibility shim:

```python
def decode_hi_dypa_user(muw, recovered, true_user_id=None):
    """Deprecated 2-level wrapper over decode_path(); returns (group_id, user_id, true_group_id)."""
```

**This is the single highest-value change in the whole extension** — it is where the decode bugs kept
recurring, and a 5-way duplicate becomes a 6-way duplicate the moment depth is variable.

### 5.4 Performance fixes required (they block the scalability experiment)

Current `get_codeword_for_user` does `self.user_metadata[self.user_metadata['UserId'] == user_id]`
— an `O(N)` pandas scan — and it is called **inside the tracing loop for every candidate user**
([watermark.py:1214](../src/watermark.py#L1214)), plus `users_in_group.index(user_id)` is `O(f)`.
The new implementation must precompute:

- `user_id → row_index` dict (`O(1)` lookup)
- `row_index → path` (mixed radix, `O(D)`, no storage)
- `path → row_index` (`O(D)`)

Target tracing cost: `O(Σ_l f_l)` codeword comparisons, versus `O(N)` for naive. That ratio is the
claim the hierarchy exists to support, so it must be measured, not asserted.

### 5.5 CLI: `src/main_multiuser.py`

- Add `--scheme hierarchical` alongside `naive | grouped | hi_dypa`.
- Add `--hierarchy-config PATH` and `--levels SPEC`.
- Relax `--min-distance` from `choices=[2]` to `[1..bits]` now that `Codebook` handles `d≥3` lazily.
- `trace` output prints the full path with per-level match, e.g.
  ```
  Traced path: region=5 → team=2 → user=10   (User ID 42, cum. distance 1)
  Level margins: region +3, team +1, user +2
  ```
- When the leaf is ambiguous, print the containment node instead of a user list:
  ```
  Ambiguous at level 3. Colluders confined to region=5 / team=2 (18 users).
  ```

### 5.6 Helper scripts

| Script | Change |
|---|---|
| `helper_scripts/design_hierarchy.py` **(new)** | Given `N`, `D`, and optional per-level fanouts, propose the minimal `L` and a bit allocation; print the capacity table and the resulting distance profile. Answers "what do I put in the config?" |
| `helper_scripts/visualize_hierarchy.py` **(new)** | ASCII tree + optional Graphviz of nodes with their codewords; replaces/extends `visualize_groups.py` |
| `helper_scripts/generate_users.py` | `--levels/--hierarchy-config` → emit `Path` / per-level columns for explicit hierarchies |
| `helper_scripts/create_collusion_scenario.py` | `--share-level l` → pick colluders that share an ancestor at level `l` (needed for the LCA experiment) |
| `helper_scripts/compute_code_capacity.py` | Add a multi-level mode: given a spec, print per-level and total capacity |
| `helper_scripts/visualize_groups.py` | Keep as a 2-level alias of the new script |

---

## 6. Algorithms

### 6.1 Assignment (sequential, default)

```
row_index i  →  path (i_1..i_D) by mixed-radix, least-significant digit at the LEAF:
    i_D = i mod f_D
    i_{D-1} = (i // f_D) mod f_{D-1}
    ...
    i_1 = i // (f_2 · f_3 · ... · f_D)
```
For `D=2` this is exactly `divmod(i, users_per_group)` — verified equal to today's behaviour by test.
`i` is the row index in metadata sorted by `UserId`, **not** `UserId` itself (matches current code).

### 6.2 Encode

```
w(i) = C_1[i_1] ++ ... ++ C_D[i_D]
assert len(w) == L
```

### 6.3 Decode — coarse-to-fine beam

```
segments = [recovered[s:e] for (s, e) in spec.offsets]
frontier = [((), 0)]                                   # (path_prefix, cumulative_distance)

for l in 1..D:
    cand = []
    for (prefix, cd) in frontier:
        for child in index.children(prefix):           # only existing children (ragged-safe)
            label = index.node_label(l, child, prefix)
            dist  = sum(segments[l][j] != label[j]
                        for j in range(b_l)
                        if segments[l][j] not in ERASURE_SYMBOLS)
            cand.append((prefix + (child,), cd + dist))
    if not cand: return DecodeResult(path=None, ...)
    best = min(cd for _, cd in cand)
    kept = [c for c in cand if c[1] <= best + margin]
    frontier = sorted(kept, key=lambda c: c[1])[:beam_width] if beam_width else kept
    record per_level_candidates[l], per_level_margin[l]

leaves = frontier; ties = all leaves at minimum cumulative distance
containment_path = longest prefix shared by all tied leaves       # ← predicted LCA
```

**Parameters:**
- `beam_width=0, margin=0` (default) — keep every minimum-distance candidate at each level. For
  `D=2` this reduces **exactly** to the current two-stage trace; that equivalence is a test.
- `beam_width=∞, margin=∞` — exhaustive search over all `N` codewords; the accuracy upper bound and
  the cost baseline for experiment E5.
- Intermediate values trade accuracy for cost — this is a tunable worth a figure in the paper.

**Erasure handling:** positions in `ERASURE_SYMBOLS` are skipped (not counted as mismatches),
identical to today. If *every* position of a level's segment is erased, that level contributes
distance 0 to all children → the frontier fans out to all children, which is the correct semantics
(no information). Record this as `level_fully_erased[l] = True` for diagnostics.

### 6.4 Hierarchical collusion localisation (new capability)

When `k` users collude, `LBitWatermarker.detect` emits `*` where the colluders' bits disagree.
Because colluders that share an ancestor agree on every segment above their divergence point:

```
predicted LCA = deepest prefix at which decoding is still unambiguous
```

Report `containment_path` and `containment_level`. New metric `lca_accuracy` = fraction of trials
where `predicted LCA == true LCA of the colluder set`. Also report `containment_size` (users under
the predicted node) — a "we narrowed 1000 users to 16" number, which is the practical payoff of
depth and is far more defensible than claiming exact identification under collusion.

---

## 7. Backward-compatibility contract (non-negotiable)

1. `HiDyPaMultiUserWatermarker(lbw, group_bits=G, user_bits=U, ...)` keeps its exact constructor
   signature, attributes, and printed output.
2. For every `(G, U)` with `G+U = L`, `L ∈ {8, 10, 12}`, and every user, the new code must produce a
   **string-identical** codeword to the current code. Enforced by a golden-file test (§9).
3. `trace_from_codeword` on 2-level configs must return the same accused set as today for a corpus of
   synthetic recovered strings (clean, erased, collided). Enforced by a differential test.
4. Existing slurm scripts and their `--group-bits/--user-bits` flags run unchanged.
5. Existing output directory naming (`hi_dypa_G4_U4`) is preserved when legacy flags are used. New
   naming `hier_L12_D3_b3-4-5` applies only to `--levels/--hierarchy-config` runs, so old and new
   results never collide in `evaluation/`.

---

## 8. The capacity/robustness trade-off (read before choosing experiments)

This is the part that decides whether the extension is a positive or negative result, so it should be
stated explicitly in the paper, not discovered late.

**`L` is fixed by robustness, not by hierarchy.** Adding levels does not add bits; it *partitions the
same `L`*. Two consequences:

1. **Capacity falls with depth at fixed `L`.** Each non-leaf level with `d=2` spends one bit on parity:

   | Config (L=12) | Depth | Capacity |
   |---|---|---|
   | `L=12` flat (naive) | 1 | 4096 |
   | `G=6, U=6` | 2 | `2^5 · 2^6` = 2048 |
   | `4/4/4` (d=2,2,1) | 3 | `2^3 · 2^3 · 2^4` = 1024 |
   | `3/3/3/3` (d=2,2,2,1) | 4 | `2^2 · 2^2 · 2^2 · 2^3` = 512 |

   Each extra level costs a factor of 2 in capacity (one parity bit). This is the price of the
   per-level distance guarantee.

2. **Segments get shorter, so single-bit damage hurts more.** A 3-bit level segment is destroyed by
   one erasure far more easily than a 6-bit one. Beam search (`margin > 0`) is the mitigation and
   should be swept.

3. **Raising `L` to compensate is not free.** `LBitLogitProcessor` cycles a permutation of the `L`
   bit positions across high-entropy blocks, so blocks-per-bit ≈ `blocks / L`. Larger `L` → fewer
   blocks per bit → lower per-bit z-scores → more `⊥`/`*`. **Instrument `blocks / L` and report it**;
   it is the mechanism behind any accuracy drop at larger `L`.

**Expected honest finding:** depth buys *tracing cost* (`O(Σ f_l)` vs `O(N)`) and *collusion
containment* (LCA localisation), and costs *capacity* and *per-level margin*, at fixed `L`. Design
the experiments to measure exactly that rather than to show "deeper is better".

---

## 9. Tests (`tests/`, new — pytest, CPU-only, no model download)

There is currently no test directory. Everything below runs on synthetic codeword strings with a
mocked/absent language model, so it runs in seconds locally and in CI.

**Spec & codebook**
- `test_spec_validation` — `Σ bits ≠ L`, `fanout > capacity`, `d > bits`, `D = 0` all raise with clear messages
- `test_codebook_min_distance` — for `bits ∈ 2..10`, `d ∈ 1..4`: every pair meets `d`
- `test_codebook_capacity` — `len(Codebook) == capacity(bits, d)`
- `test_codebook_determinism` — same index → same string across instances/processes
- `test_codebook_bit_order` — `d=2` codebook matches today's `_generate_single_group_codeword_int`

**Index**
- `test_path_index_roundtrip` — `index_of_path(path_of_index(i)) == i` for all `i` in a small hierarchy
- `test_ragged_children` — explicit assignment with uneven fanouts; `children()` returns only existing nodes

**Backward compatibility**
- `test_legacy_codeword_identical` — for `L ∈ {8,10,12}` × all `(G,U)` splits × all users: new codeword == old codeword (golden JSON committed)
- `test_legacy_trace_identical` — differential trace over a synthetic corpus of recovered strings

**Decode**
- `test_decode_clean` — every user's own codeword decodes to that user, all depths `D ∈ 1..4`
- `test_decode_erasures` — inject `k` erasures at chosen levels; assert prefix accuracy degrades monotonically and never crashes
- `test_decode_flips` — flip `< d_l/2` bits in level `l` → level `l` still correct
- `test_decode_full_erasure_level` — a fully-erased level fans out to all children, deeper levels still scored
- `test_beam_equivalence` — `beam_width=∞` == exhaustive nearest-codeword search
- `test_decode_matches_two_stage` — `D=2, beam=0, margin=0` == current `trace_from_codeword`

**Collusion**
- `test_lca_localisation` — build `*` strings from `k` colluders sharing a known ancestor; assert `containment_path` == true LCA
- `test_containment_size` — reported containment size == number of users under that node

**CLI**
- `test_cli_levels_parsing`, `test_cli_config_file`, `test_cli_legacy_flags` — smoke tests, no model

---

## 10. Evaluation & metrics changes

### 10.1 New metrics (per config, per attack)

| Metric | Definition |
|---|---|
| `level_accuracy[l]` | fraction where level `l`'s digit is correct (marginal) |
| `conditional_level_accuracy[l]` | correct at `l` **given** all ancestors correct |
| `prefix_accuracy[l]` | levels `1..l` all correct (this is the coarse-to-fine curve — the headline figure) |
| `full_path_accuracy` | `prefix_accuracy[D]`; equals today's `full_identity_accuracy` |
| `containment_accuracy` | true user is under the reported containment node |
| `containment_size_mean` | mean users under the reported node (search-space reduction) |
| `lca_accuracy` | collusion only: predicted LCA == true LCA |
| `candidates_evaluated` | codeword comparisons per trace (cost metric for E5) |
| `blocks_per_bit` | `block_count / L` (explains accuracy at large `L`) |

Keep every existing metric key so old analysis notebooks keep working.

### 10.2 Result JSON

Add a `hierarchy` provenance block to every summary:

```json
"hierarchy": {"L": 12, "depth": 3,
              "levels": [{"name":"region","bits":3,"fanout":4,"min_distance":2}, ...],
              "capacity": 1024, "assignment": "sequential",
              "beam_width": 0, "margin": 0}
```

### 10.3 Scripts to update

All of `evaluation_scripts/*.py` need: `--levels/--hierarchy-config` args, the shared decode import,
and per-level metric emission. Order of migration: `evaluate_hi_dypa_detection.py` first (simplest,
validates the whole pipeline), then `evaluate_hi_dypa_robustness.py`, then the three attack scripts,
then `compare_collusion_resistance.py` (needs the LCA metrics), then
`evaluate_multiuser_performance.py` (needs `candidates_evaluated`).

---

## 11. Experiment plan (new tables)

| ID | Question | Setup |
|---|---|---|
| **E1** Depth sweep | How does accuracy degrade with depth at fixed `L`? | `L=12`, `D ∈ {1,2,3,4}`, balanced splits, 300 prompts, DeepSeek-7B |
| **E2** Bit allocation | Where should the bits go at fixed `D=3, L=12`? | `(2,4,6)`, `(4,4,4)`, `(6,4,2)`, `(6,3,3)` |
| **E3** Robustness vs depth | Do deeper hierarchies survive paraphrase/synonym/rewrite/deletion? | E1 configs × existing 16 attack variants |
| **E4** Collusion containment | Does depth localise colluders? | colluders sharing an ancestor at level `1..D`; report `lca_accuracy`, `containment_size` |
| **E5** Tracing cost | Is `O(Σ f_l)` real? | `candidates_evaluated` and wall-clock vs `N ∈ {10², 10³, 10⁴}`, hierarchical vs naive |
| **E6** Beam sweep | Does `margin > 0` recover the accuracy lost to short segments? | `margin ∈ {0,1,2}`, `beam ∈ {0,4,16}` on the E1 configs |

New slurm scripts: `run_hierarchy_depth_sweep_deepseek_hpc.sh`, `run_hierarchy_collusion_deepseek_hpc.sh`,
following the existing Apptainer/`oz411`/`milan-gpu` template in `slurm_scripts/`.

---

## 12. Implementation phases (each phase independently reviewable & mergeable)

| Phase | Deliverable | Acceptance criteria |
|---|---|---|
| **0** | `src/hierarchy.py` skeleton + `tests/` + shared `decode_hi_dypa_user` shim; eval scripts import it instead of their local copies | All existing eval scripts produce identical output on a saved sample; 5 duplicates deleted |
| **1** | `HierarchySpec`, `Codebook`, `HierarchyIndex` + unit tests | All §9 spec/codebook/index tests pass |
| **2** | `decode_path` + beam + LCA + tests | `test_decode_matches_two_stage` and `test_beam_equivalence` pass |
| **3** | `HierarchicalMultiUserWatermarker`; `HiDyPaMultiUserWatermarker` reimplemented on top of it | Golden-file backward-compat tests pass; end-to-end generate+trace works with `gpt2` locally |
| **4** | CLI: `main_multiuser.py` `--levels/--hierarchy-config`, path-aware trace output | Manual 3-level generate/trace demo on `gpt2` |
| **5** | Eval script migration + per-level metrics + result JSON provenance | `evaluate_hi_dypa_detection.py` reproduces a previous 2-level run's numbers exactly, and runs a 3-level config |
| **6** | Helper scripts (`design_hierarchy`, `visualize_hierarchy`), collusion scenario generator | 3-level tree renders; capacity advisor matches hand calculation |
| **7** | Slurm scripts, README section, migration notes | Jobs submit on OzSTAR; README documents config format and the §8 trade-off |
| **8** *(optional)* | `path_dependent_labels` + its evaluation | Splicing-collusion comparison figure |

Phases 0–3 have no user-visible behaviour change; phases 4+ add features. **Phase 0 is worth doing
even if the rest is deferred** — it removes the duplicate-decode class of bug permanently.

---

## 13. Issues found in the current code (fix while extending)

1. **5× duplicated decode logic** across eval scripts — the recurring source of decode bugs. Phase 0.
2. **Inconsistent bit ordering** between `FingerprintingCode._generate_bch_codewords` (LSB-first) and
   the Hi-DyPa lazy `d=2` path (MSB-first). Harmless today (`d>2` is unused for Hi-DyPa) but a trap
   the moment `d≥3` levels are enabled. §5.1.
3. **Inconsistent erasure symbol sets**: `NaiveMultiUserWatermarker._match_users_from_codeword`
   filters `('⊥','*')`; the Hi-DyPa paths filter `('⊥','*','?')`. Centralise as `ERASURE_SYMBOLS`.
4. **`O(N)` pandas lookup inside the trace loop** ([watermark.py:1032](../src/watermark.py#L1032),
   called from [watermark.py:1214](../src/watermark.py#L1214)). Fix before running E5. §5.4.
5. **Repeated truncation logic** in `HiDyPaMultiUserWatermarker.load_users` — capacity is re-checked
   and users re-truncated four times ([watermark.py:857-936](../src/watermark.py#L857-L936)) with
   overlapping conditions. Replace with one `spec.capacity()` check.
6. **`FingerprintingCode._generate_bch_codewords` is `O(2^L)` eager** and allocates a `2^L × L` array
   before selecting. Make it lazy/memoised per `(bits, d)` in `Codebook`.
7. **`ZeroBitWatermarker.parse_first_block`** calls `tokenizer.encode(prefix, return_tensor='pt')` —
   typo (`return_tensor`, missing `s`) and unused code path. Fix or delete.
8. **No tests at all.** Phase 0/1 fixes this.

---

## 14. Decisions I need you to confirm before coding

| # | Question | My recommendation |
|---|---|---|
| 1 | Keep the 2-level path bit-identical (golden tests), or allow it to change? | **Keep identical.** Your Table 2/3/4 results stay valid; the cost is a few tests. |
| 2 | Default `min_distance` for intermediate levels? | **`d=2`** for all non-leaf levels, `d=1` for the leaf — matches current Hi-DyPa. |
| 3 | Default decode mode? | **`beam_width=0, margin=0`** (keep all ties), i.e. exactly today's semantics generalised. Sweep `margin` in E6. |
| 4 | Support ragged/explicit hierarchies in v1, or balanced-only? | **Both**, but `sequential` (balanced) first in phase 3; `explicit` in phase 6. Explicit is what makes "designed by the user" real. |
| 5 | `path_dependent_labels`? | **Plumb it, default off, evaluate later** (phase 8). It is a separate research claim. |
| 6 | Should `L` grow with depth in the experiments? | Run **both**: fixed `L=12` (capacity trade-off) and fixed-capacity-varying-`L` (robustness trade-off). §8 explains why. |
| 7 | New module `src/hierarchy.py` + `src/hierarchical_watermark.py`, or all inside `src/watermark.py`? | **Separate modules.** `watermark.py` is already 1247 lines and mixes torch with pure logic; the hierarchy code must be importable without torch so tests run on CPU in seconds. |
| 8 | Naming for the new scheme? | `--scheme hierarchical`, class `HierarchicalMultiUserWatermarker`, configs under `config/hierarchies/`. `hi_dypa` stays as the 2-level alias. |

---

## 15. Summary of new/changed files

**New**
```
src/hierarchy.py                          # spec, codebook, index, decode  (no torch)
src/hierarchical_watermark.py             # HierarchicalMultiUserWatermarker
config/hierarchies/*.json                 # example configs (2/3/4-level)
helper_scripts/design_hierarchy.py
helper_scripts/visualize_hierarchy.py
tests/                                    # ~25 pytest tests, CPU-only
slurm_scripts/run_hierarchy_depth_sweep_deepseek_hpc.sh
slurm_scripts/run_hierarchy_collusion_deepseek_hpc.sh
docs/MULTI_LEVEL_EXTENSION.md             # this file
```

**Changed**
```
src/watermark.py                          # HiDyPa reimplemented on hierarchy.py; perf fixes
src/fingerprinting.py                     # codebook generation moved out / delegated
src/main_multiuser.py                     # --levels, --hierarchy-config, path-aware trace output
evaluation_scripts/*.py                   # shared decode, --levels, per-level metrics
helper_scripts/generate_users.py          # emit hierarchical path columns
helper_scripts/create_collusion_scenario.py  # --share-level
helper_scripts/compute_code_capacity.py   # multi-level capacity mode
helper_scripts/visualize_groups.py        # alias to visualize_hierarchy
README.md                                 # config format, §8 trade-off, new commands
```
