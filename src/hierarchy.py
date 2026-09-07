# hierarchy.py: multi-level hierarchical codeword construction, indexing and decoding.
#
# Deliberately depends on the standard library only (no torch, no pandas) so the
# whole codeword/decode layer is unit-testable on CPU in milliseconds.

from __future__ import annotations

import json
import os
from array import array
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

# Symbols produced by LBitWatermarker.detect that carry no bit information.
# '⊥' = neither key detected, '*' = both detected (collusion), '?' = unknown.
ERASURE_SYMBOLS = frozenset({"⊥", "*", "?"})


if hasattr(int, "bit_count"):          # Python 3.10+
    def _popcount(value: int) -> int:
        return value.bit_count()
else:                                   # older interpreters (e.g. HPC login nodes)
    def _popcount(value: int) -> int:
        return bin(value).count("1")


def hamming(a: str, b: str) -> int:
    """Hamming distance between two equal-length strings."""
    return sum(x != y for x, y in zip(a, b))


# --------------------------------------------------------------------------- specs


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
    def L(self) -> int:
        return sum(level.bits for level in self.levels)

    @property
    def depth(self) -> int:
        return len(self.levels)

    @property
    def offsets(self) -> tuple[tuple[int, int], ...]:
        """(start, end) slice of the codeword string for each level."""
        out = []
        pos = 0
        for level in self.levels:
            out.append((pos, pos + level.bits))
            pos += level.bits
        return tuple(out)

    def capacity(self) -> int:
        total = 1
        for level in self.levels:
            total *= level.fanout
        return total

    def validate(self) -> None:
        if self.depth < 1:
            raise ValueError("Hierarchy must have at least one level.")
        if self.assignment not in ("sequential",):
            raise ValueError(f"Unsupported assignment mode {self.assignment!r}.")
        for i, level in enumerate(self.levels):
            if level.bits < 1:
                raise ValueError(f"Level {i} ({level.name}): bits must be >= 1, got {level.bits}.")
            if not 1 <= level.min_distance <= level.bits:
                raise ValueError(
                    f"Level {i} ({level.name}): min_distance must be in [1, {level.bits}], "
                    f"got {level.min_distance}."
                )
            available = codebook_capacity(level.bits, level.min_distance)
            if not 1 <= level.fanout <= available:
                raise ValueError(
                    f"Level {i} ({level.name}): fanout {level.fanout} exceeds what "
                    f"{level.bits} bits at min_distance {level.min_distance} can supply "
                    f"({available})."
                )

    # -- constructors ------------------------------------------------------

    @classmethod
    def from_levels(cls, levels, assignment: str = "sequential") -> "HierarchySpec":
        spec = cls(levels=tuple(levels), assignment=assignment)
        spec.validate()
        return spec

    @classmethod
    def from_dict(cls, data: dict) -> "HierarchySpec":
        levels = []
        depth = len(data["levels"])
        for i, raw in enumerate(data["levels"]):
            bits = int(raw["bits"])
            default_d = 2 if i < depth - 1 else 1
            min_distance = int(raw.get("min_distance", default_d))
            fanout = int(raw.get("fanout", codebook_capacity(bits, min_distance)))
            levels.append(
                LevelSpec(
                    name=str(raw.get("name", f"level{i + 1}")),
                    bits=bits,
                    fanout=fanout,
                    min_distance=min_distance,
                )
            )
        spec = cls(levels=tuple(levels), assignment=str(data.get("assignment", "sequential")))
        spec.validate()
        declared_L = data.get("L")
        if declared_L is not None and int(declared_L) != spec.L:
            raise ValueError(
                f"Config declares L={declared_L} but level widths sum to {spec.L}."
            )
        return spec

    @classmethod
    def from_json(cls, path: str) -> "HierarchySpec":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def from_cli(cls, text: str) -> "HierarchySpec":
        """Parse "name:bits:fanout:min_distance,..." or a bare "8,4,4"."""
        levels = []
        parts = [chunk.strip() for chunk in text.split(",") if chunk.strip()]
        depth = len(parts)
        for i, chunk in enumerate(parts):
            fields = chunk.split(":")
            if len(fields) == 1:
                name, bits = f"level{i + 1}", int(fields[0])
                min_distance = 2 if i < depth - 1 else 1
                fanout = codebook_capacity(bits, min_distance)
            elif len(fields) == 3:
                name = f"level{i + 1}"
                bits = int(fields[0])
                fanout = int(fields[1])
                min_distance = int(fields[2])
            elif len(fields) == 4:
                name = fields[0]
                bits = int(fields[1])
                fanout = int(fields[2])
                min_distance = int(fields[3])
            else:
                raise ValueError(
                    f"Cannot parse level spec {chunk!r}; expected "
                    "'name:bits:fanout:min_distance', 'bits:fanout:min_distance', "
                    "or a bare bit width."
                )
            levels.append(LevelSpec(name, bits, fanout, min_distance))
        spec = cls(levels=tuple(levels))
        spec.validate()
        return spec

    @classmethod
    def legacy_two_level(cls, group_bits: int, user_bits: int, max_groups: int | None = None,
                         users_per_group: int | None = None,
                         min_distance: int = 2) -> "HierarchySpec":
        """The pre-existing Hi-DyPa (group, user) layout expressed as a hierarchy."""
        group_fanout = max_groups if max_groups is not None else codebook_capacity(
            group_bits, min_distance
        )
        if user_bits == 0:
            spec = cls(levels=(LevelSpec("group", group_bits, group_fanout, min_distance),))
        else:
            user_fanout = users_per_group if users_per_group is not None else 2 ** user_bits
            spec = cls(
                levels=(
                    LevelSpec("group", group_bits, group_fanout, min_distance),
                    LevelSpec("user", user_bits, user_fanout, 1),
                )
            )
        spec.validate()
        return spec

    def to_dict(self) -> dict:
        return {
            "L": self.L,
            "depth": self.depth,
            "capacity": self.capacity(),
            "assignment": self.assignment,
            "levels": [
                {
                    "name": level.name,
                    "bits": level.bits,
                    "fanout": level.fanout,
                    "min_distance": level.min_distance,
                }
                for level in self.levels
            ],
        }


# ------------------------------------------------------------------------ codebooks

# Verified extended Hamming [8,4,4]: 16 codewords, minimum distance 4.
EXTENDED_HAMMING_8_4_4 = (
    "00000000", "00010111", "00101101", "00111010",
    "01001110", "01011001", "01100011", "01110100",
    "10001011", "10011100", "10100110", "10110001",
    "11000101", "11010010", "11101000", "11111111",
)

# (bits, min_distance) -> explicit codeword table for constructions we want to pin.
EXPLICIT_CODES: dict[tuple[int, int], tuple[str, ...]] = {
    (8, 4): EXTENDED_HAMMING_8_4_4,
}


@lru_cache(maxsize=None)
def _lexicode(bits: int, min_distance: int) -> tuple[int, ...]:
    """Greedy lexicographic code: keep w if it is >= min_distance from all kept words."""
    kept: list[int] = []
    for word in range(1 << bits):
        for chosen in kept:
            if _popcount(word ^ chosen) < min_distance:
                break
        else:
            kept.append(word)
    return tuple(kept)


@lru_cache(maxsize=None)
def codebook_capacity(bits: int, min_distance: int) -> int:
    """How many codewords of `bits` bits we can actually construct at `min_distance`."""
    if min_distance <= 1:
        return 1 << bits
    if min_distance == 2:
        return 1 << (bits - 1)
    key = (bits, min_distance)
    if key in EXPLICIT_CODES:
        return len(EXPLICIT_CODES[key])
    return len(_lexicode(bits, min_distance))


class Codebook:
    """
    One level's codebook: an injective map {0..fanout-1} -> {0,1}^bits with
    pairwise Hamming distance >= min_distance. Words are MSB-first strings; the
    integer view is what the decoder actually uses.
    """

    __slots__ = ("bits", "min_distance", "fanout", "ints", "mask")

    def __init__(self, bits: int, min_distance: int, fanout: int, words: tuple[int, ...]):
        if len(words) < fanout:
            raise ValueError(
                f"Codebook({bits} bits, d={min_distance}) can supply only {len(words)} "
                f"codewords but {fanout} were requested."
            )
        self.bits = bits
        self.min_distance = min_distance
        self.fanout = fanout
        self.ints = words[:fanout]
        self.mask = (1 << bits) - 1

    def __len__(self) -> int:
        return self.fanout

    def __getitem__(self, index: int) -> str:
        if not 0 <= index < self.fanout:
            raise IndexError(f"Codebook index {index} out of range [0, {self.fanout}).")
        return format(self.ints[index], f"0{self.bits}b")

    def words(self) -> list[str]:
        return [self[i] for i in range(self.fanout)]

    def achieved_min_distance(self) -> int:
        if self.fanout < 2:
            return self.bits
        return min(_popcount(a ^ b) for a, b in combinations(self.ints, 2))


def _identity_words(bits: int) -> tuple[int, ...]:
    return tuple(range(1 << bits))


def _even_parity_words(bits: int) -> tuple[int, ...]:
    # (i << 1) | parity(i) enumerates the even-parity words in ascending order.
    # This is bit-for-bit the rule used by the pre-existing 2-level Hi-DyPa code.
    return tuple((i << 1) | (_popcount(i) & 1) for i in range(1 << (bits - 1)))


@lru_cache(maxsize=None)
def make_codebook(bits: int, min_distance: int, fanout: int) -> Codebook:
    if min_distance <= 1:
        words = _identity_words(bits)
    elif min_distance == 2:
        words = _even_parity_words(bits)
    else:
        key = (bits, min_distance)
        if key in EXPLICIT_CODES:
            words = tuple(int(w, 2) for w in EXPLICIT_CODES[key])
        else:
            words = _lexicode(bits, min_distance)
    return Codebook(bits, min_distance, fanout, words)


# --------------------------------------------------------------------------- index


class HierarchyIndex:
    """Maps row index <-> path <-> codeword, and enumerates occupied children."""

    def __init__(self, spec: HierarchySpec, num_users: int):
        spec.validate()
        if num_users < 0:
            raise ValueError("num_users must be non-negative.")
        capacity = spec.capacity()
        if num_users > capacity:
            raise ValueError(
                f"num_users {num_users} exceeds hierarchy capacity {capacity}."
            )
        self.spec = spec
        self.num_users = num_users
        self.depth = spec.depth
        self.fanouts = [level.fanout for level in spec.levels]
        self.codebooks = [
            make_codebook(level.bits, level.min_distance, level.fanout)
            for level in spec.levels
        ]
        # weights[i] = number of leaves under one node at level i
        weights = [1] * self.depth
        for i in range(self.depth - 2, -1, -1):
            weights[i] = weights[i + 1] * self.fanouts[i + 1]
        self.weights = weights

    # -- paths -------------------------------------------------------------

    def path_of_row(self, row: int) -> tuple[int, ...]:
        if not 0 <= row < self.num_users:
            raise IndexError(f"Row {row} out of range [0, {self.num_users}).")
        path = []
        for i in range(self.depth):
            path.append((row // self.weights[i]) % self.fanouts[i])
        return tuple(path)

    def row_of_path(self, path: tuple[int, ...]) -> int:
        if len(path) != self.depth:
            raise ValueError(f"Path {path} must have length {self.depth}.")
        row = 0
        for i, digit in enumerate(path):
            if not 0 <= digit < self.fanouts[i]:
                raise ValueError(f"Path digit {digit} out of range at level {i}.")
            row += digit * self.weights[i]
        return row

    def codeword_of_path(self, path: tuple[int, ...]) -> str:
        return "".join(self.codebooks[i][digit] for i, digit in enumerate(path))

    def codeword_of_row(self, row: int) -> str:
        return self.codeword_of_path(self.path_of_row(row))

    def codeword_int_of_row(self, row: int) -> int:
        value = 0
        for i, digit in enumerate(self.path_of_row(row)):
            value = (value << self.spec.levels[i].bits) | self.codebooks[i].ints[digit]
        return value

    # -- tree navigation ---------------------------------------------------

    def _row_span(self, prefix: tuple[int, ...]) -> tuple[int, int]:
        """[start, end) rows covered by an existing node addressed by `prefix`."""
        start = 0
        for i, digit in enumerate(prefix):
            start += digit * self.weights[i]
        span = self.weights[len(prefix) - 1] if prefix else self.spec.capacity()
        return start, min(start + span, self.num_users)

    def children_count(self, prefix: tuple[int, ...]) -> int:
        """Number of *occupied* children of the node addressed by `prefix`."""
        level = len(prefix)
        if level >= self.depth:
            return 0
        start, end = self._row_span(prefix)
        if end <= start:
            return 0
        width = self.weights[level]
        return min(self.fanouts[level], -(-(end - start) // width))

    def rows_under(self, prefix: tuple[int, ...]) -> range:
        start, end = self._row_span(prefix)
        return range(start, max(start, end))

    def is_full(self, prefix: tuple[int, ...]) -> bool:
        level = len(prefix)
        return level < self.depth and self.children_count(prefix) == self.fanouts[level]


# -------------------------------------------------------------------------- decode


def parse_segment(segment: str) -> tuple[int, int]:
    """Return (value, erasure_mask) as MSB-first integers for one codeword segment."""
    value = 0
    erasure = 0
    for char in segment:
        value <<= 1
        erasure <<= 1
        if char == "1":
            value |= 1
        elif char != "0":
            erasure |= 1
    return value, erasure


@dataclass
class DecodeResult:
    path: tuple[int, ...] | None
    row: int | None
    ties: list[tuple[int, ...]]
    per_level_distance: list[int]
    per_level_margin: list[int | None]
    per_level_candidates: list[int]
    cumulative_distance: int | None
    containment_path: tuple[int, ...]
    containment_level: int
    candidates_evaluated: int

    @property
    def is_unique(self) -> bool:
        return len(self.ties) == 1


def _common_prefix(paths: list[tuple[int, ...]]) -> tuple[int, ...]:
    if not paths:
        return ()
    shared = []
    for column in zip(*paths):
        first = column[0]
        if all(value == first for value in column):
            shared.append(first)
        else:
            break
    return tuple(shared)


def _finalise(index: HierarchyIndex, frontier: list[tuple[tuple[int, ...], int]],
              per_level_candidates: list[int], per_level_margin: list[int | None],
              per_level_distance: list[int], evaluated: int) -> DecodeResult:
    if not frontier:
        return DecodeResult(None, None, [], per_level_distance, per_level_margin,
                            per_level_candidates, None, (), 0, evaluated)
    best = min(distance for _, distance in frontier)
    ties = [path for path, distance in frontier if distance == best]
    containment = _common_prefix(ties)
    unique_path = ties[0] if len(ties) == 1 else None
    return DecodeResult(
        path=unique_path,
        row=index.row_of_path(unique_path) if unique_path is not None else None,
        ties=ties,
        per_level_distance=per_level_distance,
        per_level_margin=per_level_margin,
        per_level_candidates=per_level_candidates,
        cumulative_distance=best,
        containment_path=containment,
        containment_level=len(containment),
        candidates_evaluated=evaluated,
    )


def decode_path(recovered: str, index: HierarchyIndex, *, beam_width: int = 0,
                margin: int = 0) -> DecodeResult:
    """
    Coarse-to-fine decode using bitwise distance (tier 1).

    beam_width = 0 keeps every minimum-distance candidate at each level, which
    reproduces the original two-stage Hi-DyPa trace when depth == 2.
    """
    spec = index.spec
    if len(recovered) != spec.L:
        raise ValueError(f"Recovered codeword has length {len(recovered)}, expected {spec.L}.")

    segments = [parse_segment(recovered[start:end]) for start, end in spec.offsets]
    frontier: list[tuple[tuple[int, ...], int]] = [((), 0)]
    per_level_candidates: list[int] = []
    per_level_margin: list[int | None] = []
    per_level_distance: list[int] = []
    evaluated = 0

    for level in range(index.depth):
        value, erasure = segments[level]
        codebook = index.codebooks[level]
        keep = codebook.mask & ~erasure
        candidates: list[tuple[tuple[int, ...], int]] = []
        for prefix, cumulative in frontier:
            words = codebook.ints
            for child in range(index.children_count(prefix)):
                distance = _popcount((value ^ words[child]) & keep)
                candidates.append((prefix + (child,), cumulative + distance))
                evaluated += 1
        if not candidates:
            return _finalise(index, [], per_level_candidates, per_level_margin,
                             per_level_distance, evaluated)

        best = min(distance for _, distance in candidates)
        runner_up = min((d for _, d in candidates if d > best), default=None)
        per_level_margin.append(None if runner_up is None else runner_up - best)
        per_level_distance.append(best - (min(d for _, d in frontier) if frontier else 0))

        kept = [item for item in candidates if item[1] <= best + margin]
        if beam_width:
            kept.sort(key=lambda item: item[1])
            kept = kept[:beam_width]
        frontier = kept
        per_level_candidates.append(len(frontier))

    return _finalise(index, frontier, per_level_candidates, per_level_margin,
                     per_level_distance, evaluated)


# ------------------------------------------------------------------ decode tables


class LevelDecodeTable:
    """
    Complete precomputed decode table for one level.

    Indexed by (value << bits) | erasure_mask, it stores the best achievable
    distance and a bitmask of every child attaining it. Feasible only because a
    level is short: the table has 4^bits entries.
    """

    __slots__ = ("bits", "fanout", "best", "mask")

    def __init__(self, codebook: Codebook):
        bits = codebook.bits
        size = 1 << (2 * bits)
        self.bits = bits
        self.fanout = codebook.fanout
        self.best = array("b", bytes(size))
        self.mask = [0] * size
        words = codebook.ints
        full = codebook.mask
        for erasure in range(1 << bits):
            keep = full & ~erasure
            base = erasure
            for value in range(1 << bits):
                best = bits + 1
                accumulated = 0
                for child in range(codebook.fanout):
                    distance = _popcount((value ^ words[child]) & keep)
                    if distance < best:
                        best = distance
                        accumulated = 1 << child
                    elif distance == best:
                        accumulated |= 1 << child
                slot = (value << bits) | base
                self.best[slot] = best
                self.mask[slot] = accumulated

    def lookup(self, value: int, erasure: int) -> tuple[int, int]:
        slot = (value << self.bits) | erasure
        return self.best[slot], self.mask[slot]


_TABLE_CACHE: dict[tuple[int, int, int], LevelDecodeTable] = {}


def get_decode_table(codebook: Codebook) -> LevelDecodeTable:
    key = (codebook.bits, codebook.min_distance, codebook.fanout)
    table = _TABLE_CACHE.get(key)
    if table is None:
        table = LevelDecodeTable(codebook)
        _TABLE_CACHE[key] = table
    return table


class TableDecoder:
    """
    Tier-3 decoder: one table lookup per level in the common case.

    A node whose children are not all present (the last, partially filled node at
    each level) falls back to a bitwise scan over its existing children, so the
    result is always identical to decode_path().
    """

    def __init__(self, index: HierarchyIndex):
        self.index = index
        self.tables = [get_decode_table(codebook) for codebook in index.codebooks]

    def decode(self, recovered: str, *, margin: int = 0) -> DecodeResult:
        index = self.index
        spec = index.spec
        if len(recovered) != spec.L:
            raise ValueError(
                f"Recovered codeword has length {len(recovered)}, expected {spec.L}."
            )

        segments = [parse_segment(recovered[start:end]) for start, end in spec.offsets]
        frontier: list[tuple[tuple[int, ...], int]] = [((), 0)]
        per_level_candidates: list[int] = []
        per_level_margin: list[int | None] = []
        per_level_distance: list[int] = []
        evaluated = 0

        for level in range(index.depth):
            value, erasure = segments[level]
            table = self.tables[level]
            codebook = index.codebooks[level]
            keep = codebook.mask & ~erasure
            candidates: list[tuple[tuple[int, ...], int]] = []

            for prefix, cumulative in frontier:
                n_children = index.children_count(prefix)
                if n_children == codebook.fanout:
                    best, bits_mask = table.lookup(value, erasure)
                    evaluated += 1
                    if margin == 0:
                        child = 0
                        while bits_mask:
                            if bits_mask & 1:
                                candidates.append((prefix + (child,), cumulative + best))
                            bits_mask >>= 1
                            child += 1
                        continue
                # partial node, or margin-widened search: scan its children
                words = codebook.ints
                for child in range(n_children):
                    distance = _popcount((value ^ words[child]) & keep)
                    candidates.append((prefix + (child,), cumulative + distance))
                    evaluated += 1

            if not candidates:
                return _finalise(index, [], per_level_candidates, per_level_margin,
                                 per_level_distance, evaluated)

            best = min(distance for _, distance in candidates)
            runner_up = min((d for _, d in candidates if d > best), default=None)
            per_level_margin.append(None if runner_up is None else runner_up - best)
            per_level_distance.append(best - (min(d for _, d in frontier) if frontier else 0))
            frontier = [item for item in candidates if item[1] <= best + margin]
            per_level_candidates.append(len(frontier))

        return _finalise(index, frontier, per_level_candidates, per_level_margin,
                         per_level_distance, evaluated)


# ------------------------------------------------------------------- config lookup

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "config", "hierarchies")


def load_spec(name_or_path: str) -> HierarchySpec:
    """Load a spec from a path, a name under config/hierarchies, or a --levels string."""
    if os.path.exists(name_or_path):
        return HierarchySpec.from_json(name_or_path)
    candidate = os.path.join(CONFIG_DIR, f"{name_or_path}.json")
    if os.path.exists(candidate):
        return HierarchySpec.from_json(candidate)
    return HierarchySpec.from_cli(name_or_path)
