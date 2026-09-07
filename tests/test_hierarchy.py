"""
Tests for the multi-level hierarchy layer.

CPU-only, no model download, no torch. Runs under pytest if it is installed, and
also standalone:

    python tests/test_hierarchy.py
"""

import os
import random
import sys
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hierarchy import (  # noqa: E402
    ERASURE_SYMBOLS,
    EXTENDED_HAMMING_8_4_4,
    HierarchyIndex,
    HierarchySpec,
    LevelSpec,
    TableDecoder,
    codebook_capacity,
    decode_path,
    hamming,
    load_spec,
    make_codebook,
    parse_segment,
)

OPTION_C3 = HierarchySpec.from_levels([
    LevelSpec("container", 8, 16, 4),
    LevelSpec("team", 4, 8, 2),
    LevelSpec("user", 4, 8, 2),
])
N_USERS = 1000


# ------------------------------------------------------------------ spec / codebook

def test_spec_validation():
    # fanout beyond what the distance allows
    try:
        HierarchySpec.from_levels([LevelSpec("a", 8, 32, 4)])
        raise AssertionError("expected ValueError for fanout > capacity")
    except ValueError:
        pass
    # min_distance wider than the segment
    try:
        HierarchySpec.from_levels([LevelSpec("a", 4, 2, 9)])
        raise AssertionError("expected ValueError for d > bits")
    except ValueError:
        pass
    # declared L disagreeing with the level widths
    try:
        HierarchySpec.from_dict({"L": 99, "levels": [{"bits": 8}, {"bits": 8}]})
        raise AssertionError("expected ValueError for L mismatch")
    except ValueError:
        pass


def test_codebook_capacity():
    assert codebook_capacity(8, 1) == 256
    assert codebook_capacity(8, 2) == 128
    assert codebook_capacity(8, 4) == 16
    assert codebook_capacity(4, 2) == 8


def test_codebook_min_distance():
    for bits in range(2, 9):
        for d in range(1, 5):
            if d > bits:
                continue
            fanout = codebook_capacity(bits, d)
            if fanout < 2:
                continue
            book = make_codebook(bits, d, fanout)
            assert book.achieved_min_distance() >= d, (bits, d)
            assert len(set(book.words())) == fanout
            assert all(len(w) == bits for w in book.words())


def test_layer1_is_extended_hamming():
    book = make_codebook(8, 4, 16)
    assert len(book) == 16
    assert book.achieved_min_distance() == 4
    assert tuple(book.words()) == EXTENDED_HAMMING_8_4_4
    assert sorted({w.count("1") for w in book.words()}) == [0, 4, 8]


def test_layer23_even_parity():
    book = make_codebook(4, 2, 8)
    assert len(book) == 8
    assert book.achieved_min_distance() == 2
    assert book.words() == ["0000", "0011", "0101", "0110", "1001", "1010", "1100", "1111"]
    assert all(w.count("1") % 2 == 0 for w in book.words())


def test_even_parity_matches_legacy_rule():
    """The d=2 codebook must reproduce watermark.py::_generate_single_group_codeword_int."""
    for bits in (4, 6, 8):
        book = make_codebook(bits, 2, codebook_capacity(bits, 2))
        for i in range(len(book)):
            base_parity = bin(i).count("1") % 2
            legacy = (i << 1) | base_parity
            assert book[i] == format(legacy, f"0{bits}b"), (bits, i)


def test_codebook_raises_when_short():
    try:
        make_codebook(8, 4, 17)  # only 16 exist
        raise AssertionError("expected ValueError for over-requested fanout")
    except ValueError:
        pass


# ------------------------------------------------------------------------- index

def test_capacity_and_uniqueness():
    assert OPTION_C3.capacity() == 1024
    assert OPTION_C3.L == 16
    index = HierarchyIndex(OPTION_C3, N_USERS)
    words = {index.codeword_of_row(r) for r in range(N_USERS)}
    assert len(words) == N_USERS


def test_path_row_roundtrip():
    index = HierarchyIndex(OPTION_C3, N_USERS)
    for row in range(N_USERS):
        assert index.row_of_path(index.path_of_row(row)) == row


def test_known_paths():
    index = HierarchyIndex(OPTION_C3, N_USERS)
    assert index.path_of_row(0) == (0, 0, 0)
    assert index.path_of_row(8) == (0, 1, 0)
    assert index.path_of_row(64) == (1, 0, 0)
    assert index.path_of_row(500) == (7, 6, 4)
    assert index.path_of_row(999) == (15, 4, 7)


def test_children_only_occupied():
    index = HierarchyIndex(OPTION_C3, N_USERS)
    assert index.children_count(()) == 16
    assert index.children_count((0,)) == 8
    assert index.children_count((15,)) == 5      # container 15 holds 40 users -> 5 teams
    assert index.children_count((15, 4)) == 8    # rows 992..999
    assert len(index.rows_under((15,))) == 40


def test_mixed_radix_matches_legacy_divmod():
    """Depth 2 must reduce to the original group_id = row // users_per_group."""
    spec = HierarchySpec.from_levels([
        LevelSpec("group", 8, 128, 2),
        LevelSpec("user", 8, 256, 1),
    ])
    index = HierarchyIndex(spec, N_USERS)
    for row in range(N_USERS):
        assert index.path_of_row(row) == divmod(row, 256)


def test_distance_profile():
    index = HierarchyIndex(OPTION_C3, N_USERS)
    cw = [index.codeword_of_row(r) for r in range(128)]
    same_team = min(hamming(cw[a], cw[b]) for a, b in combinations(range(8), 2))
    same_container = min(hamming(cw[a], cw[b]) for a in range(8) for b in range(8, 16))
    diff_container = min(hamming(cw[a], cw[b]) for a in range(64) for b in range(64, 128))
    assert same_team == 2
    assert same_container == 2
    assert diff_container == 4


# ------------------------------------------------------------------------ decode

def _flip(word: str, position: int) -> str:
    return word[:position] + ("1" if word[position] == "0" else "0") + word[position + 1:]


def _erase(word: str, positions) -> str:
    chars = list(word)
    for p in positions:
        chars[p] = "⊥"
    return "".join(chars)


def test_parse_segment():
    assert parse_segment("0000") == (0, 0)
    assert parse_segment("1010") == (0b1010, 0)
    assert parse_segment("1⊥10") == (0b1010, 0b0100)
    assert parse_segment("****") == (0, 0b1111)


def test_decode_clean():
    index = HierarchyIndex(OPTION_C3, N_USERS)
    for row in range(N_USERS):
        result = decode_path(index.codeword_of_row(row), index)
        assert result.row == row, row
        assert result.cumulative_distance == 0


def test_layer1_corrects_one_flip():
    index = HierarchyIndex(OPTION_C3, N_USERS)
    for row in (0, 64, 500, 999):
        base = index.codeword_of_row(row)
        container = index.path_of_row(row)[0]
        for position in range(8):
            result = decode_path(_flip(base, position), index)
            assert result.ties, (row, position)
            assert all(p[0] == container for p in result.ties), (row, position)


def test_layer1_corrects_three_erasures():
    index = HierarchyIndex(OPTION_C3, N_USERS)
    base = index.codeword_of_row(500)
    for positions in combinations(range(8), 3):
        result = decode_path(_erase(base, positions), index)
        assert result.ties
        assert all(p[0] == 7 for p in result.ties), positions


def test_layer2_corrects_one_erasure():
    index = HierarchyIndex(OPTION_C3, N_USERS)
    base = index.codeword_of_row(500)
    for position in range(8, 12):
        result = decode_path(_erase(base, [position]), index)
        assert result.row == 500, position


def test_full_erasure_of_top_level_fans_out():
    index = HierarchyIndex(OPTION_C3, N_USERS)
    base = index.codeword_of_row(500)
    result = decode_path(_erase(base, range(8)), index)
    # Row 500 is (7, 6, 4). With layer 1 fully erased every container ties at
    # level 1, but container 15 holds only 5 teams, so team 6 does not exist
    # there and it is eliminated at level 2. Ragged handling is what makes the
    # answer 15 rather than 16.
    containers = {p[0] for p in result.ties}
    assert containers == set(range(15)), sorted(containers)
    assert result.containment_level == 0
    assert index.children_count((15,)) == 5


def test_containment_on_team_collusion():
    """Two users in the same team differ only in the layer-3 segment."""
    index = HierarchyIndex(OPTION_C3, N_USERS)
    a = index.codeword_of_row(500)          # (7, 6, 4)
    b = index.codeword_of_row(501)          # (7, 6, 5)
    merged = "".join(x if x == y else "*" for x, y in zip(a, b))
    result = decode_path(merged, index)
    assert result.containment_path == (7, 6)
    assert result.containment_level == 2
    assert set(result.ties) <= {(7, 6, i) for i in range(8)}


def test_containment_on_container_collusion():
    index = HierarchyIndex(OPTION_C3, N_USERS)
    a = index.codeword_of_row(448)          # container 7, team 0
    b = index.codeword_of_row(500)          # container 7, team 6
    merged = "".join(x if x == y else "*" for x, y in zip(a, b))
    result = decode_path(merged, index)
    assert result.containment_path == (7,)
    assert result.containment_level == 1


def test_tables_match_scan():
    index = HierarchyIndex(OPTION_C3, N_USERS)
    decoder = TableDecoder(index)
    rng = random.Random(20260908)
    alphabet = "01⊥*"
    for _ in range(4000):
        row = rng.randrange(N_USERS)
        word = list(index.codeword_of_row(row))
        for _ in range(rng.randrange(0, 6)):
            word[rng.randrange(16)] = rng.choice(alphabet)
        corrupted = "".join(word)
        a = decode_path(corrupted, index)
        b = decoder.decode(corrupted)
        assert sorted(a.ties) == sorted(b.ties), corrupted
        assert a.cumulative_distance == b.cumulative_distance
        assert a.containment_path == b.containment_path


def test_table_entries_match_bruteforce():
    """Every table entry must equal an exhaustive nearest-codeword search."""
    book = make_codebook(4, 2, 8)
    index = HierarchyIndex(
        HierarchySpec.from_levels([LevelSpec("only", 4, 8, 2)]), 8
    )
    table = TableDecoder(index).tables[0]
    for value in range(16):
        for erasure in range(16):
            keep = 0b1111 & ~erasure
            best = min((value ^ w) .bit_count() if False else bin((value ^ w) & keep).count("1")
                       for w in book.ints)
            mask = 0
            for child, w in enumerate(book.ints):
                if bin((value ^ w) & keep).count("1") == best:
                    mask |= 1 << child
            got_best, got_mask = table.lookup(value, erasure)
            assert got_best == best and got_mask == mask, (value, erasure)


def test_configs_load():
    for name in ("l16_8_4_4", "l16_8_8_optionC", "l16_8_8_legacy", "l16_flat"):
        spec = load_spec(name)
        assert spec.L == 16
    assert load_spec("container:8:16:4,user:8:64:2").depth == 2
    assert load_spec("8:16:4,4:8:2,4:8:2").depth == 3
    assert load_spec("l16_8_4_4").capacity() == 1024


def test_erasure_symbols_are_shared():
    assert {"⊥", "*", "?"} == set(ERASURE_SYMBOLS)


# ---------------------------------------------------------------------- standalone

def _run_all():
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
