# hierarchical_watermark.py: multi-level (D >= 1) multi-user watermarking.
#
# Additive: nothing in watermark.py is modified, so the existing naive/grouped/
# hi_dypa schemes and every result already collected are untouched.

from __future__ import annotations

import os

import pandas as pd

from .hierarchy import (
    DecodeResult,
    HierarchyIndex,
    HierarchySpec,
    TableDecoder,
    decode_path,
)
from .watermark import NaiveMultiUserWatermarker


class HierarchicalMultiUserWatermarker(NaiveMultiUserWatermarker):
    """
    Multi-user watermarking over an arbitrary-depth hierarchy.

    A user's codeword is the concatenation of one codeword per level, selected by
    the user's path through the tree. Tracing walks the tree coarse-to-fine, so
    the search cost is O(sum of fanouts) instead of O(N) -- and with
    use_tables=True it is one table lookup per level.
    """

    def __init__(self, lbit_watermarker, spec: HierarchySpec, use_tables: bool = True):
        super().__init__(lbit_watermarker=lbit_watermarker)
        spec.validate()
        if spec.L != self.lbw.L:
            raise ValueError(
                f"Hierarchy spans {spec.L} bits but the L-bit watermarker is "
                f"configured for L={self.lbw.L}. They must match."
            )
        self.spec = spec
        self.use_tables = use_tables
        self.index: HierarchyIndex | None = None
        self._decoder: TableDecoder | None = None
        self._row_of_user: dict[int, int] = {}
        self._user_of_row: list[int] = []

    # -- loading -----------------------------------------------------------

    def load_users(self, users_file: str) -> pd.DataFrame:
        if not os.path.exists(users_file):
            raise FileNotFoundError(f"User metadata file {users_file} not found")
        df = pd.read_csv(users_file)
        return self.load_users_frame(df)

    def load_users_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if "UserId" not in df.columns:
            raise ValueError("users_file must contain a 'UserId' column.")
        df = df.copy()
        df["UserId"] = df["UserId"].astype(int)
        if df["UserId"].duplicated().any():
            raise ValueError("Duplicate UserId entries detected in users file.")
        df = df.sort_values("UserId").reset_index(drop=True)

        capacity = self.spec.capacity()
        if len(df) > capacity:
            print(
                f"Warning: users file has {len(df)} rows but the hierarchy capacity is "
                f"{capacity}; truncating to {capacity}."
            )
            df = df.head(capacity)

        self.user_metadata = df
        self.N = len(df)
        self.user_lookup = {int(row["UserId"]): row for _, row in df.iterrows()}
        self._user_of_row = [int(v) for v in df["UserId"].tolist()]
        self._row_of_user = {user_id: row for row, user_id in enumerate(self._user_of_row)}

        self.index = HierarchyIndex(self.spec, self.N)
        self._decoder = TableDecoder(self.index) if self.use_tables else None

        names = " / ".join(
            f"{level.name}:{level.bits}b x{level.fanout} (d={level.min_distance})"
            for level in self.spec.levels
        )
        print(f"Loaded {self.N} users into a {self.spec.depth}-level hierarchy: {names}")
        print(f"  L = {self.spec.L} bits, capacity = {capacity}")
        return self.user_metadata

    def _require_index(self):
        if self.index is None:
            raise ValueError("Users not loaded. Call load_users(...) first.")

    # -- encoding ----------------------------------------------------------

    def path_for_user(self, user_id: int) -> tuple[int, ...]:
        self._require_index()
        if user_id not in self._row_of_user:
            raise ValueError(f"User ID {user_id} not found in loaded metadata.")
        return self.index.path_of_row(self._row_of_user[user_id])

    def get_codeword_for_user(self, user_id: int) -> str:
        self._require_index()
        if user_id not in self._row_of_user:
            raise ValueError(f"User ID {user_id} not found in loaded metadata.")
        return self.index.codeword_of_row(self._row_of_user[user_id])

    def _log_embed(self, user_id: int, codeword: str):
        path = self.path_for_user(user_id)
        parts = [
            f"{level.name}={digit}" for level, digit in zip(self.spec.levels, path)
        ]
        print(f"User ID {user_id} path: {' -> '.join(parts)}")
        print(f"Embedding hierarchical codeword '{codeword}' for User ID {user_id}...")

    # -- decoding ----------------------------------------------------------

    def decode(self, recovered: str, *, margin: int = 0) -> DecodeResult:
        self._require_index()
        if self._decoder is not None and margin == 0:
            return self._decoder.decode(recovered)
        return decode_path(recovered, self.index, margin=margin)

    def trace_from_codeword(self, recovered: str) -> list[dict]:
        self._require_index()
        if len(recovered) != self.spec.L:
            print(
                f"Warning: recovered codeword length ({len(recovered)}) != L ({self.spec.L})"
            )
            return []

        result = self.decode(recovered)
        if not result.ties:
            print("Could not match the recovered codeword to any user.")
            return []

        total = self.spec.L
        accused = []
        for path in result.ties:
            row = self.index.row_of_path(path)
            if row >= self.N:
                continue
            user_id = self._user_of_row[row]
            meta = self.user_lookup.get(user_id)
            distance = result.cumulative_distance or 0
            accused.append(
                {
                    "user_id": user_id,
                    "username": meta.get("Username") if meta is not None else None,
                    "match_score_percent": ((total - distance) / total * 100) if total else 0.0,
                    # path[0] is the top-level container; kept under the legacy key
                    # so existing reporting code keeps working.
                    "group_id": path[0],
                    "path": list(path),
                    "path_names": {
                        level.name: digit for level, digit in zip(self.spec.levels, path)
                    },
                    "cumulative_distance": distance,
                    "containment_path": list(result.containment_path),
                    "containment_level": result.containment_level,
                }
            )

        if len(accused) == 1:
            entry = accused[0]
            trail = " -> ".join(f"{k}={v}" for k, v in entry["path_names"].items())
            print(f"Traced path: {trail}  (User ID {entry['user_id']}, "
                  f"cumulative distance {entry['cumulative_distance']})")
        else:
            depth = result.containment_level
            if depth == 0:
                print(f"Ambiguous at every level: {len(accused)} candidate users.")
            else:
                node = " -> ".join(
                    f"{self.spec.levels[i].name}={d}"
                    for i, d in enumerate(result.containment_path)
                )
                contained = len(self.index.rows_under(result.containment_path))
                print(f"Ambiguous below level {depth}. Confined to {node} "
                      f"({contained} users, was {self.N}).")
        return accused

    def trace(self, master_key: bytes, text: str, **kwargs) -> list[dict]:
        self._require_index()
        print("Extracting L-bit codeword from text...")
        recovered = self.lbw.detect(master_key, text, **kwargs)
        print(f"  - Recovered Codeword: {recovered}")
        return self.trace_from_codeword(recovered)

    def describe(self) -> dict:
        return {
            "scheme": "hierarchical",
            "hierarchy": self.spec.to_dict(),
            "num_users": self.N,
            "decoder": "tables" if self._decoder is not None else "bitwise-scan",
        }
