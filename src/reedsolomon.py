
PRIMITIVE_POLY = {
    2: 0b111,
    3: 0b1011,
    4: 0b10011,
    5: 0b100101,
    6: 0b1000011,
    7: 0b10001001,
    8: 0b100011101,
}


class GaloisField:
    """Arithmetic in GF(2^m) using exp/log tables."""

    def __init__(self, m: int):
        if m not in PRIMITIVE_POLY:
            raise ValueError(f"Unsupported symbol size m={m} (supported: 2..8)")
        self.m = m
        self.order = 1 << m           # 2^m
        self.max = self.order - 1     # multiplicative group order

        prim = PRIMITIVE_POLY[m]
        self.exp = [0] * (2 * self.order)
        self.log = [0] * self.order

        x = 1
        for i in range(self.max):
            self.exp[i] = x
            self.log[x] = i
            x <<= 1
            if x & self.order:
                x ^= prim
        # Duplicate the table so exp[i + j] never needs a modulo.
        for i in range(self.max, 2 * self.order):
            self.exp[i] = self.exp[i - self.max]

    def mul(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return self.exp[self.log[a] + self.log[b]]

    def div(self, a: int, b: int) -> int:
        if b == 0:
            raise ZeroDivisionError("division by zero in GF(2^m)")
        if a == 0:
            return 0
        return self.exp[(self.log[a] - self.log[b]) % self.max]

    def inv(self, a: int) -> int:
        if a == 0:
            raise ZeroDivisionError("inverse of zero in GF(2^m)")
        return self.exp[(self.max - self.log[a]) % self.max]

    def power(self, a: int, n: int) -> int:
        if a == 0:
            return 0
        return self.exp[(self.log[a] * n) % self.max]

    # --- polynomial helpers (poly[0] is the highest-degree coefficient) ---

    def poly_mul(self, p: list[int], q: list[int]) -> list[int]:
        result = [0] * (len(p) + len(q) - 1)
        for i, pi in enumerate(p):
            if pi == 0:
                continue
            lp = self.log[pi]
            for j, qj in enumerate(q):
                if qj:
                    result[i + j] ^= self.exp[lp + self.log[qj]]
        return result

    def poly_add(self, p: list[int], q: list[int]) -> list[int]:
        size = max(len(p), len(q))
        result = [0] * size
        for i, pi in enumerate(p):
            result[i + size - len(p)] = pi
        for i, qi in enumerate(q):
            result[i + size - len(q)] ^= qi
        return result

    def poly_scale(self, p: list[int], x: int) -> list[int]:
        return [self.mul(c, x) for c in p]

    def poly_eval(self, p: list[int], x: int) -> int:
        """Horner's method."""
        y = p[0]
        for coeff in p[1:]:
            y = self.mul(y, x) ^ coeff
        return y

    def poly_mod(self, dividend: list[int], divisor: list[int]) -> list[int]:
        """Remainder of dividend / divisor (synthetic division)."""
        out = list(dividend)
        normaliser = divisor[0]
        for i in range(len(dividend) - len(divisor) + 1):
            coeff = out[i]
            if coeff == 0:
                continue
            coeff = self.div(coeff, normaliser)
            for j in range(1, len(divisor)):
                if divisor[j]:
                    out[i + j] ^= self.mul(divisor[j], coeff)
        separator = len(divisor) - 1
        return out[-separator:] if separator else []


class ReedSolomon:
    """
    Narrow-sense systematic Reed-Solomon code RS(n, k) over GF(2^m).

    Encodes k message symbols into n symbols and corrects up to
    t = (n - k) // 2 symbol errors.
    """

    def __init__(self, n: int, k: int, m: int, fcr: int = 1):
        gf = GaloisField(m)
        if not 0 < k < n <= gf.max:
            raise ValueError(
                f"Invalid RS parameters n={n}, k={k}, m={m}: require 0 < k < n <= {gf.max}"
            )
        self.gf = gf
        self.n = n
        self.k = k
        self.m = m
        self.fcr = fcr
        self.nsym = n - k
        self.t = self.nsym // 2
        self.generator = self._make_generator()

    def _make_generator(self) -> list[int]:
        g = [1]
        for i in range(self.nsym):
            g = self.gf.poly_mul(g, [1, self.gf.power(2, i + self.fcr)])
        return g

    def encode(self, message: list[int]) -> list[int]:
        """message: k symbols -> codeword: n symbols (systematic: message + parity)."""
        if len(message) != self.k:
            raise ValueError(f"Expected {self.k} message symbols, got {len(message)}")
        for s in message:
            if not 0 <= s < self.gf.order:
                raise ValueError(f"Symbol {s} out of range for GF(2^{self.m})")
        padded = list(message) + [0] * self.nsym
        remainder = self.gf.poly_mod(padded, self.generator)
        return list(message) + list(remainder)

    def _syndromes(self, codeword: list[int]) -> list[int]:
        return [
            self.gf.poly_eval(codeword, self.gf.power(2, i + self.fcr))
            for i in range(self.nsym)
        ]

    def _berlekamp_massey(self, syndromes: list[int]) -> list[int]:
        err_loc = [1]
        old_loc = [1]
        for i in range(self.nsym):
            delta = syndromes[i]
            for j in range(1, len(err_loc)):
                delta ^= self.gf.mul(err_loc[len(err_loc) - 1 - j], syndromes[i - j])
            old_loc = old_loc + [0]
            if delta != 0:
                if len(old_loc) > len(err_loc):
                    new_loc = self.gf.poly_scale(old_loc, delta)
                    old_loc = self.gf.poly_scale(err_loc, self.gf.inv(delta))
                    err_loc = new_loc
                err_loc = self.gf.poly_add(err_loc, self.gf.poly_scale(old_loc, delta))
        # Strip leading zeros
        while err_loc and err_loc[0] == 0:
            err_loc.pop(0)
        return err_loc

    def _chien_search(self, err_loc: list[int], n: int) -> list[int]:
        """Return error positions as indices into the codeword (0 = first symbol)."""
        errs = len(err_loc) - 1
        positions = []
        for i in range(n):
            if self.gf.poly_eval(err_loc, self.gf.power(2, self.gf.max - i)) == 0:
                positions.append(n - 1 - i)
        if len(positions) != errs:
            return []
        return positions

    def _forney(self, syndromes: list[int], err_loc: list[int], positions: list[int],
                n: int) -> list[int]:
        """Compute error magnitudes; returns a full-length error vector."""
        # Reversed syndrome polynomial (highest degree first).
        synd_poly = list(reversed(syndromes))
        err_eval = self.gf.poly_mul(synd_poly, err_loc)
        err_eval = err_eval[len(err_eval) - self.nsym:]

        coord_pos = [n - 1 - p for p in positions]
        x = [self.gf.power(2, p - self.gf.max) for p in coord_pos]

        err_vec = [0] * n
        for i, xi in enumerate(x):
            xi_inv = self.gf.inv(xi)

            # Formal derivative of the error locator, evaluated at xi_inv.
            err_loc_prime = 1
            for j, xj in enumerate(x):
                if j != i:
                    err_loc_prime = self.gf.mul(
                        err_loc_prime, 1 ^ self.gf.mul(xi_inv, xj)
                    )
            if err_loc_prime == 0:
                raise ValueError("Forney: singular error locator derivative")

            y = self.gf.mul(
                self.gf.power(xi, -self.fcr),
                self.gf.poly_eval(err_eval, xi_inv),
            )
            err_vec[positions[i]] = self.gf.div(y, err_loc_prime)
        return err_vec

    def decode(self, received: list[int]) -> list[int]:
        """
        Correct up to t symbol errors and return the k message symbols.

        Raises ValueError when the received word is not decodable; callers should
        fall back to the uncorrected prefix (this is what the reference
        Segment-WM implementation does).
        """
        if len(received) != self.n:
            raise ValueError(f"Expected {self.n} symbols, got {len(received)}")

        codeword = list(received)
        syndromes = self._syndromes(codeword)
        if max(syndromes) == 0:
            return codeword[:self.k]

        err_loc = self._berlekamp_massey(syndromes)
        num_errors = len(err_loc) - 1
        if num_errors > self.t or num_errors == 0:
            raise ValueError("Too many errors to correct")

        positions = self._chien_search(err_loc, self.n)
        if not positions:
            raise ValueError("Could not locate errors")

        err_vec = self._forney(syndromes, err_loc, positions, self.n)
        corrected = [c ^ e for c, e in zip(codeword, err_vec)]

        if max(self._syndromes(corrected)) != 0:
            raise ValueError("Decoding failed verification")
        return corrected[:self.k]

    def decode_safe(self, received: list[int]) -> tuple[list[int], bool]:
        """
        Like decode() but never raises.

        Returns (message_symbols, corrected) where corrected is False when the
        word was undecodable and the uncorrected first k symbols were returned.
        """
        try:
            return self.decode(received), True
        except (ValueError, ZeroDivisionError, IndexError):
            return list(received[:self.k]), False


class ReedSolomonCodebook:
    """
    Exhaustive minimum-distance (maximum-likelihood) decoder.

    For the payload sizes used in the Hi-DyPa tables (L = 8 bits => 256
    codewords) enumerating the whole codebook is cheap, and nearest-codeword
    decoding is strictly stronger than the bounded-distance syndrome decoder
    used by the reference Segment-WM implementation: it still corrects every
    error pattern the syndrome decoder corrects, and additionally returns the
    most likely payload instead of failing when more than t symbols are wrong.
    Giving the baseline the stronger decoder keeps the comparison conservative.
    """

    def __init__(self, rs: ReedSolomon, num_payloads: int):
        self.rs = rs
        self.num_payloads = num_payloads
        self.codewords = [
            rs.encode(payload_to_symbols(p, rs.k, rs.m)) for p in range(num_payloads)
        ]

    def decode(self, received: list[int]) -> tuple[int, int, int]:
        """
        Return (payload, symbol_distance, num_ties) for the nearest codeword.
        Ties are broken by the lowest payload value.
        """
        best_payload = 0
        best_distance = self.rs.n + 1
        ties = 0
        for payload, codeword in enumerate(self.codewords):
            distance = 0
            for a, b in zip(codeword, received):
                if a != b:
                    distance += 1
                    if distance >= best_distance:
                        break
            if distance < best_distance:
                best_distance = distance
                best_payload = payload
                ties = 1
            elif distance == best_distance:
                ties += 1
        return best_payload, best_distance, ties


def payload_to_symbols(payload: int, k: int, m: int) -> list[int]:
    """Split an integer payload into k symbols of m bits, least-significant first."""
    mask = (1 << m) - 1
    return [(payload >> (m * i)) & mask for i in range(k)]


def symbols_to_payload(symbols: list[int], m: int) -> int:
    """Inverse of payload_to_symbols."""
    payload = 0
    for i, s in enumerate(symbols):
        payload |= (int(s) & ((1 << m) - 1)) << (m * i)
    return payload
