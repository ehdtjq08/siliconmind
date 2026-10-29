"""Check rtl/fp4_block_dot.v against the exact reference with random + corner-case vectors.

usage: python tests/rtl_check.py [--n 3000] [--configs 6:rne,8:rz,...]
"""
import argparse
import random
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

from siliconmind.numerics import E2M1_VALUES, ref_round_sig

EW = 10
ROOT = Path(__file__).resolve().parents[1]


def e2m1_val(code):
    return (-1 if code & 8 else 1) * Fraction(E2M1_VALUES[code & 7])


def e4m3_val(code):
    e, m = (code >> 3) & 0xF, code & 7
    return Fraction(m, 8) * Fraction(2) ** -6 if e == 0 else (1 + Fraction(m, 8)) * Fraction(2) ** (e - 7)


def encode(x: Fraction, p):
    """Fraction (already p-bit representable) -> (sign, exp, sig) with sig normalized."""
    if x == 0:
        return 0, 0, 0
    s, a = (1 if x < 0 else 0), abs(x)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    while a >= Fraction(2) ** (e + 1):
        e += 1
    while a < Fraction(2) ** e:
        e -= 1
    exp = e - p + 1
    sig = a / Fraction(2) ** exp
    assert sig.denominator == 1 and (1 << (p - 1)) <= sig.numerator < (1 << p)
    return s, exp, int(sig)


def decode(s, exp, sig):
    return (-1 if s else 1) * Fraction(sig) * Fraction(2) ** exp


def block_r(a_codes, b_codes, sa, sb):
    return sum(e2m1_val(x) * e2m1_val(y) for x, y in zip(a_codes, b_codes)) * e4m3_val(sa) * e4m3_val(sb)


def random_case(rng, p, mode):
    a = [rng.randrange(16) for _ in range(16)]
    b = [rng.randrange(16) for _ in range(16)]
    sa = rng.choice([0, rng.randrange(1, 8), rng.randrange(0x7F)])     # zero / subnormal / any
    sb = rng.randrange(0x7F)
    if rng.random() < 0.1:
        a = [rng.choice([0, 8]) for _ in range(16)]                      # P == 0
    r = block_r(a, b, sa, sb)
    kind = rng.random()
    if kind < 0.1:
        acc = Fraction(0)
    elif kind < 0.2 and r != 0:
        acc = -ref_round_sig(r, p, mode)                                 # (near) cancellation
    else:
        ref = abs(r) if r != 0 else Fraction(1)
        e_ref = ref.numerator.bit_length() - ref.denominator.bit_length()
        gap = rng.choice([rng.randint(-3, 3), rng.randint(-30, 30), rng.randint(-120, 120)])
        sig = rng.randrange(1 << (p - 1), 1 << p)
        acc = rng.choice([-1, 1]) * Fraction(sig) * Fraction(2) ** (e_ref + gap - p + 1)
    return a, b, sa, sb, acc


def run(p, mode, n, seed, q=None):
    rng = random.Random(seed)
    cases = [random_case(rng, p, mode) for _ in range(n)]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        lines = []
        for a, b, sa, sb, acc in cases:
            s, e, m = encode(acc, p)
            pack = lambda codes: sum(c << (4 * i) for i, c in enumerate(codes))
            lines.append(f"{pack(a):016x} {pack(b):016x} {sa:02x} {sb:02x} {s:x} {e & ((1 << EW) - 1):x} {m:x}")
        (td / "in.txt").write_text("\n".join(lines) + "\n")
        sim = td / "sim"
        subprocess.run(["iverilog", "-g2012", f"-Ptb.P={p}", f"-Ptb.RNE={int(mode == 'rne')}", f"-Ptb.Q={q or 19}", f"-Ptb.EW={EW}",
                        "-o", str(sim), str(ROOT / "rtl/tb_fp4_block_dot.v"), str(ROOT / "rtl/fp4_block_dot.v")],
                       check=True)
        subprocess.run(["vvp", "-n", str(sim), f"+in={td / 'in.txt'}", f"+out={td / 'out.txt'}"],
                       check=True, capture_output=True)
        outs = (td / "out.txt").read_text().split("\n")
    bad = []
    for case, line in zip(cases, outs):
        a, b, sa, sb, acc = case
        r = block_r(a, b, sa, sb)
        want = ref_round_sig(acc + (r if q is None else ref_round_sig(r, q, mode)), p, mode)
        s, e, m = (int(t, 16) for t in line.split())
        e = e - (1 << EW) if e >> (EW - 1) else e
        if decode(s, e, m) != want or (m and not m >> (p - 1)):
            bad.append((case, (s, e, m), want))
    return len(cases), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--configs", default="6:rne,8:rne,8:rz,12:rne,12:rz,16:rne,24:rne,24:rz,"
                                         "8:rne:8,8:rz:8,12:rne:10,12:rne:14,16:rz:12,24:rne:16")
    args = ap.parse_args()
    ok = True
    for i, c in enumerate(args.configs.split(",")):
        f = c.split(":")
        p, mode, q = int(f[0]), f[1], (int(f[2]) if len(f) > 2 else None)
        n, bad = run(p, mode, args.n, seed=i, q=q)
        print(f"P={p:2d} Q={q or 'exact':>5} {mode}: {n - len(bad)}/{n} match")
        for case, got, want in bad[:3]:
            print("   got", got, "want", encode(want, p) if want else 0)
        ok &= not bad
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
