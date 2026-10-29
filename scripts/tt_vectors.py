"""Accumulation chains for the Tiny Tapeout test (test/vectors.txt), computed with the exact reference.

Each chain clears the accumulator and steps it through random NVFP4 blocks; after every step the expected
accumulator register {sign, exponent, significand} is recorded.

    PYTHONPATH=. python scripts/tt_vectors.py --p 9 --q 7 --mode rne
"""
import argparse
import random
from fractions import Fraction
from pathlib import Path

from siliconmind.numerics import ref_round_sig
from tests.rtl_check import EW, block_r, encode

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, required=True)
    ap.add_argument("--q", type=int, default=None)
    ap.add_argument("--mode", choices=["rne", "rz"], required=True)
    ap.add_argument("--chains", type=int, default=6)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    lines = [f"# P={a.p} Q={a.q or 19} RNE={int(a.mode == 'rne')}  (a b sa sb -> expected accumulator, hex)"]
    for c in range(a.chains):
        lines.append("clear")
        acc = Fraction(0)
        for _ in range(a.steps):
            x = [rng.randrange(16) for _ in range(16)]
            y = [rng.randrange(16) for _ in range(16)]
            if c % 3 == 2 and rng.random() < 0.5:                     # opposite-sign blocks: cancellation
                y = [v ^ 8 for v in x]
            sa, sb = rng.randrange(0x18, 0x60), rng.randrange(0x18, 0x60)   # scales 2^-4 .. 2^4
            r = block_r(x, y, sa, sb)
            acc = ref_round_sig(acc + (r if a.q is None else ref_round_sig(r, a.q, a.mode)), a.p, a.mode)
            s, e, m = encode(acc, a.p)
            bits = (s << (EW + a.p)) | ((e & ((1 << EW) - 1)) << a.p) | m
            pack = lambda codes: sum(v << (4 * i) for i, v in enumerate(codes))
            lines.append(f"{pack(x):016x} {pack(y):016x} {sa:02x} {sb:02x} {bits:x}")
    out = ROOT / "test/vectors.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}: {a.chains} chains x {a.steps} steps")


if __name__ == "__main__":
    main()
