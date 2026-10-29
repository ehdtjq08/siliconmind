import random
from fractions import Fraction

import torch

from siliconmind.numerics import (
    E2M1_VALUES, block_dot_accumulate, e2m1_code, e4m3_code, quantize_nvfp4,
    ref_block_step, round_e2m1, round_e4m3,
)


def all_e4m3_values():
    vals = [m * 2.0 ** -9 for m in range(8)]                          # subnormals
    vals += [(1 + m / 8) * 2.0 ** (e - 7) for e in range(1, 16) for m in range(8) if not (e == 15 and m == 7)]
    return vals


def test_e2m1_grid_and_ties():
    grid = torch.tensor(E2M1_VALUES)
    assert torch.equal(round_e2m1(grid), grid)
    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0])
    assert round_e2m1(ties).tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0]


def test_e4m3_roundtrip_and_codes():
    vals = all_e4m3_values()
    t = torch.tensor(vals, dtype=torch.float32)
    assert torch.equal(round_e4m3(t), t)
    codes = [e4m3_code(v) for v in vals]
    assert len(set(codes)) == len(codes) and max(codes) == 0x7E
    assert [e2m1_code(v) for v in E2M1_VALUES] == list(range(8))


def test_quantize_reconstructs_on_grid():
    x = torch.randn(4, 64)
    q, s, g = quantize_nvfp4(x)
    assert set(q.abs().unique().tolist()) <= set(E2M1_VALUES)
    assert torch.equal(round_e4m3(s), s)
    err = (q.reshape(4, 4, 16) * s.unsqueeze(-1) * g).reshape(4, 64) - x
    assert err.abs().max() < x.abs().max() / 2


def test_fast_emulator_matches_exact_reference():
    rng = random.Random(0)
    scales = [v for v in all_e4m3_values() if v > 0]
    t, o, nb = 3, 4, 40
    qx = torch.tensor([[rng.choice(E2M1_VALUES) * rng.choice([-1, 1]) for _ in range(nb * 16)] for _ in range(t)])
    qw = torch.tensor([[rng.choice(E2M1_VALUES) * rng.choice([-1, 1]) for _ in range(nb * 16)] for _ in range(o)])
    sx = torch.tensor([[rng.choice(scales) for _ in range(nb)] for _ in range(t)])
    sw = torch.tensor([[rng.choice(scales) for _ in range(nb)] for _ in range(o)])
    for p, mode in [(None, "rne"), (24, "rne"), (12, "rne"), (8, "rz"), (6, "rne"), (5, "rz")]:
        fast = block_dot_accumulate(qx, sx, qw, sw, p=p, mode=mode)
        for i in range(t):
            for j in range(o):
                acc = Fraction(0)
                for b in range(nb):
                    sl = slice(b * 16, b * 16 + 16)
                    acc = ref_block_step(acc, qx[i, sl].tolist(), qw[j, sl].tolist(),
                                         sx[i, b].item(), sw[j, b].item(), p, mode)
                assert Fraction(fast[i, j].item()) == acc, (p, mode, i, j)
