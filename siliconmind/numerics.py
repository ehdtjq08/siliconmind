"""Bit-exact numerics of an NVFP4 block dot-product unit.

Spec shared by the RTL, the exact reference and the fast LLM emulator:

  operands  per 16-element block: a_i, b_i in E2M1, block scales sa, sb in E4M3 (OCP E4M3FN)
  P = sum_i a_i * b_i        exact; multiples of 1/4, |P| <= 16*36 -> 13-bit signed in units of 1/4
  R = P * sa * sb            exact (<= 20 significant bits)
  acc <- round_p(acc + round_q(R))
                             p = significand bits incl. hidden bit, q = pre-rounding of R
                             (None = exact), mode "rne" or "rz" for both, unbounded exponent
  y = acc * gx * gw          per-tensor FP32 global scales, applied once at the end

`promote=N` adds DeepSeek-style promotion: the narrow accumulator is flushed into an
exact (float64) accumulator every N blocks.
"""
import math
from fractions import Fraction

import torch

BLOCK = 16
E2M1_MAX = 6.0
E4M3_MAX = 448.0
E4M3_MIN_NORMAL = 2.0 ** -6
E4M3_SUBNORMAL_STEP = 2.0 ** -9


# ---------------------------------------------------------------- element formats

def round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round to the E2M1 grid {0, .5, 1, 1.5, 2, 3, 4, 6}, ties to even, saturating."""
    a = x.abs().clamp(max=E2M1_MAX)
    # step is 0.5 below 2, 1 in [2, 4), 2 in [4, 6]; torch.round is half-to-even,
    # which coincides with ties-to-even on the E2M1 mantissa bit for every step
    step = torch.where(a < 2, 0.5, torch.where(a < 4, 1.0, 2.0)).to(a.dtype)
    return torch.sign(x) * torch.round(a / step) * step


def round_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round non-negative values to E4M3FN (4 significant bits, subnormals, max 448), RNE."""
    a = x.clamp(min=0, max=E4M3_MAX)
    m, e = torch.frexp(a)                                   # a = m * 2**e, m in [0.5, 1)
    normal = torch.ldexp(torch.round(m * 16) / 16, e)
    sub = torch.round(a / E4M3_SUBNORMAL_STEP) * E4M3_SUBNORMAL_STEP
    return torch.where(a < E4M3_MIN_NORMAL, sub, normal).clamp(max=E4M3_MAX)


def round_sig(x: torch.Tensor, p: int, mode: str = "rne") -> torch.Tensor:
    """Round float64 x to p significant bits (unbounded exponent) by masking the low 53-p bits.

    For RNE, adding (half - 1 + lsb) before masking rounds ties to even; a carry out of the
    significand bumps the exponent, which is the correct result. Truncation is round toward zero.
    """
    d = 53 - p
    if d <= 0:
        return x
    bits = x.view(torch.int64)
    if mode == "rne":
        bits = bits + (((bits >> d) & 1) + ((1 << (d - 1)) - 1))
    return (bits & ~((1 << d) - 1)).view(torch.float64)


def _accumulate_chunk(acc, R, p: int, mode: str):
    for j in range(R.shape[0]):
        acc = round_sig(acc + R[j], p, mode)
    return acc


_accumulate_chunk_compiled = torch.compile(_accumulate_chunk, dynamic=False)


# ---------------------------------------------------------------- NVFP4 quantization

def quantize_nvfp4(x: torch.Tensor):
    """Two-level NVFP4 quantization along the last dim (block 16).

    Returns (q, s, g): E2M1 values (..., K), E4M3 block scales (..., K/16), FP32 global scale,
    with x ~= q * s * g.
    """
    x = x.float()
    *lead, k = x.shape
    xb = x.reshape(*lead, k // BLOCK, BLOCK)
    amax = x.abs().amax()
    g = (amax / (E2M1_MAX * E4M3_MAX)) if amax > 0 else torch.ones((), device=x.device)
    s = round_e4m3(xb.abs().amax(-1) / (E2M1_MAX * g))
    denom = (s * g).unsqueeze(-1)
    q = torch.where(denom > 0, round_e2m1(xb / torch.where(denom > 0, denom, 1)), 0)
    return q.reshape(*lead, k), s, g


# ---------------------------------------------------------------- fast emulator

def block_dot_accumulate(qx, sx, qw, sw, p=None, mode="rne", promote=None, q=None, block_chunk=64,
                         compiled=True):
    """sum_b round_p(acc + round_q(P_b * sx_b * sw_b)) for every (row of x, row of w).

    qx (T, K), sx (T, K/16), qw (O, K), sw (O, K/16). p=None / q=None mean exact.
    Returns float64 (T, O) without the global scales.
    """
    t, k = qx.shape
    o = qw.shape[0]
    nb = k // BLOCK
    xb = qx.reshape(t, nb, BLOCK).permute(1, 0, 2)          # (nb, T, 16)
    wb = qw.reshape(o, nb, BLOCK).permute(1, 2, 0)          # (nb, 16, O)
    acc = torch.zeros(t, o, dtype=torch.float64, device=qx.device)
    wide = torch.zeros_like(acc) if promote else None
    for c0 in range(0, nb, block_chunk):
        c1 = min(nb, c0 + block_chunk)
        P = torch.bmm(xb[c0:c1], wb[c0:c1])                 # exact in fp32: multiples of 1/4, <= 576
        S = sx[:, c0:c1].T.unsqueeze(2) * sw[:, c0:c1].T.unsqueeze(1)   # exact: 4b x 4b significands
        R = (P * S).double()                                # exact: <= 20 significant bits
        if q is not None:
            R = round_sig(R, q, mode)
        if p is not None and not promote and compiled and acc.is_cuda:
            acc = _accumulate_chunk_compiled(acc, R, p, mode)
            continue
        for j in range(c1 - c0):
            acc = acc + R[j]
            if p is not None:
                acc = round_sig(acc, p, mode)
            if promote and (c0 + j + 1) % promote == 0:
                wide += acc
                acc = torch.zeros_like(acc)
    return acc + wide if promote else acc


# ---------------------------------------------------------------- exact reference (spec)

def _frac(v) -> Fraction:
    return Fraction(float(v))


def ref_round_sig(x: Fraction, p: int, mode: str) -> Fraction:
    if x == 0:
        return Fraction(0)
    sign = -1 if x < 0 else 1
    a = abs(x)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    while a >= Fraction(2) ** e:
        e += 1
    while a < Fraction(2) ** (e - 1):
        e -= 1
    scaled = a / Fraction(2) ** (e - p)                     # in [2**(p-1), 2**p)
    n, rem = divmod(scaled.numerator, scaled.denominator)
    if mode == "rne":
        frac2 = 2 * rem
        if frac2 > scaled.denominator or (frac2 == scaled.denominator and n % 2 == 1):
            n += 1
    return sign * Fraction(n) * Fraction(2) ** (e - p)


def ref_block_step(acc: Fraction, a, b, sa, sb, p, mode, q=None) -> Fraction:
    """One step of the unit on python scalars: acc <- round_p(acc + round_q(sum(a*b) * sa * sb))."""
    P = sum(_frac(ai) * _frac(bi) for ai, bi in zip(a, b))
    R = P * _frac(sa) * _frac(sb)
    if q is not None:
        R = ref_round_sig(R, q, mode)
    s = acc + R
    return s if p is None else ref_round_sig(s, p, mode)


# ---------------------------------------------------------------- codes (for RTL vectors)

E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def e2m1_code(v: float) -> int:
    return (8 if v < 0 else 0) | E2M1_VALUES.index(abs(v))


def e4m3_code(v: float) -> int:
    """Positive E4M3FN value -> 8-bit code."""
    if v == 0:
        return 0
    if v < E4M3_MIN_NORMAL:
        return int(round(v / E4M3_SUBNORMAL_STEP))
    mant, exp = math.frexp(v)                               # v = mant * 2**exp, mant in [0.5, 1)
    return ((exp - 1 + 7) << 3) | int(round((mant * 2 - 1) * 8))
