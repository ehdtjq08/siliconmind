"""Run a HF causal LM with every decoder Linear routed through the emulated NVFP4 dot-product unit."""
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from .numerics import block_dot_accumulate, quantize_nvfp4

LINEAR_RE = re.compile(r"model\.layers\.(\d+)\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$")


class UnitConfig:
    """Shared, mutable hardware config. `default` applies to every emulated layer unless
    `per_layer` maps a module name (or a regex) to an override.

    A config is a dict: {"kind": "bf16" | "nvfp4", "p": int | None, "q": int | None,
                         "mode": "rne" | "rz", "promote": int | None}
    """

    def __init__(self, default):
        self.default = default
        self.per_layer = {}

    def for_layer(self, name):
        for pat, cfg in self.per_layer.items():
            if re.search(pat, name):
                return cfg
        return self.default


class EmuLinear(nn.Module):
    def __init__(self, lin: nn.Linear, name: str, unit: UnitConfig, block_chunk: int = 32):
        super().__init__()
        self.name, self.unit, self.block_chunk = name, unit, block_chunk
        self.weight = lin.weight
        self.bias = lin.bias
        qw, sw, gw = quantize_nvfp4(lin.weight.data)
        self.register_buffer("qw", qw.to(torch.bfloat16), persistent=False)   # E2M1 values are exact in bf16
        self.register_buffer("sw", sw, persistent=False)
        self.register_buffer("gw", gw, persistent=False)

    def forward(self, x):
        cfg = self.unit.for_layer(self.name)
        if cfg["kind"] == "bf16":
            return F.linear(x, self.weight, self.bias)
        shp = x.shape
        qx, sx, gx = quantize_nvfp4(x.reshape(-1, shp[-1]))
        acc = block_dot_accumulate(qx, sx, self.qw, self.sw, p=cfg.get("p"), mode=cfg.get("mode", "rne"),
                                   promote=cfg.get("promote"), q=cfg.get("q"), block_chunk=self.block_chunk)
        y = (acc * (gx * self.gw).double()).to(x.dtype).reshape(*shp[:-1], -1)
        return y if self.bias is None else y + self.bias


def patch_model(model, unit: UnitConfig, block_chunk: int = 32):
    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and LINEAR_RE.match(n)]
    for n in names:
        parent_name, child = n.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, child, EmuLinear(getattr(parent, child), n, unit, block_chunk))
    return names


@torch.no_grad()
def segment_nll(model, ids: torch.Tensor):
    """Sum of next-token NLL and token count for one segment (1, L)."""
    logits = model(ids).logits.float()
    nll = F.cross_entropy(logits[0, :-1], ids[0, 1:], reduction="sum")
    return nll.item(), ids.shape[1] - 1


def ppl(total_nll, total_tokens):
    return math.exp(total_nll / total_tokens)
