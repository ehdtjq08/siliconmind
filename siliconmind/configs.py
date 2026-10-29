"""Named accumulator configs shared by the local runner and the Modal app."""

CONFIGS = {
    "bf16": {"kind": "bf16"},
    "nvfp4_exact": {"kind": "nvfp4", "p": None},
    "p24_rne": {"kind": "nvfp4", "p": 24, "mode": "rne"},
    "p16_rne": {"kind": "nvfp4", "p": 16, "mode": "rne"},
    "p12_rne": {"kind": "nvfp4", "p": 12, "mode": "rne"},
    "p10_rne": {"kind": "nvfp4", "p": 10, "mode": "rne"},
    "p8_rne": {"kind": "nvfp4", "p": 8, "mode": "rne"},
    "p7_rne": {"kind": "nvfp4", "p": 7, "mode": "rne"},
    "p6_rne": {"kind": "nvfp4", "p": 6, "mode": "rne"},
    "p24_rz": {"kind": "nvfp4", "p": 24, "mode": "rz"},
    "p20_rz": {"kind": "nvfp4", "p": 20, "mode": "rz"},
    "p16_rz": {"kind": "nvfp4", "p": 16, "mode": "rz"},
    "p14_rz": {"kind": "nvfp4", "p": 14, "mode": "rz"},
    "p12_rz": {"kind": "nvfp4", "p": 12, "mode": "rz"},
    "p10_rz": {"kind": "nvfp4", "p": 10, "mode": "rz"},
    "p8_rz": {"kind": "nvfp4", "p": 8, "mode": "rz"},
    # R pre-rounded to q bits before the add (narrow alignment datapath)
    "p12_q12_rne": {"kind": "nvfp4", "p": 12, "q": 12, "mode": "rne"},
    "p10_q10_rne": {"kind": "nvfp4", "p": 10, "q": 10, "mode": "rne"},
    "p8_q8_rne": {"kind": "nvfp4", "p": 8, "q": 8, "mode": "rne"},
    "p12_q8_rne": {"kind": "nvfp4", "p": 12, "q": 8, "mode": "rne"},
    "p16_q8_rne": {"kind": "nvfp4", "p": 16, "q": 8, "mode": "rne"},
    # DeepSeek-style promotion: narrow accumulator flushed to exact every N blocks
    "p8_rz_promote4": {"kind": "nvfp4", "p": 8, "mode": "rz", "promote": 4},
    "p8_rne_promote4": {"kind": "nvfp4", "p": 8, "mode": "rne", "promote": 4},
}

# per-layer splits: narrow config on a subset of layers, exact elsewhere
SPLITS = {
    "p6_rne_down_only": ({"kind": "nvfp4", "p": None}, {r"down_proj$": {"kind": "nvfp4", "p": 6, "mode": "rne"}}),
    "p6_rne_except_down": ({"kind": "nvfp4", "p": 6, "mode": "rne"}, {r"down_proj$": {"kind": "nvfp4", "p": None}}),
    "p12_rz_down_only": ({"kind": "nvfp4", "p": None}, {r"down_proj$": {"kind": "nvfp4", "p": 12, "mode": "rz"}}),
    "p12_rz_except_down": ({"kind": "nvfp4", "p": 12, "mode": "rz"}, {r"down_proj$": {"kind": "nvfp4", "p": None}}),
}
