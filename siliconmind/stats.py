"""Paired bootstrap for perplexity changes measured against a reference on the same text segments."""
import numpy as np


def paired_bootstrap(res_nll, ref_nll, tok, budget, resamples=20000):
    """Perplexity change in %, its 95% interval, and the fraction of resamples within budget."""
    diff, tok = np.asarray(res_nll) - np.asarray(ref_nll), np.asarray(tok)
    idx = np.random.default_rng(0).integers(0, len(tok), (resamples, len(tok)))
    boot = 100 * np.expm1(diff[idx].sum(1) / tok[idx].sum(1))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"delta_ppl_percent": 100 * float(np.expm1(diff.sum() / tok.sum())), "ci95": [float(lo), float(hi)],
            "p_within_budget": float((boot <= budget).mean())}
