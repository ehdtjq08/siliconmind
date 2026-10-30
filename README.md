# SiliconMind

**Muse Spark designs the accumulator of a 4-bit AI chip, and every design it proposes is checked against real silicon
cost and real LLM accuracy, bit for bit.**

NVFP4 dot-product units multiply 4-bit numbers cheaply, but they still add the results into a wide accumulator, and
that adder is a large share of the unit. How narrow can it get before the model running on the chip gets worse? The
answer depends on both the hardware and the model, so SiliconMind gives an agent tools for both and lets it search:

| Tool | What it does | Cost |
|---|---|---|
| `evaluate_accuracy` | Runs Llama-3.2-1B with every decoder matrix multiply computed by a bit-exact emulator of the candidate chip; returns the WikiText-2 perplexity change | expensive (10 calls allowed) |
| `synthesize_area` | Synthesizes the parameterized Verilog with Yosys and returns the transistor estimate | cheap |
| `verify_rtl` | Simulates the Verilog against an exact reference on random and corner-case vectors | cheap |

After every round the agent also sees the current area-vs-accuracy plot as an image.

## Result

Four independent agent runs, each allowed 10 accuracy evaluations, against non-agent searches with the same budget
and against the exhaustive ground truth (every one of the 169 designs below 14,000 accumulator transistors, measured on
32 held-out segments). Budget: perplexity within +1% of exact accumulation.

| Method | Chosen design | Accumulator transistors | Perplexity change |
|---|---|---:|---:|
| Full-size accumulator | P24, exact block result, RNE | 22,044 | +0.02% |
| Random search, 3 seeds | P15/Q9 RNE · P13/Q14 RZ · P12/Q7 RZ | 15,556 · 14,394 · 12,518 | +0.24% · +0.13% · +0.43% |
| Coarse grid, cheapest first | P10/Q8 RNE | 12,414 | +0.13% |
| **Muse Spark, run 1** | **P9/Q7 RNE** | **11,604 (−47%)** | **−0.13%** |
| Muse Spark, runs 2–4 | P8/Q8 RNE | 11,538 | +0.94% |
| Exhaustive ground truth | P8/Q5 RNE | 10,726 | +0.87% |

Every agent run beats every baseline, and all four land 7.6–8.2% above the true optimum (the best baseline is 15.7%
above). None of them found the optimum itself: no run tried rounding the block result below 7 bits, and the optimum
keeps P8 but rounds block results to 5. At P8, Q between 5 and 8 moves perplexity by under 0.2 points.

P9/Q7 is the design carried to layout: it is 0.6% larger than P8/Q8, within the area estimate's noise, but it has
real accuracy margin where every P8 design sits within 0.25 points of the limit. It also passes RTL verification with 0
mismatches.

## The design space

Each step the unit computes one 16-element NVFP4 block: the 16 products of E2M1 values are summed exactly (a 13-bit
integer), multiplied by the two E4M3 block scales (at most 19 significant bits), and added to the running sum:

```
acc <- round_P( acc + round_Q(R) )
```

* **P** — accumulator significand width (bits, including the hidden bit)
* **Q** — width the block result R is rounded to before the add (19 = exact)
* **mode** — round to nearest even (RNE) or truncate (RZ)

The non-obvious part, which the agent found on its own from cheap synthesis calls before spending accuracy budget:
the 19-bit block result pins the width of the alignment shifter, so narrowing P alone saves little. Q is the knob that
shrinks the adder.

## Bit-exact chain

1. `siliconmind/numerics.py` — the spec, with an exact reference in Python fractions and a fast GPU emulator
   (float64 bit-mask rounding). The emulator matches the reference with 0 mismatches on real Llama activations.
2. `rtl/fp4_block_dot.v` — one parameterized Verilog design for any P, Q and mode. `tests/rtl_check.py` simulates it with
   Icarus Verilog against the reference: 0 mismatches on 14 designs × 3,000 random and corner-case vectors, and a
   deliberately broken sticky bit is caught.
3. `scripts/synth_sweep.sh` — Yosys synthesis to CMOS gates (`abc -g cmos2`) for all 600 designs in the space.

## Running it

```bash
pip install torch transformers safetensors matplotlib
# Llama-3.2-1B (and -Instruct for the demo) from Hugging Face into models/
python -m pytest tests/                                      # numerics; RTL checks need iverilog, area needs yosys
PYTHONPATH=. python scripts/run_ppl.py --model models/Llama-3.2-1B --configs nvfp4_exact,p8_rne
```

The agent sends accuracy jobs to GPU workers over ssh (`scripts/eval_worker.py`). Point it at the worker box with
`SILICONMIND_WORKER_HOST` / `SILICONMIND_WORKER_DIR` (or `~/.config/siliconmind/worker.json` with `host` and `dir`),
put a Meta Model API key in `~/.config/meta/api_key`, then `PYTHONPATH=. python -m siliconmind.agent --budget 10`.

## Tiny Tapeout

The repository is also a Tiny Tapeout project (`info.yaml`, `src/`, `test/`, `docs/`). `src/project.v` wraps the unit
in a byte-serial interface on a 2x2 tile; on every push GitHub Actions hardens it to a sky130 layout, runs the
Tiny Tapeout precheck, and re-runs `test/test.py` on the gate-level netlist. The test steps the accumulator through
chains of random blocks and compares it with the exact reference after every step (`scripts/tt_vectors.py`).

For P9/Q7 the flow passes end to end: 5,744 standard cells placed and routed at 69% utilization, all precheck checks
clean, timing met at 10 MHz in every corner (worst setup slack 6.2 ns at ss, 100 °C, 1.60 V), and the gate-level
test matches the reference.

## Caveats

* Area comparisons use a Yosys transistor estimate (only the chosen design has been placed and routed); the heuristic
  mapper adds about ±5% noise between designs.
* Accuracy is perplexity of a 1B model on WikiText-2. Quick checks (4 segments) are noisy at about ±1.5%, which is why
  finalists are confirmed on 32 held-out segments. The +1% budget is tight for the P8 designs: P8/Q8 measured +1.08%
  on the quick-check segments and +0.94% on the held-out ones.

## Credits

Built with Llama (Llama 3.2, Llama 3.2 Community License; weights are not included in this repository) and Muse Spark
through the Meta Model API. WikiText-2 (Merity et al.) is used under CC BY-SA 3.0.
