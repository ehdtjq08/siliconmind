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

| Method | Chosen design | Accumulator transistors | sky130 cell area (µm²) | Perplexity change (95% interval) |
|---|---|---:|---:|---:|
| Full-size accumulator | P24, exact block result, RNE | 22,044 | 14,294 | +0.02% (0.00 to +0.06) |
| Random search, seed 1 | P15/Q9 RNE | 15,556 | 10,244 | +0.24% (−0.41 to +0.93) |
| Random search, seed 2 | P13/Q14 RZ | 14,394 | 9,210 | +0.13% (−0.43 to +0.71) |
| Random search, seed 3 | P12/Q7 RZ | 12,518 | 7,936 | +0.43% (−0.22 to +1.08) |
| Coarse grid, cheapest first | P10/Q8 RNE | 12,414 | 8,339 | +0.13% (−0.37 to +0.63) |
| **Muse Spark, run 1** | **P9/Q7 RNE** | **11,604 (−47%)** | **7,839 (−45%)** | **−0.13% (−0.56 to +0.32)** |
| Muse Spark, runs 2–4 | P8/Q8 RNE | 11,538 | 7,406 | +0.94% (+0.42 to +1.48) |
| Exhaustive: smallest measured within budget | P8/Q5 RNE | 10,726 | 7,282 | +0.87% (+0.41 to +1.36) |
| Exhaustive: smallest 95% sure within budget | P9/Q7 RNE | 11,604 | 7,839 | −0.13% (−0.56 to +0.32) |

Every agent run beats every baseline. Taking each measurement at face value, the optimum is P8/Q5 and the agent lands
7.6–8.2% above it (the best baseline is 15.7% above). None of the runs tried rounding the block result below 7 bits,
which is where that optimum sits.

That optimum is borderline, though. The intervals come from a paired bootstrap over the 32 held-out segments
(`scripts/confidence.py`), and every P8 design's interval crosses the budget: P8/Q5 is within it in 70% of
resamples, P8/Q8 in 59%. If you require 95% confidence (a rule we added after the benchmark ran), the optimum is
P9/Q7, exactly the design run 1 chose. P9/Q7 is the design carried to layout, and it passes RTL verification with 0
mismatches.

The area metric holds up in real cells. Mapping all 600 designs onto sky130_fd_sc_hd standard cells
(`scripts/sky130_area.py`) ranks them almost exactly as the transistor estimate does (Spearman 0.99). In cell area,
P9/Q7 ranks second among the designs that are 95% sure within budget, 1.5% behind P9/Q6, which is inside synthesis
noise.

### Does it hold on a bigger model?

The designs were chosen on Llama-3.2-1B. Re-measured on Llama-3.2-3B with the same 32 held-out segments
(`scripts/model_check.py`):

| Design | On 1B | On 3B (95% interval) |
|---|---:|---:|
| Full-size accumulator, P24 | +0.02% | −0.03% (−0.46 to +0.38) |
| **P9/Q7 RNE, the chip** | **−0.13%** | **+0.19% (−0.28 to +0.65)** |
| P8/Q8 RNE, agent runs 2–4 | +0.94% | +0.80% (+0.28 to +1.31) |
| P8/Q5 RNE, face-value optimum on 1B | +0.87% | +0.87% (+0.42 to +1.30) |
| P10/Q8 RNE, coarse grid | +0.13% | −0.03% (−0.56 to +0.50) |
| P12/Q7 RZ, random search | +0.43% | +0.89% (+0.41 to +1.34) |

The chip's design stays within budget with margin, and the P8 designs stay borderline. On 3B even the full-size
accumulator moves single segments by about ±1%: rounding activations to 4 bits turns tiny arithmetic differences into
discrete flips. The runs are deterministic (re-runs match bit for bit), so this is a property of the quantized model,
and it means 32 segments cannot resolve differences much below 0.4 points on 3B.

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
   `scripts/sky130_area.py` repeats this on sky130_fd_sc_hd standard cells as a cross-check of the area metric.

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

* The agent and the benchmark rank designs by a Yosys transistor estimate. The sky130 cell-area cross-check agrees
  (Spearman 0.99), but both come from heuristic technology mapping, which adds a few percent of noise between
  designs. Only the chosen design has been placed and routed.
* Accuracy is perplexity of a 1B model on WikiText-2. Quick checks (4 segments) are noisy at about ±1.5%, which is why
  finalists are confirmed on 32 held-out segments. Even 32 segments leave about ±0.5 points of uncertainty, which is why
  the intervals above matter.

## Credits

Built with Llama (Llama 3.2, Llama 3.2 Community License; weights are not included in this repository) and Muse Spark
through the Meta Model API. WikiText-2 (Merity et al.) is used under CC BY-SA 3.0.
