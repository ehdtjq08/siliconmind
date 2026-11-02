# SiliconMind

**Muse Spark designs the accumulator of a 4-bit AI chip, and every design it proposes is checked against real silicon
cost and real LLM accuracy, bit for bit.**

**[Watch the 3-minute video](https://youtu.be/npCvxI2Hu80)** · **[Try the live demo](https://ehdtjq08.github.io/siliconmind/demo/)**:
shrink the accumulator yourself and watch Llama's answers, replay an agent run, and see the chip's layout.

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

Four independent agent runs, each allowed 10 accuracy evaluations, against non-agent searches with the same budget,
against Muse Spark answering without tools, and against the exhaustive ground truth (every one of the 162 designs
below 14,000 accumulator transistors, measured on 32 held-out segments). Budget: perplexity within +1% of exact
accumulation. Intervals are 95%, from a paired bootstrap over the segments (`scripts/confidence.py`).

| Method | Accuracy evals | Chosen design | Accumulator transistors | sky130 cell area (µm²) | Perplexity change (95% interval) |
|---|---:|---|---:|---:|---:|
| Full-size accumulator | – | P24, exact block result, RNE | 21,486 | 14,095 | +0.02% (0.00 to +0.06) |
| Muse Spark without tools, 4 runs | 1 each | P14/Q12 · P16/exact · P14/Q10 · P16/exact | 15,990–18,134 | 9,961–12,203 | all within budget |
| Random search, 3 seeds | 10 | P15/Q9 RNE · P13/Q14 RZ · P12/Q7 RZ | 16,012 · 14,456 · 12,552 | 10,411 · 9,303 · 8,065 | +0.24% · +0.13% · +0.43% (−0.22 to +1.08) |
| Bayesian optimization (TPE), 3 seeds | 10 | P9/Q8 RNE · P18/Q7 RZ · P12/Q13 RNE | 12,564 · 15,488 · 15,136 | 7,861 · 9,688 · 9,849 | +0.41% (−0.18 to +1.01) · +0.19% · +0.12% |
| Coarse grid, cheapest first | 10 | P10/Q8 RNE | 12,456 | 8,268 | +0.13% (−0.37 to +0.63) |
| **Muse Spark, run 1** | **9** | **P9/Q7 RNE** | **11,520 (−46%)** | **7,797 (−45%)** | **−0.13% (−0.56 to +0.32)** |
| Muse Spark, runs 2–4 | 10 | P8/Q8 RNE | 11,706 | 7,787 | +0.94% (+0.42 to +1.48) |
| Exhaustive: smallest measured within budget | 162 | P8/Q6 RNE | 11,152 | 7,367 | +0.82% (+0.27 to +1.39) |
| Exhaustive: smallest 95% sure within budget | 162 | P9/Q7 RNE | 11,520 | 7,797 | −0.13% (−0.56 to +0.32) |

Every agent run finds a smaller accumulator than every baseline; the best baseline (coarse grid) is 8% larger than
run 1. Without tools, Muse Spark plays safe and picks accumulators 39–57% larger. Taking each measurement at face
value, the smallest in-budget design is P8/Q6, and the agent lands 3.3–5.0% above it (the best baseline is 11.7%
above).

That face-value optimum is borderline, though: every P8 design's interval crosses the budget (P8/Q6 is within it in
73% of resamples, P8/Q8 in 59%). If you require 95% confidence (a rule we added after the benchmark ran), the optimum
is P9/Q7, exactly the design run 1 chose. P9/Q7 is the design carried to layout.

The area metric holds up in real cells: mapping all 600 designs onto sky130_fd_sc_hd standard cells
(`scripts/sky130_area.py`) ranks them almost exactly as the transistor estimate does (Spearman 0.99). In cell area,
P9/Q7 is second among the designs that are 95% sure within budget, 1.1% behind P9/Q6, which is inside synthesis
noise.

### Agent v2: told about the uncertainty

`python -m siliconmind.agent --robust` runs the same task, but confirm evaluations also return the 95% interval, and a
design only counts when the interval's upper end is within budget.

| Run | Chosen design | Accumulator transistors | Perplexity change (95% interval) |
|---|---|---:|---:|
| v2 run 1 | P12/Q8 RNE | 13,702 | −0.21% (−0.70 to +0.30) |
| v2 run 2 | P9/Q9 RNE | 12,308 | −0.30% (−0.88 to +0.29) |
| v2 run 3 | P10/exact RNE | 16,490 | −0.09% (−0.65 to +0.49) |
| v2 run 4 | **P9/Q7 RNE** | **11,520** | −0.13% (−0.56 to +0.32) |

All four v2 runs deliver a design that is 95% sure to be within budget; the original agent did so in one run of four.
Run 4 finds the smallest such design, after confirming that P8/Q7's interval crosses the budget. The price is caution:
v2 designs range from 11,520 to 16,490 transistors, and two of the four are larger than the coarse grid's pick. Four
runs each is a small sample.

### Speed and energy

The accumulator's add-and-round is the one path a pipeline cannot cut, so its delay sets the unit's peak clock.
`scripts/ppa.py` times it for every design with OpenSTA, and simulates the whole multiply-accumulate unit (block
product, accumulator and its register) at gate level on 9,216 real Llama-3.2-1B blocks from six layers
(`scripts/capture_operands.py`) to get its energy:

| Design | Accumulate loop | Unit energy per block |
|---|---:|---:|
| Full-size accumulator, P24 | 34.2 ns | 26.3 pJ |
| **P9/Q7 RNE, the chip** | **20.8 ns (−39%)** | **21.9 pJ (−17%)** |
| P8/Q8 RNE | 20.1 ns | 21.7 pJ |
| P10/Q8 RNE, coarse grid | 22.4 ns | 22.1 pJ |

The block product is the same in every design, so whole-unit energy moves less than the accumulator itself. On every
design measured, the gate-level results match the RTL bit for bit. Limits: typical corner, no wire parasitics,
area-oriented mapping, and zero-delay switching activity (no glitch power).

### Placed and routed

`scripts/pnr_runs.py` places and routes the whole multiply-accumulate unit (input registers, block product,
accumulator and its register) with LibreLane on sky130_fd_sc_hd, targeting a 40 ns clock
(`results/pnr/summary.json`):

| Design | Standard cells | Cell area (µm²) | Setup slack, typical corner |
|---|---:|---:|---:|
| Full-size accumulator, P24 | 7,431 | 61,456 | −0.67 ns (misses 25 MHz) |
| **P9/Q7 RNE, the chip** | **5,841** | **49,491 (−19%)** | **+4.56 ns** |
| P8/Q8 RNE, agent runs 2–4 | 5,906 | 49,887 | +5.12 ns |
| P9/Q6 RNE | 5,921 | 50,184 | +4.56 ns |
| P10/Q8 RNE, coarse grid | 6,048 | 51,285 | +5.58 ns |
| P12/Q7 RZ, random search | 6,044 | 51,837 | +4.97 ns |

All six are DRC, LVS and antenna clean. Narrowing the accumulator shrinks the whole unit by 19%, even though the
block product is the same in every design. After routing, P9/Q7 is the smallest of the six, so the 1.1% cell-area
edge P9/Q6 had in synthesis was noise. The slack is for the whole single-cycle path (block product, then
accumulate). At the slow corner (ss, 100 °C, 1.60 V) all six need a slower clock (worst slack −26 to −38 ns); the
Tiny Tapeout tile below runs at 10 MHz and meets every corner.

### Does it hold on a bigger model?

The designs were chosen on Llama-3.2-1B. Re-measured on Llama-3.2-3B with the same 32 held-out segments
(`scripts/model_check.py`):

| Design | On 1B | On 3B (95% interval) |
|---|---:|---:|
| Full-size accumulator, P24 | +0.02% | −0.03% (−0.46 to +0.38) |
| **P9/Q7 RNE, the chip** | **−0.13%** | **+0.19% (−0.28 to +0.65)** |
| P8/Q8 RNE, agent runs 2–4 | +0.94% | +0.80% (+0.28 to +1.31) |
| P8/Q5 RNE, borderline on 1B | +0.87% | +0.87% (+0.42 to +1.30) |
| P10/Q8 RNE, coarse grid | +0.13% | −0.03% (−0.56 to +0.50) |
| P12/Q7 RZ, random search | +0.43% | +0.89% (+0.41 to +1.34) |

The chip's design stays within budget with margin, and the P8 designs stay borderline. On 3B even the full-size
accumulator moves single segments by about ±1%: rounding activations to 4 bits turns tiny arithmetic differences into
discrete flips. The runs are deterministic (re-runs match bit for bit), so this is a property of the quantized model,
and it means 32 segments cannot resolve differences much below 0.4 points on 3B.

### Five models, six zero-shot benchmarks

The budget was set on one model and one dataset. To check the chip elsewhere, five open models ran with every
decoder matrix multiply on the emulated unit, using lm-evaluation-harness prompts and metrics on ARC-Easy,
ARC-Challenge, PIQA, WinoGrande, LAMBADA and HellaSwag (full evaluation sets, 21,848 items). Each cell is the change
in average accuracy against exact accumulation with the same NVFP4 weights and activations, in points, with a 95%
paired bootstrap interval:

| Model | Exact accumulation, avg accuracy | P24 | **P9/Q7, the chip** | P8/Q8 |
|---|---:|---:|---:|---:|
| Llama-3.2-1B | 56.65 | −0.02 | **−0.32 (−1.03 to +0.46)** | −0.04 (−0.86 to +0.75) |
| Llama-3.2-3B | 66.08 | +0.03 | **−0.11 (−0.76 to +0.56)** | −0.23 (−0.88 to +0.44) |
| Qwen2.5-1.5B | 61.37 | −0.04 | **−0.50 (−1.26 to +0.25)** | −0.37 (−1.15 to +0.36) |
| Qwen2.5-3B | 65.71 | −0.01 | **+0.39 (−0.29 to +1.09)** | +0.02 (−0.73 to +0.73) |
| Mistral-7B-v0.3 | 73.16 | −0.30 | **−0.26 (−0.73 to +0.18)** | −0.62 (−1.10 to −0.19) |

For P9/Q7, every interval contains zero. P8/Q8, the pick of agent runs 2–4, loses 0.6 points on Mistral-7B, and that
interval excludes zero. The bigger cost is NVFP4 itself: without calibration, exact accumulation is 1.0–4.4 points
below bf16 on these models. The evaluation harness for this table is not in the repository yet.

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
   `scripts/sky130_area.py` repeats this on sky130_fd_sc_hd standard cells as a cross-check of the area metric, and
   `scripts/ppa.py` simulates sky130 gate-level netlists on real Llama blocks: 0 mismatches against the RTL.

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

* Correction (Oct 30): the synthesis scripts passed P, Q and the rounding mode with Yosys `chparam`, which makes
  them unsigned. That made the exponent arithmetic unsigned, so for negative accumulator exponents the synthesized
  circuit differed from the simulated RTL. The gate-level power simulation caught it. Synthesis now passes them as
  Verilog literals through a wrapper, as the Tiny Tapeout tile always did, and every area here is re-synthesized
  (on average 1.8% different; rank correlation 0.99 with the old numbers). The agent runs and the random and grid
  baselines made their choices with the old numbers; their choices are rescored here.
* The agent and the benchmark rank designs by a Yosys transistor estimate. The sky130 cell-area cross-check agrees
  (Spearman 0.99), but both come from heuristic technology mapping, which adds a few percent of noise between
  designs. Six designs have been placed and routed; the rest are ranked by synthesis alone.
* Accuracy is perplexity of a 1B model on WikiText-2. Quick checks (4 segments) are noisy at about ±1.5%, which is why
  finalists are confirmed on 32 held-out segments. Even 32 segments leave about ±0.5 points of uncertainty, which is why
  the intervals above matter.

## Credits

Built with Llama (Llama 3.2, Llama 3.2 Community License; weights are not included in this repository) and Muse Spark
through the Meta Model API. WikiText-2 (Merity et al.) is used under CC BY-SA 3.0.
