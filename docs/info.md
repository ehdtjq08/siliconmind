## How it works

One step of an NVFP4 dot product: 16 pairs of E2M1 (4-bit float) values are multiplied and summed exactly, scaled
by two E4M3 block scales, and added to a floating-point accumulator:

```
acc <- round_P( acc + round_Q( sum(a_i * b_i) * sa * sb ) )
```

The accumulator width P, the pre-rounding width Q and the rounding mode were chosen by Muse Spark, an AI agent that
measured Llama-3.2-1B accuracy on a bit-exact emulation of each candidate design and synthesized each candidate with
Yosys. This tile is the design it picked: P = 9, Q = 7, round to nearest even. Its accumulator is about half the
size of a full-precision one, with no measurable change in Llama's perplexity.
See https://github.com/ehdtjq08/siliconmind.

## How to test

The unit is loaded one byte at a time:

| Pins | Meaning |
|---|---|
| `uio[0]` WE | on the rising clock edge, write `ui_in` to register ADDR |
| `uio[5:1]` ADDR | 0–7: vector a (byte i holds elements 2i and 2i+1, low nibble first), 8–15: vector b, 16: scale a, 17: scale b |
| `uio[6]` STEP | on the rising clock edge, add the loaded block to the accumulator |
| `uio[7]` CLEAR | on the rising clock edge, clear the accumulator |
| `uo_out` | byte ADDR[2:0] of the accumulator register {sign, exponent[9:0], significand[8:0]} |

Load the 18 bytes of a block, pulse STEP, repeat for the next block, then read bytes 0–2 of the accumulator.
The value is (-1)^sign × significand × 2^exponent, with the exponent in two's complement.

## External hardware

None.
