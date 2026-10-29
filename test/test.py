# SPDX-License-Identifier: MIT
"""Drive the byte-serial interface through accumulation chains and compare the accumulator register with the
exact reference after every step (vectors from scripts/tt_vectors.py)."""
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

WE, STEP, CLEAR = 1, 1 << 6, 1 << 7


async def write(dut, addr, byte):
    dut.ui_in.value = byte
    dut.uio_in.value = WE | (addr << 1)
    await ClockCycles(dut.clk, 1)


async def read_acc(dut):
    v = 0
    for i in range(8):
        dut.uio_in.value = i << 1
        await ClockCycles(dut.clk, 1)
        v |= int(dut.uo_out.value) << (8 * i)
    return v


@cocotb.test()
async def test_accumulation_chains(dut):
    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1

    lines = Path(__file__).with_name("vectors.txt").read_text().splitlines()
    dut._log.info(lines[0])
    steps = 0
    for line in lines[1:]:
        if line == "clear":
            dut.uio_in.value = CLEAR
            await ClockCycles(dut.clk, 1)
            assert await read_acc(dut) == 0
            continue
        a, b, sa, sb, want = (int(t, 16) for t in line.split())
        for i in range(8):
            await write(dut, i, (a >> (8 * i)) & 0xFF)
            await write(dut, 8 + i, (b >> (8 * i)) & 0xFF)
        await write(dut, 16, sa)
        await write(dut, 17, sb)
        dut.uio_in.value = STEP
        await ClockCycles(dut.clk, 1)
        got = await read_acc(dut)
        assert got == want, f"step {steps}: got {got:x}, want {want:x}"
        steps += 1
    dut._log.info(f"{steps} accumulation steps match the exact reference")
