#!/usr/bin/env python3
"""TensorFlow Lite Micro tensor arena for one graph: the SRAM a Cortex-M needs
to run it with the reference kernels. Uses the recording allocator, so the
head (activations) / tail (persistent per-op buffers) split is printed too.

    ~/venvs/tflm313/bin/python tflm_arena.py ~/mcu/sudoku_mlp/inner_step_int8.tflite
"""
import sys

from tflite_micro.python.tflite_micro import runtime

path = sys.argv[1]
arena = int(sys.argv[2]) if len(sys.argv) > 2 else 1024 * 1024 * 1024
it = runtime.Interpreter.from_file(
    path, arena_size=arena, intrepreter_config=runtime.InterpreterConfig.kAllocationRecording)
print(f"[tflm] {path}", flush=True)
it.print_allocations()
