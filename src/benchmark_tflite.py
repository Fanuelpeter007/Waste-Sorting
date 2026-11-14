"""Measure TensorFlow Lite inference time and frame rate on the current device.

Runs the model on random input of the right shape, skips warm-up runs, and
reports the mean, median and 95th percentile latency in milliseconds, plus the
frame rate that the mean latency allows.

Usage:
    python src/benchmark_tflite.py --model models/efficientnet_b3.tflite --runs 200
"""

import argparse
import statistics
import time

import numpy as np
import tensorflow as tf


def main():
    p = argparse.ArgumentParser(description="Benchmark a TFLite model.")
    p.add_argument("--model", required=True, help="Path to the .tflite file")
    p.add_argument("--runs", type=int, default=200, help="Timed runs (default: 200)")
    p.add_argument("--warmup", type=int, default=10, help="Untimed warm-up runs (default: 10)")
    p.add_argument("--threads", type=int, default=4, help="Interpreter threads (default: 4)")
    args = p.parse_args()

    interpreter = tf.lite.Interpreter(model_path=args.model, num_threads=args.threads)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    data = np.random.random_sample(inp["shape"]).astype(inp["dtype"])

    times_ms = []
    for i in range(args.warmup + args.runs):
        start = time.perf_counter()
        interpreter.set_tensor(inp["index"], data)
        interpreter.invoke()
        interpreter.get_tensor(out["index"])
        if i >= args.warmup:
            times_ms.append((time.perf_counter() - start) * 1000.0)

    mean = statistics.mean(times_ms)
    p95 = sorted(times_ms)[int(0.95 * (len(times_ms) - 1))]
    print(f"Model:   {args.model}")
    print(f"Input:   {list(inp['shape'])} {np.dtype(inp['dtype']).name}")
    print(f"Runs:    {args.runs} (threads={args.threads})")
    print(f"Latency: mean {mean:.1f} ms, median {statistics.median(times_ms):.1f} ms, p95 {p95:.1f} ms")
    print(f"FPS:     {1000.0 / mean:.1f} (inference only)")


if __name__ == "__main__":
    main()
