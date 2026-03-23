"""CoreML DistilBERT baseline benchmark.

Downloads Apple's ANE-optimized DistilBERT, runs 1000 inferences, reports latency.
This is the number to beat with the private ANE API.

Usage: python3 benchmark_coreml.py
"""
import os
import shutil
import time

import coremltools as ct
import numpy as np
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

MODEL_ID = "apple/ane-distilbert-base-uncased-finetuned-sst-2-english"
MLPACKAGE = "DistilBERT_fp16.mlpackage"
LOCAL_PATH = "/tmp/DistilBERT_fp16.mlpackage"
SEQ_LEN = 128

# Same input as ANE private API benchmark (distilbert_bench.rs)
BENCH_TEXT = "This is a test sentence for benchmarking."

# Download if needed
if not os.path.exists(LOCAL_PATH):
    print("Downloading Apple's DistilBERT CoreML model...")
    snap = snapshot_download(MODEL_ID)
    shutil.copytree(os.path.join(snap, MLPACKAGE), LOCAL_PATH)

print("Loading CoreML model...")
model = ct.models.MLModel(LOCAL_PATH, compute_units=ct.ComputeUnit.ALL)

# Tokenize the same sentence used by the ANE bench
print(f"Tokenizing: \"{BENCH_TEXT}\"")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
enc = tokenizer(BENCH_TEXT, padding="max_length", max_length=SEQ_LEN, truncation=True, return_tensors="np")
input_data = {
    "input_ids": enc["input_ids"].astype(np.int32),
    "attention_mask": enc["attention_mask"].astype(np.int32),
}

# Warmup
print("Warming up...")
for _ in range(50):
    model.predict(input_data)

# Benchmark
print("Benchmarking (1000 iterations)...")
times = []
for _ in range(1000):
    start = time.perf_counter()
    model.predict(input_data)
    times.append((time.perf_counter() - start) * 1000)

times.sort()
median = times[len(times) // 2]
mean = sum(times) / len(times)
p5 = times[len(times) // 20]
p95 = times[len(times) * 19 // 20]

print()
print("=" * 50)
print("  CoreML DistilBERT (Apple's ANE-optimized)")
print("  Sequence length: 128, Batch size: 1")
print("=" * 50)
print(f"  Mean:   {mean:.3f} ms")
print(f"  Median: {median:.3f} ms")
print(f"  P5:     {p5:.3f} ms")
print(f"  P95:    {p95:.3f} ms")
print(f"  Min:    {min(times):.3f} ms")
print("=" * 50)
print(f"\nTo record: ./ane_kernel/target/release/ane-bench baseline {median:.3f}")
