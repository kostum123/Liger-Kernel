import torch
import time
import argparse
from transformers import get_activation
from liger_kernel.transformers.functional import liger_relu_squared
from benchmark.scripts.utils import _test_memory

def benchmark_transformers_relu_squared(input_data):
    activation = get_activation("relu2")
    start_time = time.time()
    output = activation(input_data)
    end_time = time.time()
    execution_time = end_time - start_time
    memory_usage = _test_memory(lambda: activation(input_data))
    return execution_time, memory_usage, output

def benchmark_liger_relu_squared(input_data):
    start_time = time.time()
    output = liger_relu_squared(input_data)
    end_time = time.time()
    execution_time = end_time - start_time
    memory_usage = _test_memory(lambda: liger_relu_squared(input_data))
    return execution_time, memory_usage, output

def main():
    parser = argparse.ArgumentParser(description="Benchmark ReLU Squared Activation Functions")
    parser.add_argument("--input_size", type=int, default=1024, help="Size of the input tensor")
    parser.add_argument("--num_runs", type=int, default=100, help="Number of runs for benchmarking")
    args = parser.parse_args()

    input_data = torch.randn(args.input_size, device="cuda")

    transformers_times = []
    transformers_memory = []
    liger_times = []
    liger_memory = []

    for _ in range(args.num_runs):
        transformers_time, transformers_mem, _ = benchmark_transformers_relu_squared(input_data)
        liger_time, liger_mem, _ = benchmark_liger_relu_squared(input_data)
        transformers_times.append(transformers_time)
        transformers_memory.append(transformers_mem)
        liger_times.append(liger_time)
        liger_memory.append(liger_mem)

    avg_transformers_time = sum(transformers_times) / len(transformers_times)
    avg_transformers_memory = sum(transformers_memory) / len(transformers_memory)
    avg_liger_time = sum(liger_times) / len(liger_times)
    avg_liger_memory = sum(liger_memory) / len(liger_memory)

    print(f"Average execution time for transformers ReLU Squared: {avg_transformers_time:.6f} seconds")
    print(f"Average memory usage for transformers ReLU Squared: {avg_transformers_memory:.2f} MB")
    print(f"Average execution time for Liger ReLU Squared: {avg_liger_time:.6f} seconds")
    print(f"Average memory usage for Liger ReLU Squared: {avg_liger_memory:.2f} MB")

if __name__ == "__main__":
    main()
