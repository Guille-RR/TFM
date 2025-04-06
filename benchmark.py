import argparse
import json
from src.benchmark.qubitBenchmark import Benchmark_3qubit
def main():
    parser = argparse.ArgumentParser(description="Benchmarking script")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON with the configuration")
    args = parser.parse_args()
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        benchmark = Benchmark_3qubit(config)
        benchmark.run_experiment()
    except Exception as e:
        print(f"Error loading config or running experiment: {e}")

if __name__ == "__main__":
    main()