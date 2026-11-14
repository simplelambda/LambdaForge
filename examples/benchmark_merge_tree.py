"""Disposable scaling benchmark: python examples/benchmark_merge_tree.py --sizes 1000 10000.

No Work, dataset, cluster or scientific record is created. Reports wall time and
Python-allocation peak (tracemalloc is not process RSS or all native allocations).
"""

import argparse
import time
import tracemalloc

import numpy as np

from lambdaforge.topology import SuperlevelMergeTree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 10000, 50000])
    args = parser.parse_args()
    for size in args.sizes:
        rng = np.random.default_rng(42)
        values = rng.random(size)
        edges = np.column_stack([np.arange(size - 1), np.arange(1, size)])
        tracemalloc.start()
        start = time.perf_counter()
        tree = SuperlevelMergeTree().fit(values, edges)
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(
            f"N={size} E={len(edges)} events={len(tree.events)} "
            f"seconds={elapsed:.4f} python_peak_MiB={peak / 2**20:.2f}"
        )


if __name__ == "__main__":
    main()
