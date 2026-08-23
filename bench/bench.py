"""mojo-bitarray benchmarks against upstream bitarray."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time

import numpy as np
from bitarray import __version__ as upstream_version
from bitarray import bitarray as upstream_bitarray
from bitarray.util import count_xor as upstream_count_xor

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

from mojo_bitarray import bitarray  # noqa: E402
from mojo_bitarray.util import count_xor  # noqa: E402


def best_time(function, repeat=5):
    best = float("inf")
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter() - start)
    return best, result


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def mojo_version():
    result = subprocess.run(
        ["mojo", "--version"], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def main():
    rng = np.random.default_rng(2026)
    size = int(os.environ.get("MOJO_BITARRAY_BENCH_MIB", "8")) * 1024 * 1024
    left_bytes = rng.integers(0, 256, size=size, dtype=np.uint8).tobytes()
    right_bytes = rng.integers(0, 256, size=size, dtype=np.uint8).tobytes()
    mojo_a, mojo_b = bitarray(left_bytes), bitarray(right_bytes)
    ref_a, ref_b = upstream_bitarray(left_bytes), upstream_bitarray(right_bytes)

    assert mojo_a.tobytes() == ref_a.tobytes()
    assert mojo_b.tobytes() == ref_b.tobytes()
    assert mojo_a.count() == ref_a.count()
    assert (mojo_a & mojo_b).tobytes() == (ref_a & ref_b).tobytes()
    assert (~mojo_a).tobytes() == (~ref_a).tobytes()
    assert (mojo_a << 13).tobytes() == (ref_a << 13).tobytes()
    assert count_xor(mojo_a, mojo_b) == upstream_count_xor(ref_a, ref_b)

    cases = [
        ("count", lambda: mojo_a.count(), lambda: ref_a.count()),
        ("bitwise AND", lambda: mojo_a & mojo_b, lambda: ref_a & ref_b),
        ("invert", lambda: ~mojo_a, lambda: ~ref_a),
        (
            "fused count_xor",
            lambda: count_xor(mojo_a, mojo_b),
            lambda: upstream_count_xor(ref_a, ref_b),
        ),
        ("left shift by 13", lambda: mojo_a << 13, lambda: ref_a << 13),
    ]

    print(f"Machine: {cpu_name()}")
    print(
        f"Linux {platform.release()}, Python {platform.python_version()}, "
        f"{mojo_version()}, bitarray {upstream_version}"
    )
    print(
        f"Best of 5 runs; two {size / 1024 / 1024:.0f} MiB inputs "
        f"({len(mojo_a):,} bits each)"
    )
    print()
    print("| operation | Mojo | upstream bitarray | upstream / Mojo |")
    print("|---|---:|---:|---:|")
    for name, mojo_fn, ref_fn in cases:
        mojo_seconds, _ = best_time(mojo_fn)
        ref_seconds, _ = best_time(ref_fn)
        speed = ref_seconds / mojo_seconds
        print(
            f"| {name} | {mojo_seconds * 1e3:.3f} ms | "
            f"{ref_seconds * 1e3:.3f} ms | {speed:.2f}x |"
        )


if __name__ == "__main__":
    main()
