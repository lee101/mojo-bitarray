# mojo-bitarray

`mojo-bitarray` is a standalone Mojo port of the compute-heavy core of the
Python [`bitarray`](https://pypi.org/project/bitarray/) package. It stores bits
densely, eight per byte, and calls compiled Mojo for whole-vector bitwise
operations, inversion, population counts, shifts, scans, reversal, and fused
set-count operations.

The Python API follows upstream names and signatures for the covered subset.
Existing code normally only needs to change its import:

```python
from mojo_bitarray import bitarray
```

## Coverage

Covered:

- `bitarray` construction from lengths, `01` strings, iterables, and bytes,
  with both `big` and `little` bit endianness
- packed byte and file conversion: `tobytes`, `frombytes`, `pack`, `unpack`,
  `tofile`, and `fromfile`
- mutable sequence operations, indexing, slicing, concatenation, repetition,
  comparison, sorting, and reversal
- `&`, `|`, `^`, `~`, left/right shifts, and their in-place forms
- `count`, `all`, `any`, `find`, `index`, `search`, `setall`, `fill`,
  `invert`, and `bytereverse`
- prefix-code `encode` and `decode`
- `bitarray.util`-style `count_and`, `count_or`, `count_xor`, `any_and`,
  `subset`, `parity`, `zeros`, and `ones`

Not covered:

- `frozenbitarray` and `decodetree`
- zero-copy imported buffers, writable buffer exports, and memory-mapped
  storage
- the remainder of the large `bitarray.util` module, including integer/base
  conversion, canonical Huffman helpers, compression codecs, and
  serialization helpers
- ABI compatibility with upstream's C extension

The repository tests behavioral and packed-byte parity directly against
upstream `bitarray` 3.8.0.

## Install

The repository pins the tested Mojo nightly and installs all Python
dependencies through Pixi:

```bash
pixi install
pixi run build
```

The build creates `dist/libmojo-bitarray.so`. Tests and benchmarks always run
inside the same environment:

```bash
pixi run test
pixi run bench
```

## Usage

```python
from mojo_bitarray import bitarray
from mojo_bitarray.util import count_xor

a = bitarray("10110010")
b = bitarray("11100010")

changed = a ^ b
assert changed.to01() == "01010000"
assert count_xor(a, b) == 2
assert (a << 2).to01() == "11001000"
assert a.tobytes() == b"\xb2"
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-136-generic, Python 3.13.14, Mojo
1.1.0.dev2026081105 (e929a99e), and upstream bitarray 3.8.0. Each row is the best of
five runs on two 8 MiB packed vectors (67,108,864 bits). A value above `1.00x`
in the last column means Mojo was faster.

| operation | Mojo | upstream bitarray | upstream / Mojo |
|---|---:|---:|---:|
| count | 1.521 ms | 3.880 ms | 2.55x |
| bitwise AND | 1.909 ms | 2.211 ms | 1.16x |
| invert | 1.217 ms | 1.938 ms | 1.59x |
| fused count_xor | 1.140 ms | 3.881 ms | 3.41x |
| left shift by 13 | 3.847 ms | 4.201 ms | 1.09x |

In this run Mojo was faster for every measured operation. The fused count
avoids allocating a result vector. These are measured results, not projected
speedups. A separate locked 128 MiB crossover run measured allocating AND at
128.382 ms with four workers versus 146.613 ms serial, and inversion at
121.306 ms versus 134.966 ms serial. Smaller inputs remain serial.

There is intentionally no GPU path. These kernels perform at most one simple
bitwise operation per byte while streaming two or three bytes, far below the
roughly two-flops-per-byte arithmetic intensity needed to justify device
transfer and launch overhead.

## How it works

The Python object owns a `bytearray` and an exact logical bit length. Logical
bit zero is the most-significant bit of the first byte for `endian="big"` and
the least-significant bit for `endian="little"`. Unused tail bits are kept at
zero, so population counts and fused operations can process every stored byte
without a separate tail branch.

Python owns allocation, resizing, validation, and sequence semantics. It
passes contiguous byte-buffer views and scalar sizes through `ctypes`; the C ABI exports
accept addresses as Mojo `Int` values and reconstruct
`UnsafePointer[UInt8, AnyOrigin[mut=True]]` inside the kernel. The shared
library never retains a pointer and never allocates Python-visible memory.
Calls retain the GIL and a live Python buffer export, so the owning bytearray
cannot be resized while Mojo is using its address.
Bitwise operations, inversion, and shifts process full SIMD vectors followed
by a scalar tail. Allocating bitwise and inversion operations write directly
from the source into the new result buffer instead of copying the source first.
Shifts combine adjacent packed-byte vectors directly and clear only the bytes
vacated by the shift. Bitwise and inversion calls at 128 MiB and above use four
CPU workers; smaller buffers stay serial. All exports live in one Mojo
compilation unit.
