"""ctypes loader for the Mojo packed-bit kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "bitarray.mojo")
LIB = os.environ.get("MOJO_BITARRAY_LIB") or os.path.join(
    ROOT, "dist", "libmojo-bitarray.so"
)

I = ctypes.c_int64
P = ctypes.c_void_p

_SIGNATURES = {
    "mba_binary": ([P, P, P, I, I], None),
    "mba_invert": ([P, P, I, I, I], None),
    "mba_count": ([P, I, I], I),
    "mba_count_binary": ([P, P, I, I], I),
    "mba_subset": ([P, P, I], I),
    "mba_find": ([P, I, I, I, I, I], I),
    "mba_shift": ([P, P, I, I, I, I], None),
    "mba_reverse": ([P, P, I, I], None),
    "mba_setall": ([P, I, I, I, I], None),
    "mba_bytereverse": ([P, I, I], None),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJO_BITARRAY_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    manifest = os.path.join(ROOT, "pixi.toml")
    if os.path.exists(pixi) and os.path.exists(manifest):
        return [pixi, "run", "--manifest-path", manifest, "mojo"]
    raise BuildError("mojo not found; set MOJO_BITARRAY_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_BITARRAY_LIB") and os.path.exists(LIB) and not force:
        return LIB
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(SRC):
        return LIB
    if not os.path.exists(SRC):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(f"no Mojo source at {SRC} and no shared library at {LIB}")
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + ["build", "--emit", "shared-lib", SRC, "-o", LIB]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        # PyDLL retains the GIL for the call.  Besides making mutations atomic
        # from Python's perspective, addr() below creates a live buffer export
        # which prevents another thread from resizing a bytearray mid-call.
        _library = ctypes.PyDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def addr(data: bytearray):
    """Return a one-byte-stride ctypes view kept alive for the FFI call."""
    if not isinstance(data, bytearray):
        raise TypeError("FFI buffers must be bytearrays")
    if not data:
        raise ValueError("empty buffers have no FFI address")
    return (ctypes.c_ubyte * len(data)).from_buffer(data)


def numeric_addr(data: bytearray) -> int:
    """Return an address for metadata only; never use it for an FFI call."""
    return ctypes.addressof(addr(data))


def main() -> int:
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
