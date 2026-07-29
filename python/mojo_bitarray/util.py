"""Covered helpers from :mod:`bitarray.util`."""

from __future__ import annotations

from ._lib import addr, lib
from .core import bitarray


def _pair(a: bitarray, b: bitarray) -> None:
    if not isinstance(a, bitarray) or not isinstance(b, bitarray):
        raise TypeError("bitarray expected")
    if len(a) != len(b):
        raise ValueError("bitarrays of equal length expected")
    if a.endian != b.endian:
        raise ValueError("bitarrays of equal bit-endianness expected")


def count_and(a: bitarray, b: bitarray, /) -> int:
    _pair(a, b)
    return lib().mba_count_binary(addr(a._data), addr(b._data), a.nbytes, 0) if a.nbytes else 0


def count_or(a: bitarray, b: bitarray, /) -> int:
    _pair(a, b)
    return lib().mba_count_binary(addr(a._data), addr(b._data), a.nbytes, 1) if a.nbytes else 0


def count_xor(a: bitarray, b: bitarray, /) -> int:
    _pair(a, b)
    return lib().mba_count_binary(addr(a._data), addr(b._data), a.nbytes, 2) if a.nbytes else 0


def any_and(a: bitarray, b: bitarray, /) -> bool:
    return bool(count_and(a, b))


def subset(a: bitarray, b: bitarray, /) -> bool:
    _pair(a, b)
    return bool(lib().mba_subset(addr(a._data), addr(b._data), a.nbytes)) if a.nbytes else True


def parity(a: bitarray, /) -> int:
    if not isinstance(a, bitarray):
        raise TypeError("bitarray expected")
    return a.count() & 1


def zeros(length: int, /, endian="big") -> bitarray:
    return bitarray(length, endian=endian)


def ones(length: int, /, endian="big") -> bitarray:
    result = bitarray(length, endian=endian)
    result.setall(1)
    return result
