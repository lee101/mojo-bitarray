"""Python sequence API backed by packed bytes and Mojo kernels."""

from __future__ import annotations

import operator
from collections import namedtuple
from collections.abc import Iterable, Iterator, Mapping

from ._lib import addr, lib, numeric_addr

BufferInfo = namedtuple(
    "BufferInfo",
    "address nbytes endian padbits alloc readonly imported exports",
)


def bits2bytes(n: int, /) -> int:
    n = operator.index(n)
    if n < 0:
        raise ValueError("non-negative integer expected")
    return (n + 7) // 8


def get_default_endian() -> str:
    return "big"


def _endian(value: str) -> str:
    if value not in ("big", "little"):
        raise ValueError("bit-endianness must be either 'little' or 'big'")
    return value


def _bit_value(value) -> int:
    value = operator.index(value)
    if value not in (0, 1):
        raise ValueError(f"bit must be 0 or 1, got {value}")
    return value


class bitarray:
    """bitarray(initializer=0, /, endian='big', buffer=None) -> bitarray"""

    __slots__ = ("_data", "_nbits", "_endian")
    __hash__ = None

    def __init__(self, initializer=0, /, endian="big", buffer=None):
        self._endian = _endian(endian)
        self._data = bytearray()
        self._nbits = 0
        if buffer is not None:
            if initializer not in (0, None):
                raise TypeError("buffer requires an empty initializer")
            self._data = bytearray(memoryview(buffer).cast("B"))
            self._nbits = 8 * len(self._data)
            return
        if initializer is None:
            return
        if isinstance(initializer, bitarray):
            self._data = bytearray(initializer._data)
            self._nbits = initializer._nbits
            if endian == "big":
                self._endian = initializer._endian
            return
        if isinstance(initializer, str):
            self.extend(initializer)
            return
        if isinstance(initializer, (bytes, bytearray, memoryview)):
            self.frombytes(initializer)
            return
        try:
            length = operator.index(initializer)
        except TypeError:
            self.extend(initializer)
        else:
            if length < 0:
                raise ValueError("bitarray length must be >= 0")
            self._nbits = length
            self._data = bytearray(bits2bytes(length))

    @classmethod
    def _from_storage(cls, data: bytes | bytearray, nbits: int, endian: str):
        obj = cls.__new__(cls)
        obj._data = bytearray(data)
        obj._nbits = nbits
        obj._endian = endian
        obj._clear_padding()
        return obj

    def _mask(self, index: int) -> int:
        shift = index & 7
        return 1 << (shift if self._endian == "little" else 7 - shift)

    def _get(self, index: int) -> int:
        return int(bool(self._data[index >> 3] & self._mask(index)))

    def _put(self, index: int, value: int) -> None:
        byte = index >> 3
        mask = self._mask(index)
        if value:
            self._data[byte] |= mask
        else:
            self._data[byte] &= ~mask

    def _clear_padding(self) -> None:
        remaining = self._nbits & 7
        if not remaining or not self._data:
            return
        mask = (1 << remaining) - 1
        if self._endian == "big":
            mask <<= 8 - remaining
        self._data[-1] &= mask

    def _replace_bits(self, values: Iterable[int]) -> None:
        values = list(values)
        self._nbits = len(values)
        self._data = bytearray(bits2bytes(self._nbits))
        for i, value in enumerate(values):
            self._put(i, _bit_value(value))

    def _coerce_other(self, other):
        if not isinstance(other, bitarray):
            return NotImplemented
        if self._nbits != other._nbits:
            raise ValueError("bitarrays of equal length expected")
        if self._endian != other._endian:
            raise ValueError("bitarrays of equal bit-endianness expected")
        return other

    @property
    def endian(self) -> str:
        return self._endian

    @property
    def nbytes(self) -> int:
        return len(self._data)

    @property
    def padbits(self) -> int:
        return 8 * len(self._data) - self._nbits

    @property
    def readonly(self) -> bool:
        return False

    def buffer_info(self) -> BufferInfo:
        address = numeric_addr(self._data) if self._data else 0
        return BufferInfo(
            address, self.nbytes, self.endian, self.padbits, self.nbytes, False, False, 0
        )

    def __len__(self) -> int:
        return self._nbits

    def __iter__(self) -> Iterator[int]:
        for i in range(self._nbits):
            yield self._get(i)

    def __reversed__(self) -> Iterator[int]:
        for i in range(self._nbits - 1, -1, -1):
            yield self._get(i)

    def __getitem__(self, key):
        if isinstance(key, slice):
            indices = range(*key.indices(self._nbits))
            result = bitarray(0, endian=self._endian)
            result._replace_bits(self._get(i) for i in indices)
            return result
        index = operator.index(key)
        if index < 0:
            index += self._nbits
        if not 0 <= index < self._nbits:
            raise IndexError("bitarray index out of range")
        return self._get(index)

    def __setitem__(self, key, value) -> None:
        if not isinstance(key, slice):
            index = operator.index(key)
            if index < 0:
                index += self._nbits
            if not 0 <= index < self._nbits:
                raise IndexError("bitarray assignment index out of range")
            self._put(index, _bit_value(value))
            return
        indices = list(range(*key.indices(self._nbits)))
        try:
            scalar = _bit_value(value)
        except (TypeError, ValueError):
            scalar = None
        if scalar is not None:
            for index in indices:
                self._put(index, scalar)
            return
        if not isinstance(value, bitarray):
            raise TypeError(
                f"bitarray or int expected for slice assignment, not '{type(value).__name__}'"
            )
        replacement = value.tolist()
        if key.step not in (None, 1) and len(replacement) != len(indices):
            raise ValueError(
                f"attempt to assign sequence of size {len(replacement)} "
                f"to extended slice of size {len(indices)}"
            )
        values = self.tolist()
        values[key] = replacement
        self._replace_bits(values)

    def __delitem__(self, key) -> None:
        values = self.tolist()
        del values[key]
        self._replace_bits(values)

    def __repr__(self) -> str:
        return f"bitarray('{self.to01()}')" if self else "bitarray()"

    __str__ = __repr__

    def __eq__(self, other) -> bool:
        if not isinstance(other, bitarray):
            return False
        return self.tolist() == other.tolist()

    def __lt__(self, other):
        if not isinstance(other, bitarray):
            return NotImplemented
        return self.tolist() < other.tolist()

    def __le__(self, other):
        if not isinstance(other, bitarray):
            return NotImplemented
        return self.tolist() <= other.tolist()

    def __gt__(self, other):
        if not isinstance(other, bitarray):
            return NotImplemented
        return self.tolist() > other.tolist()

    def __ge__(self, other):
        if not isinstance(other, bitarray):
            return NotImplemented
        return self.tolist() >= other.tolist()

    def copy(self):
        return type(self)._from_storage(self._data, self._nbits, self._endian)

    __copy__ = copy

    def tolist(self) -> list[int]:
        return list(self)

    def to01(self) -> str:
        return "".join("1" if bit else "0" for bit in self)

    def tobytes(self) -> bytes:
        return bytes(self._data)

    def frombytes(self, data, /) -> None:
        raw = bytes(data)
        if not raw:
            return
        if self.padbits:
            self.extend(bitarray(raw, endian=self._endian))
        else:
            self._data.extend(raw)
            self._nbits += 8 * len(raw)

    def unpack(self, zero=b"\x00", one=b"\x01") -> bytes:
        zero, one = bytes(zero), bytes(one)
        if len(zero) != 1 or len(one) != 1:
            raise ValueError("unpack bytes must be length 1")
        return b"".join(one if bit else zero for bit in self)

    def pack(self, data, /) -> None:
        self.extend(1 if value else 0 for value in bytes(data))

    def tofile(self, file, /) -> None:
        file.write(self.tobytes())

    def fromfile(self, file, n=-1, /) -> None:
        n = operator.index(n)
        data = file.read() if n < 0 else file.read(n)
        if n >= 0 and len(data) < n:
            self.frombytes(data)
            raise EOFError("not enough bytes in file")
        self.frombytes(data)

    def append(self, value, /) -> None:
        value = _bit_value(value)
        if not (self._nbits & 7):
            self._data.append(0)
        self._nbits += 1
        self._put(self._nbits - 1, value)

    def extend(self, iterable, /) -> None:
        if isinstance(iterable, str):
            for char in iterable:
                if char.isspace() or char == "_":
                    continue
                if char not in "01":
                    raise ValueError(f"expected '0' or '1' (or whitespace), got {char!r}")
                self.append(char == "1")
            return
        if isinstance(iterable, bitarray):
            for value in iterable:
                self.append(value)
            return
        for value in iterable:
            self.append(value)

    def insert(self, index, value, /) -> None:
        values = self.tolist()
        values.insert(operator.index(index), _bit_value(value))
        self._replace_bits(values)

    def pop(self, index=-1, /) -> int:
        values = self.tolist()
        value = values.pop(operator.index(index))
        self._replace_bits(values)
        return value

    def remove(self, value, /) -> None:
        values = self.tolist()
        values.remove(_bit_value(value))
        self._replace_bits(values)

    def clear(self) -> None:
        self._data.clear()
        self._nbits = 0

    def setall(self, value, /) -> None:
        value = _bit_value(value)
        if self._data:
            lib().mba_setall(
                addr(self._data),
                self.nbytes,
                self._nbits,
                value,
                self._endian == "little",
            )

    def fill(self) -> int:
        added = self.padbits
        self._nbits += added
        return added

    def all(self) -> bool:
        return self.count(1) == self._nbits

    def any(self) -> bool:
        return bool(self.count(1))

    def count(self, value=1, start=0, stop=None, step=1, /) -> int:
        value = _bit_value(value)
        start = operator.index(start)
        stop = self._nbits if stop is None else operator.index(stop)
        step = operator.index(step)
        indices = range(*slice(start, stop, step).indices(self._nbits))
        if step == 1 and indices.start == 0 and indices.stop == self._nbits:
            ones = (
                lib().mba_count(addr(self._data), self.nbytes, self._nbits)
                if self._data
                else 0
            )
            return ones if value else self._nbits - ones
        return sum(self._get(i) == value for i in indices)

    def find(self, sub, start=0, stop=None, /, *, right=False) -> int:
        start = operator.index(start)
        stop = self._nbits if stop is None else operator.index(stop)
        start, stop, _ = slice(start, stop, 1).indices(self._nbits)
        if isinstance(sub, bitarray):
            width = len(sub)
            if width == 0:
                return stop if right else start
            positions = range(stop - width, start - 1, -1) if right else range(
                start, stop - width + 1
            )
            target = sub.tolist()
            for pos in positions:
                if self[pos : pos + width].tolist() == target:
                    return pos
            return -1
        value = _bit_value(sub)
        if right:
            for index in range(stop - 1, start - 1, -1):
                if self._get(index) == value:
                    return index
            return -1
        if not self._data:
            return -1
        return lib().mba_find(
            addr(self._data),
            self._nbits,
            value,
            start,
            stop,
            self._endian == "little",
        )

    def index(self, sub, start=0, stop=None, /, *, right=False) -> int:
        result = self.find(sub, start, stop, right=right)
        if result < 0:
            raise ValueError("sub-bitarray not found")
        return result

    def search(self, sub, start=0, stop=None, /, *, right=False) -> Iterator[int]:
        if not isinstance(sub, bitarray):
            sub = bitarray([_bit_value(sub)], endian=self._endian)
        if not sub:
            raise ValueError("can't search for empty bitarray")
        start = operator.index(start)
        stop = self._nbits if stop is None else operator.index(stop)
        start, stop, _ = slice(start, stop, 1).indices(self._nbits)
        matches = [
            i
            for i in range(start, stop - len(sub) + 1)
            if self[i : i + len(sub)] == sub
        ]
        return iter(reversed(matches) if right else matches)

    def __contains__(self, item) -> bool:
        try:
            return self.find(item) >= 0
        except (TypeError, ValueError):
            return False

    def sort(self, *, reverse=False) -> None:
        ones = self.count()
        self.setall(0 if reverse else 1)
        split = ones if reverse else self._nbits - ones
        for i in range(split):
            self._put(i, 1 if reverse else 0)

    def reverse(self) -> None:
        if not self._data:
            return
        dst = bytearray(self.nbytes)
        lib().mba_reverse(
            addr(self._data), addr(dst), self._nbits, self._endian == "little"
        )
        self._data = dst

    def bytereverse(self, start=0, stop=None, /) -> None:
        stop = self.nbytes if stop is None else operator.index(stop)
        start, stop, _ = slice(operator.index(start), stop, 1).indices(self.nbytes)
        if self._data and start < stop:
            lib().mba_bytereverse(addr(self._data), start, stop)
            self._clear_padding()

    def _binary(self, other, operation: int, inplace: bool):
        other = self._coerce_other(other)
        if other is NotImplemented:
            return NotImplemented
        result = self if inplace else self.copy()
        if self._data:
            lib().mba_binary(
                addr(result._data),
                addr(other._data),
                addr(result._data),
                self.nbytes,
                operation,
            )
            result._clear_padding()
        return result

    def __and__(self, other):
        return self._binary(other, 0, False)

    def __or__(self, other):
        return self._binary(other, 1, False)

    def __xor__(self, other):
        return self._binary(other, 2, False)

    def __iand__(self, other):
        return self._binary(other, 0, True)

    def __ior__(self, other):
        return self._binary(other, 1, True)

    def __ixor__(self, other):
        return self._binary(other, 2, True)

    def __invert__(self):
        result = self.copy()
        if self._data:
            lib().mba_invert(
                addr(result._data),
                addr(result._data),
                self.nbytes,
                self._nbits,
                self._endian == "little",
            )
        return result

    def invert(self, index=None, /) -> None:
        if index is None:
            if self._data:
                lib().mba_invert(
                    addr(self._data),
                    addr(self._data),
                    self.nbytes,
                    self._nbits,
                    self._endian == "little",
                )
            return
        index = operator.index(index)
        self[index] = 1 - self[index]

    def _shift(self, amount, direction: int, inplace: bool):
        try:
            amount = operator.index(amount)
        except TypeError:
            return NotImplemented
        if amount < 0:
            raise ValueError("negative shift count")
        result = self if inplace else bitarray(self._nbits, endian=self._endian)
        if self._data:
            source = self._data if not inplace else bytearray(self._data)
            lib().mba_shift(
                addr(source),
                addr(result._data),
                self._nbits,
                amount,
                direction,
                self._endian == "little",
            )
        return result

    def __lshift__(self, amount):
        return self._shift(amount, 0, False)

    def __rshift__(self, amount):
        return self._shift(amount, 1, False)

    def __ilshift__(self, amount):
        return self._shift(amount, 0, True)

    def __irshift__(self, amount):
        return self._shift(amount, 1, True)

    def __add__(self, other):
        if not isinstance(other, bitarray):
            return NotImplemented
        result = self.copy()
        result.extend(other)
        return result

    def __iadd__(self, other):
        if not isinstance(other, bitarray):
            return NotImplemented
        self.extend(other)
        return self

    def __mul__(self, count):
        try:
            count = operator.index(count)
        except TypeError:
            return NotImplemented
        result = bitarray(0, endian=self._endian)
        result._replace_bits(self.tolist() * max(count, 0))
        return result

    __rmul__ = __mul__

    def __imul__(self, count):
        result = self * count
        self._data, self._nbits = result._data, result._nbits
        return self

    def encode(self, code: Mapping, iterable: Iterable, /) -> None:
        for symbol in iterable:
            self.extend(code[symbol])

    def decode(self, code: Mapping, /) -> Iterator:
        inverse = {}
        for symbol, bits in code.items():
            key = tuple(bits)
            if not key or key in inverse or any(
                key[: len(other)] == other or other[: len(key)] == key
                for other in inverse
            ):
                raise ValueError(f"prefix code ambiguous: {symbol!r}")
            inverse[key] = symbol
        prefix = []
        decoded = []
        for value in self:
            prefix.append(value)
            key = tuple(prefix)
            if key in inverse:
                decoded.append(inverse[key])
                prefix.clear()
            elif not any(candidate[: len(key)] == key for candidate in inverse):
                raise ValueError("prefix code does not match data")
        if prefix:
            raise ValueError("incomplete prefix code")
        return iter(decoded)
