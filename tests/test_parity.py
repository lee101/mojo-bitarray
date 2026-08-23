import io
import random

import pytest
from bitarray import bitarray as upstream_bitarray
from bitarray.util import (
    any_and as upstream_any_and,
    count_and as upstream_count_and,
    count_or as upstream_count_or,
    count_xor as upstream_count_xor,
    parity as upstream_parity,
    subset as upstream_subset,
)

from mojo_bitarray import bitarray, bits2bytes, get_default_endian
from mojo_bitarray.util import (
    any_and,
    count_and,
    count_or,
    count_xor,
    ones,
    parity,
    subset,
    zeros,
)


def pair(bits="", endian="big"):
    return bitarray(bits, endian=endian), upstream_bitarray(bits, endian=endian)


def assert_same(actual, expected):
    assert actual.tolist() == expected.tolist()
    assert actual.tobytes() == expected.tobytes()
    assert len(actual) == len(expected)
    assert actual.endian == expected.endian
    assert actual.nbytes == expected.nbytes
    assert actual.padbits == expected.padbits


@pytest.mark.parametrize("endian", ["big", "little"])
@pytest.mark.parametrize(
    "initializer",
    ["", "0", "101", "10110010", "101100101", [1, 0, True, False], b"\x00\xa5\xff", 19],
)
def test_construction(initializer, endian):
    actual = bitarray(initializer, endian=endian)
    expected = upstream_bitarray(initializer, endian=endian)
    assert_same(actual, expected)
    assert actual.to01() == expected.to01()
    assert actual.tolist() == expected.tolist()
    assert repr(actual) == repr(expected)


@pytest.mark.parametrize("n", [0, 1, 7, 8, 9, 127])
def test_bits2bytes(n):
    assert bits2bytes(n) == (n + 7) // 8
    assert get_default_endian() == "big"


@pytest.mark.parametrize("endian", ["big", "little"])
@pytest.mark.parametrize("n", [0, 1, 2, 7, 8, 9, 63, 64, 65, 1003])
def test_random_bitwise_count_invert_and_shifts(endian, n):
    rng = random.Random(1000 + n)
    left_bits = [rng.randrange(2) for _ in range(n)]
    right_bits = [rng.randrange(2) for _ in range(n)]
    actual_a, expected_a = pair(left_bits, endian)
    actual_b, expected_b = pair(right_bits, endian)

    assert actual_a.count() == expected_a.count()
    assert actual_a.count(0) == expected_a.count(0)
    assert_same(~actual_a, ~expected_a)
    assert_same(actual_a & actual_b, expected_a & expected_b)
    assert_same(actual_a | actual_b, expected_a | expected_b)
    assert_same(actual_a ^ actual_b, expected_a ^ expected_b)

    for amount in (0, 1, 5, n, n + 3):
        assert_same(actual_a << amount, expected_a << amount)
        assert_same(actual_a >> amount, expected_a >> amount)

    for symbol in ("and", "or", "xor"):
        ma, ua = actual_a.copy(), expected_a.copy()
        if symbol == "and":
            ma &= actual_b
            ua &= expected_b
        elif symbol == "or":
            ma |= actual_b
            ua |= expected_b
        else:
            ma ^= actual_b
            ua ^= expected_b
        assert_same(ma, ua)


@pytest.mark.parametrize("endian", ["big", "little"])
@pytest.mark.parametrize("n", [8 * 31 + 3, 8 * 32, 8 * 32 + 5, 8 * 33])
def test_shift_simd_tail(endian, n):
    rng = random.Random(2000 + n)
    bits = [rng.randrange(2) for _ in range(n)]
    actual, expected = pair(bits, endian)

    for amount in (0, 1, 7, 8, 13, n - 1, n, n + 1):
        assert_same(actual << amount, expected << amount)
        assert_same(actual >> amount, expected >> amount)

        inplace_actual, inplace_expected = actual.copy(), expected.copy()
        inplace_actual <<= amount
        inplace_expected <<= amount
        assert_same(inplace_actual, inplace_expected)

        inplace_actual, inplace_expected = actual.copy(), expected.copy()
        inplace_actual >>= amount
        inplace_expected >>= amount
        assert_same(inplace_actual, inplace_expected)


@pytest.mark.parametrize("endian", ["big", "little"])
@pytest.mark.parametrize("n", [8 * 127 + 3, 8 * 128, 8 * 128 + 5, 8 * 129])
def test_bitwise_simd_unroll_tail(endian, n):
    rng = random.Random(3000 + n)
    left = [rng.randrange(2) for _ in range(n)]
    right = [rng.randrange(2) for _ in range(n)]
    actual_a, expected_a = pair(left, endian)
    actual_b, expected_b = pair(right, endian)

    assert_same(actual_a & actual_b, expected_a & expected_b)
    assert_same(actual_a | actual_b, expected_a | expected_b)
    assert_same(actual_a ^ actual_b, expected_a ^ expected_b)
    assert_same(~actual_a, ~expected_a)


def test_parallel_threshold():
    nbytes = 128 * 1024 * 1024 + 37
    actual_a = bitarray(b"\xaa" * nbytes)
    actual_b = bitarray(b"\xcc" * nbytes)

    combined = actual_a & actual_b
    assert combined.tobytes() == b"\x88" * nbytes
    del combined

    inverted = ~actual_a
    assert inverted.tobytes() == b"\x55" * nbytes


@pytest.mark.parametrize("endian", ["big", "little"])
def test_sequence_mutations(endian):
    actual, expected = pair("101001", endian)

    operations = [
        lambda x: x.append(1),
        lambda x: x.extend("001"),
        lambda x: x.insert(2, 0),
        lambda x: x.__setitem__(slice(1, 4), bitarray("11", endian=endian))
        if isinstance(x, bitarray)
        else x.__setitem__(slice(1, 4), upstream_bitarray("11", endian=endian)),
        lambda x: x.__setitem__(slice(None, None, 2), 0),
        lambda x: x.__delitem__(slice(2, 5)),
        lambda x: x.reverse(),
        lambda x: x.sort(reverse=True),
    ]
    for operation in operations:
        operation(actual)
        operation(expected)
        assert_same(actual, expected)

    assert actual.pop() == expected.pop()
    actual.remove(1)
    expected.remove(1)
    assert_same(actual, expected)
    actual.clear()
    expected.clear()
    assert_same(actual, expected)


@pytest.mark.parametrize("endian", ["big", "little"])
def test_slices_concat_and_repeat(endian):
    actual, expected = pair("10110010110", endian)
    for item in [0, -1, slice(None), slice(1, 9), slice(None, None, 2), slice(None, None, -1)]:
        left, right = actual[item], expected[item]
        if isinstance(item, slice):
            assert_same(left, right)
        else:
            assert left == right
    assert_same(actual + actual, expected + expected)
    assert_same(actual * 3, expected * 3)
    assert_same(2 * actual, 2 * expected)
    inplace_actual, inplace_expected = actual.copy(), expected.copy()
    inplace_actual += actual
    inplace_expected += expected
    assert_same(inplace_actual, inplace_expected)
    inplace_actual *= 2
    inplace_expected *= 2
    assert_same(inplace_actual, inplace_expected)


def test_comparisons_and_index_assignment():
    actual, expected = pair("10110")
    for other_bits in ("", "1011", "10110", "10111", "110"):
        other_actual, other_expected = pair(other_bits)
        assert (actual == other_actual) == (expected == other_expected)
        assert (actual < other_actual) == (expected < other_expected)
        assert (actual <= other_actual) == (expected <= other_expected)
        assert (actual > other_actual) == (expected > other_expected)
        assert (actual >= other_actual) == (expected >= other_expected)
    actual[-1] = 1
    expected[-1] = 1
    assert_same(actual, expected)


@pytest.mark.parametrize("endian", ["big", "little"])
def test_search_find_index_and_membership(endian):
    actual, expected = pair("0010110110110", endian)
    for value in (0, 1):
        for start, stop in [(0, None), (2, 11), (-8, -1)]:
            if stop is None:
                assert actual.find(value, start) == expected.find(value, start)
                assert actual.find(value, start, right=True) == expected.find(
                    value, start, right=True
                )
            else:
                assert actual.find(value, start, stop) == expected.find(value, start, stop)
                assert actual.find(value, start, stop, right=True) == expected.find(
                    value, start, stop, right=True
                )
    for pattern in ("1", "10", "110", "111"):
        ma = bitarray(pattern, endian=endian)
        ua = upstream_bitarray(pattern, endian=endian)
        assert actual.find(ma) == expected.find(ua)
        assert actual.find(ma, right=True) == expected.find(ua, right=True)
        assert list(actual.search(ma)) == list(expected.search(ua))
        assert list(actual.search(ma, right=True)) == list(expected.search(ua, right=True))
        assert (ma in actual) == (ua in expected)
    with pytest.raises(ValueError):
        actual.index(bitarray("111"))


@pytest.mark.parametrize("endian", ["big", "little"])
def test_bytes_pack_unpack_and_files(endian):
    actual, expected = pair("101", endian)
    actual.frombytes(b"\xa5\x03")
    expected.frombytes(b"\xa5\x03")
    assert_same(actual, expected)
    assert actual.unpack(b"F", b"T") == expected.unpack(b"F", b"T")

    actual.pack(b"\x00\x02\x00\xff")
    expected.pack(b"\x00\x02\x00\xff")
    assert_same(actual, expected)

    afile, efile = io.BytesIO(), io.BytesIO()
    actual.tofile(afile)
    expected.tofile(efile)
    assert afile.getvalue() == efile.getvalue()

    more_actual, more_expected = pair("", endian)
    more_actual.fromfile(io.BytesIO(b"\x12\x34"), 2)
    more_expected.fromfile(io.BytesIO(b"\x12\x34"), 2)
    assert_same(more_actual, more_expected)


@pytest.mark.parametrize("endian", ["big", "little"])
@pytest.mark.parametrize("bits", ["", "1", "101", "11110000", "101010101"])
def test_setall_fill_reverse_and_bytereverse(endian, bits):
    for value in (0, 1):
        actual, expected = pair(bits, endian)
        actual.setall(value)
        expected.setall(value)
        assert_same(actual, expected)

    actual, expected = pair(bits, endian)
    assert actual.fill() == expected.fill()
    assert_same(actual, expected)

    actual, expected = pair(bits, endian)
    actual.reverse()
    expected.reverse()
    assert_same(actual, expected)

    actual, expected = pair(bits, endian)
    actual.tobytes()
    expected.tobytes()
    actual.bytereverse()
    expected.bytereverse()
    assert_same(actual, expected)


@pytest.mark.parametrize("endian", ["big", "little"])
def test_count_ranges_all_any_sort_and_invert_method(endian):
    actual, expected = pair("010110010111", endian)
    for args in [(1,), (0,), (1, 2, 10), (0, 1, 11, 2), (1, 0, 12, -1)]:
        assert actual.count(*args) == expected.count(*args)
    assert actual.all() == expected.all()
    assert actual.any() == expected.any()
    actual.invert()
    expected.invert()
    assert_same(actual, expected)
    actual.invert(3)
    expected.invert(3)
    assert_same(actual, expected)
    for reverse in (False, True):
        ma, ua = actual.copy(), expected.copy()
        ma.sort(reverse=reverse)
        ua.sort(reverse=reverse)
        assert_same(ma, ua)


@pytest.mark.parametrize("endian", ["big", "little"])
def test_fused_utilities(endian):
    ma, ua = pair("101101001001011", endian)
    mb, ub = pair("110001011101001", endian)
    assert count_and(ma, mb) == upstream_count_and(ua, ub)
    assert count_or(ma, mb) == upstream_count_or(ua, ub)
    assert count_xor(ma, mb) == upstream_count_xor(ua, ub)
    assert any_and(ma, mb) == upstream_any_and(ua, ub)
    assert subset(ma, mb) == upstream_subset(ua, ub)
    assert parity(ma) == upstream_parity(ua)
    assert subset(ma & mb, ma)
    assert_same(zeros(15, endian=endian), upstream_bitarray(15, endian=endian))
    expected_ones = upstream_bitarray(15, endian=endian)
    expected_ones.setall(1)
    assert_same(ones(15, endian=endian), expected_ones)


def test_encode_decode_prefix_code():
    mojo_code = {"a": bitarray("0"), "b": bitarray("10"), "c": bitarray("11")}
    upstream_code = {
        "a": upstream_bitarray("0"),
        "b": upstream_bitarray("10"),
        "c": upstream_bitarray("11"),
    }
    actual, expected = bitarray(), upstream_bitarray()
    actual.encode(mojo_code, "abacaba")
    expected.encode(upstream_code, "abacaba")
    assert_same(actual, expected)
    assert list(actual.decode(mojo_code)) == list(expected.decode(upstream_code))


@pytest.mark.parametrize(
    "code",
    [
        {"a": bitarray("0"), "b": bitarray("0")},
        {"a": bitarray("0"), "b": bitarray("01")},
        {"a": bitarray(), "b": bitarray("1")},
    ],
)
def test_decode_rejects_ambiguous_prefix_codes(code):
    with pytest.raises(ValueError, match="prefix code ambiguous"):
        bitarray("001").decode(code)


def test_validation_errors():
    with pytest.raises(ValueError):
        bitarray(-1)
    with pytest.raises(ValueError):
        bitarray(endian="middle")
    with pytest.raises(ValueError):
        bitarray("012")
    with pytest.raises(ValueError):
        bitarray("1") << -1
    with pytest.raises(ValueError):
        bitarray("1") & bitarray("10")
    with pytest.raises(ValueError):
        bitarray("1", endian="big") & bitarray("1", endian="little")
    with pytest.raises(ValueError):
        bitarray("10").append(2)


def test_buffer_metadata_and_copy_independence():
    actual = bitarray("101")
    info = actual.buffer_info()
    assert info.nbytes == 1
    assert info.endian == "big"
    assert info.padbits == 5
    assert not info.readonly
    copied = actual.copy()
    copied[0] = 0
    assert actual.to01() == "101"
    assert copied.to01() == "001"
