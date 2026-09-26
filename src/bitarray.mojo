"""Packed-bit kernels exposed through a stable C ABI."""

from std.sys.info import simd_width_of as simdwidthof

comptime BPtr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime U64Ptr = Pointer[UInt64, AnyOrigin[mut=True]]
comptime W = simdwidthof[DType.float64]()
comptime VECTOR_BYTES = W * 8
comptime UNROLL = 4
comptime PARALLEL_THRESHOLD = 128 << 20
comptime PARALLEL_TASKS = 4


def bit_mask(index: Int, little: Bool) -> UInt8:
    var shift = index & 7
    if not little:
        shift = 7 - shift
    return UInt8(1) << UInt8(shift)


def get_bit(data: BPtr, index: Int, little: Bool) -> Bool:
    return (data[unsafe_offset=index >> 3] & bit_mask(index, little)) != 0


def put_bit(data: BPtr, index: Int, value: Bool, little: Bool):
    var byte_index = index >> 3
    var mask = bit_mask(index, little)
    if value:
        data[unsafe_offset=byte_index] |= mask
    else:
        data[unsafe_offset=byte_index] &= ~mask


def pop_count(byte: UInt8) -> Int:
    var v = byte
    var count = 0
    while v != 0:
        v &= v - 1
        count += 1
    return count


def pop_count64(value: UInt64) -> Int:
    var v = value
    v -= (v >> 1) & UInt64(0x5555555555555555)
    v = (v & UInt64(0x3333333333333333)) + (
        (v >> 2) & UInt64(0x3333333333333333)
    )
    v = (v + (v >> 4)) & UInt64(0x0F0F0F0F0F0F0F0F)
    return Int((v * UInt64(0x0101010101010101)) >> 56)


def clear_padding(data: BPtr, nbits: Int, little: Bool):
    var remaining = nbits & 7
    if remaining == 0 or nbits == 0:
        return
    var mask = UInt8((1 << remaining) - 1)
    if not little:
        mask <<= UInt8(8 - remaining)
    data[unsafe_offset=nbits >> 3] &= mask


def binary_range(
    a: BPtr, b: BPtr, dst: BPtr, start: Int, stop: Int, operation: Int
):
    var a64 = a.unsafe_offset(start).unsafe_bitcast[UInt64]()
    var b64 = b.unsafe_offset(start).unsafe_bitcast[UInt64]()
    var dst64 = dst.unsafe_offset(start).unsafe_bitcast[UInt64]()
    var words = (stop - start) >> 3
    var word = 0
    if operation == 0:
        while word + W * UNROLL <= words:
            comptime for lane in range(UNROLL):
                dst64.unsafe_store[alignment=1](
                    word + lane * W,
                    a64.unsafe_load[width=W, alignment=1](word + lane * W)
                    & b64.unsafe_load[width=W, alignment=1](word + lane * W),
                )
            word += W * UNROLL
        while word + W <= words:
            dst64.unsafe_store[alignment=1](
                word,
                a64.unsafe_load[width=W, alignment=1](word)
                & b64.unsafe_load[width=W, alignment=1](word),
            )
            word += W
    elif operation == 1:
        while word + W * UNROLL <= words:
            comptime for lane in range(UNROLL):
                dst64.unsafe_store[alignment=1](
                    word + lane * W,
                    a64.unsafe_load[width=W, alignment=1](word + lane * W)
                    | b64.unsafe_load[width=W, alignment=1](word + lane * W),
                )
            word += W * UNROLL
        while word + W <= words:
            dst64.unsafe_store[alignment=1](
                word,
                a64.unsafe_load[width=W, alignment=1](word)
                | b64.unsafe_load[width=W, alignment=1](word),
            )
            word += W
    else:
        while word + W * UNROLL <= words:
            comptime for lane in range(UNROLL):
                dst64.unsafe_store[alignment=1](
                    word + lane * W,
                    a64.unsafe_load[width=W, alignment=1](word + lane * W)
                    ^ b64.unsafe_load[width=W, alignment=1](word + lane * W),
                )
            word += W * UNROLL
        while word + W <= words:
            dst64.unsafe_store[alignment=1](
                word,
                a64.unsafe_load[width=W, alignment=1](word)
                ^ b64.unsafe_load[width=W, alignment=1](word),
            )
            word += W
    var i = start + (word << 3)
    while i < stop:
        if operation == 0:
            dst[unsafe_offset=i] = a[unsafe_offset=i] & b[unsafe_offset=i]
        elif operation == 1:
            dst[unsafe_offset=i] = a[unsafe_offset=i] | b[unsafe_offset=i]
        else:
            dst[unsafe_offset=i] = a[unsafe_offset=i] ^ b[unsafe_offset=i]
        i += 1


def invert_range(src: BPtr, dst: BPtr, start: Int, stop: Int):
    var src64 = src.unsafe_offset(start).unsafe_bitcast[UInt64]()
    var dst64 = dst.unsafe_offset(start).unsafe_bitcast[UInt64]()
    var words = (stop - start) >> 3
    var word = 0
    while word + W * UNROLL <= words:
        comptime for lane in range(UNROLL):
            dst64.unsafe_store[alignment=1](
                word + lane * W,
                ~src64.unsafe_load[width=W, alignment=1](word + lane * W),
            )
        word += W * UNROLL
    while word + W <= words:
        dst64.unsafe_store[alignment=1](
            word,
            ~src64.unsafe_load[width=W, alignment=1](word),
        )
        word += W
    var i = start + (word << 3)
    while i < stop:
        dst[unsafe_offset=i] = ~src[unsafe_offset=i]
        i += 1


def binary_parallel(a: BPtr, b: BPtr, dst: BPtr, nbytes: Int, operation: Int):
    var chunk_size = ((nbytes + PARALLEL_TASKS - 1) // PARALLEL_TASKS)
    chunk_size = (chunk_size + VECTOR_BYTES - 1) // VECTOR_BYTES * VECTOR_BYTES

    for task in range(PARALLEL_TASKS):
        var start = task * chunk_size
        var stop = min(start + chunk_size, nbytes)
        if start < stop:
            binary_range(a, b, dst, start, stop, operation)


def invert_parallel(src: BPtr, dst: BPtr, nbytes: Int):
    var chunk_size = ((nbytes + PARALLEL_TASKS - 1) // PARALLEL_TASKS)
    chunk_size = (chunk_size + VECTOR_BYTES - 1) // VECTOR_BYTES * VECTOR_BYTES

    for task in range(PARALLEL_TASKS):
        var start = task * chunk_size
        var stop = min(start + chunk_size, nbytes)
        if start < stop:
            invert_range(src, dst, start, stop)


@export("mba_binary")
def mba_binary(
    a_addr: Int, b_addr: Int, dst_addr: Int, nbytes: Int, operation: Int
) abi("C"):
    var a = BPtr(unsafe_from_address=a_addr)
    var b = BPtr(unsafe_from_address=b_addr)
    var dst = BPtr(unsafe_from_address=dst_addr)
    if nbytes >= PARALLEL_THRESHOLD:
        binary_parallel(a, b, dst, nbytes, operation)
    else:
        binary_range(a, b, dst, 0, nbytes, operation)


@export("mba_invert")
def mba_invert(
    src_addr: Int, dst_addr: Int, nbytes: Int, nbits: Int, little_int: Int
) abi("C"):
    var src = BPtr(unsafe_from_address=src_addr)
    var dst = BPtr(unsafe_from_address=dst_addr)
    if nbytes >= PARALLEL_THRESHOLD:
        invert_parallel(src, dst, nbytes)
    else:
        invert_range(src, dst, 0, nbytes)
    clear_padding(dst, nbits, little_int != 0)


@export("mba_count")
def mba_count(src_addr: Int, nbytes: Int, nbits: Int) abi("C") -> Int:
    var src = BPtr(unsafe_from_address=src_addr)
    var total = 0
    var i = 0
    while i + 8 <= nbytes:
        total += pop_count64(
            src.unsafe_offset(i)
            .unsafe_bitcast[UInt64]()
            .unsafe_load[alignment=1]()
        )
        i += 8
    while i < nbytes:
        total += pop_count(src[unsafe_offset=i])
        i += 1
    # Storage maintained by the Python layer has zero padding.
    _ = nbits
    return total


@export("mba_count_binary")
def mba_count_binary(
    a_addr: Int, b_addr: Int, nbytes: Int, operation: Int
) abi("C") -> Int:
    var a = BPtr(unsafe_from_address=a_addr)
    var b = BPtr(unsafe_from_address=b_addr)
    var total = 0
    var i = 0
    if operation == 0:
        while i + 8 <= nbytes:
            var av = (
                a.unsafe_offset(i)
                .unsafe_bitcast[UInt64]()
                .unsafe_load[alignment=1]()
            )
            var bv = (
                b.unsafe_offset(i)
                .unsafe_bitcast[UInt64]()
                .unsafe_load[alignment=1]()
            )
            total += pop_count64(av & bv)
            i += 8
        while i < nbytes:
            total += pop_count(a[unsafe_offset=i] & b[unsafe_offset=i])
            i += 1
    elif operation == 1:
        while i + 8 <= nbytes:
            var av = (
                a.unsafe_offset(i)
                .unsafe_bitcast[UInt64]()
                .unsafe_load[alignment=1]()
            )
            var bv = (
                b.unsafe_offset(i)
                .unsafe_bitcast[UInt64]()
                .unsafe_load[alignment=1]()
            )
            total += pop_count64(av | bv)
            i += 8
        while i < nbytes:
            total += pop_count(a[unsafe_offset=i] | b[unsafe_offset=i])
            i += 1
    else:
        while i + 8 <= nbytes:
            var av = (
                a.unsafe_offset(i)
                .unsafe_bitcast[UInt64]()
                .unsafe_load[alignment=1]()
            )
            var bv = (
                b.unsafe_offset(i)
                .unsafe_bitcast[UInt64]()
                .unsafe_load[alignment=1]()
            )
            total += pop_count64(av ^ bv)
            i += 8
        while i < nbytes:
            total += pop_count(a[unsafe_offset=i] ^ b[unsafe_offset=i])
            i += 1
    return total


@export("mba_subset")
def mba_subset(a_addr: Int, b_addr: Int, nbytes: Int) abi("C") -> Int:
    var a = BPtr(unsafe_from_address=a_addr)
    var b = BPtr(unsafe_from_address=b_addr)
    for i in range(nbytes):
        if (a[unsafe_offset=i] & ~b[unsafe_offset=i]) != 0:
            return 0
    return 1


@export("mba_find")
def mba_find(
    src_addr: Int,
    nbits: Int,
    value_int: Int,
    start: Int,
    stop: Int,
    little_int: Int,
) abi("C") -> Int:
    var src = BPtr(unsafe_from_address=src_addr)
    var value = value_int != 0
    var little = little_int != 0
    var end = min(stop, nbits)
    for i in range(max(start, 0), end):
        if get_bit(src, i, little) == value:
            return i
    return -1


@export("mba_shift")
def mba_shift(
    src_addr: Int,
    dst_addr: Int,
    nbits: Int,
    amount: Int,
    direction: Int,
    little_int: Int,
) abi("C"):
    var src = BPtr(unsafe_from_address=src_addr)
    var dst = BPtr(unsafe_from_address=dst_addr)
    var little = little_int != 0
    var nbytes = (nbits + 7) >> 3
    if amount >= nbits:
        for i in range(nbytes):
            dst[unsafe_offset=i] = 0
        return
    var byte_shift = amount >> 3
    var bit_shift = amount & 7
    if direction == 0 and not little:
        var stop = nbytes - byte_shift
        var i = 0
        if bit_shift == 0:
            while i + W <= stop:
                dst.unsafe_store(i, src.unsafe_load[width=W](i + byte_shift))
                i += W
        else:
            while i + W < stop:
                var source_index = i + byte_shift
                dst.unsafe_store(
                    i,
                    (src.unsafe_load[width=W](source_index) << UInt8(bit_shift))
                    | (
                        src.unsafe_load[width=W](source_index + 1)
                        >> UInt8(8 - bit_shift)
                    ),
                )
                i += W
        while i < stop:
            var source_index = i + byte_shift
            var value = UInt16(src[unsafe_offset=source_index]) << UInt16(
                bit_shift
            )
            if bit_shift != 0 and source_index + 1 < nbytes:
                value |= UInt16(src[unsafe_offset=source_index + 1]) >> UInt16(
                    8 - bit_shift
                )
            dst[unsafe_offset=i] = UInt8(value)
            i += 1
        while i < nbytes:
            dst[unsafe_offset=i] = 0
            i += 1
    elif direction == 0:
        var stop = nbytes - byte_shift
        var i = 0
        if bit_shift == 0:
            while i + W <= stop:
                dst.unsafe_store(i, src.unsafe_load[width=W](i + byte_shift))
                i += W
        else:
            while i + W < stop:
                var source_index = i + byte_shift
                dst.unsafe_store(
                    i,
                    (src.unsafe_load[width=W](source_index) >> UInt8(bit_shift))
                    | (
                        src.unsafe_load[width=W](source_index + 1)
                        << UInt8(8 - bit_shift)
                    ),
                )
                i += W
        while i < stop:
            var source_index = i + byte_shift
            var value = UInt16(src[unsafe_offset=source_index]) >> UInt16(
                bit_shift
            )
            if bit_shift != 0 and source_index + 1 < nbytes:
                value |= UInt16(src[unsafe_offset=source_index + 1]) << UInt16(
                    8 - bit_shift
                )
            dst[unsafe_offset=i] = UInt8(value)
            i += 1
        while i < nbytes:
            dst[unsafe_offset=i] = 0
            i += 1
    elif not little:
        var i = 0
        while i < byte_shift:
            dst[unsafe_offset=i] = 0
            i += 1
        if bit_shift != 0 and i < nbytes:
            var source_index = i - byte_shift
            var value = UInt16(src[unsafe_offset=source_index]) >> UInt16(
                bit_shift
            )
            if bit_shift != 0 and source_index > 0:
                value |= UInt16(src[unsafe_offset=source_index - 1]) << UInt16(
                    8 - bit_shift
                )
            dst[unsafe_offset=i] = UInt8(value)
            i += 1
        if bit_shift == 0:
            while i + W <= nbytes:
                dst.unsafe_store(i, src.unsafe_load[width=W](i - byte_shift))
                i += W
        else:
            while i + W <= nbytes:
                var source_index = i - byte_shift
                dst.unsafe_store(
                    i,
                    (src.unsafe_load[width=W](source_index) >> UInt8(bit_shift))
                    | (
                        src.unsafe_load[width=W](source_index - 1)
                        << UInt8(8 - bit_shift)
                    ),
                )
                i += W
        while i < nbytes:
            var source_index = i - byte_shift
            var value = UInt16(src[unsafe_offset=source_index]) >> UInt16(
                bit_shift
            )
            if bit_shift != 0 and source_index > 0:
                value |= UInt16(src[unsafe_offset=source_index - 1]) << UInt16(
                    8 - bit_shift
                )
            dst[unsafe_offset=i] = UInt8(value)
            i += 1
    else:
        var i = 0
        while i < byte_shift:
            dst[unsafe_offset=i] = 0
            i += 1
        if bit_shift != 0 and i < nbytes:
            var source_index = i - byte_shift
            var value = UInt16(src[unsafe_offset=source_index]) << UInt16(
                bit_shift
            )
            if bit_shift != 0 and source_index > 0:
                value |= UInt16(src[unsafe_offset=source_index - 1]) >> UInt16(
                    8 - bit_shift
                )
            dst[unsafe_offset=i] = UInt8(value)
            i += 1
        if bit_shift == 0:
            while i + W <= nbytes:
                dst.unsafe_store(i, src.unsafe_load[width=W](i - byte_shift))
                i += W
        else:
            while i + W <= nbytes:
                var source_index = i - byte_shift
                dst.unsafe_store(
                    i,
                    (src.unsafe_load[width=W](source_index) << UInt8(bit_shift))
                    | (
                        src.unsafe_load[width=W](source_index - 1)
                        >> UInt8(8 - bit_shift)
                    ),
                )
                i += W
        while i < nbytes:
            var source_index = i - byte_shift
            var value = UInt16(src[unsafe_offset=source_index]) << UInt16(
                bit_shift
            )
            if bit_shift != 0 and source_index > 0:
                value |= UInt16(src[unsafe_offset=source_index - 1]) >> UInt16(
                    8 - bit_shift
                )
            dst[unsafe_offset=i] = UInt8(value)
            i += 1
    clear_padding(dst, nbits, little)


@export("mba_reverse")
def mba_reverse(
    src_addr: Int, dst_addr: Int, nbits: Int, little_int: Int
) abi("C"):
    var src = BPtr(unsafe_from_address=src_addr)
    var dst = BPtr(unsafe_from_address=dst_addr)
    var little = little_int != 0
    var nbytes = (nbits + 7) >> 3
    for i in range(nbytes):
        dst[unsafe_offset=i] = 0
    for i in range(nbits):
        put_bit(dst, i, get_bit(src, nbits - 1 - i, little), little)


@export("mba_setall")
def mba_setall(
    dst_addr: Int, nbytes: Int, nbits: Int, value_int: Int, little_int: Int
) abi("C"):
    var dst = BPtr(unsafe_from_address=dst_addr)
    var value = UInt8(255) if value_int != 0 else UInt8(0)
    for i in range(nbytes):
        dst[unsafe_offset=i] = value
    clear_padding(dst, nbits, little_int != 0)


@export("mba_bytereverse")
def mba_bytereverse(dst_addr: Int, start: Int, stop: Int) abi("C"):
    var dst = BPtr(unsafe_from_address=dst_addr)
    for i in range(start, stop):
        var v = dst[unsafe_offset=i]
        v = ((v & 0x55) << 1) | ((v >> 1) & 0x55)
        v = ((v & 0x33) << 2) | ((v >> 2) & 0x33)
        dst[unsafe_offset=i] = (v << 4) | (v >> 4)
