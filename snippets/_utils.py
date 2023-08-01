import numpy as np


def get_numpy_bytes_len(dtype: np.dtype, shape: tuple) -> int:
    return np.dtype(dtype).itemsize * np.prod(shape, dtype=np.int64)


MAX_NDIM = 8
INT_BYTE_SIZE = 4  # bytes
# dtype, ndim, *shape
INIT_MESSAGE_LEN = INT_BYTE_SIZE + INT_BYTE_SIZE + MAX_NDIM * INT_BYTE_SIZE

DTYPE_TO_INT = {
    np.int32: 0,
}
INT_TO_DTYPE = {
    0: np.int32,
}


def encode_init_message(dtype: np.dtype, shape: tuple) -> str:
    # dtype, ndim, *shape
    int_to_encode = (DTYPE_TO_INT[dtype], len(shape), *shape)
    int_byte_str = b"".join([i.to_bytes(INT_BYTE_SIZE, 'big')
                             for i in int_to_encode])
    assert len(int_byte_str) <= INIT_MESSAGE_LEN

    # complete with zeros
    int_byte_str += b" " * (INIT_MESSAGE_LEN - len(int_byte_str))

    return int_byte_str


def decode_init_message(byte_str: str) -> tuple:
    dtype_int, ndim = [int.from_bytes(byte_str[off:off + INT_BYTE_SIZE], 'big')
                       for off in range(0, 2 * INT_BYTE_SIZE, INT_BYTE_SIZE)]

    shape = tuple([int.from_bytes(byte_str[off: off + INT_BYTE_SIZE], 'big')
                   for off in range(2 * INT_BYTE_SIZE, (2 + ndim) * INT_BYTE_SIZE, INT_BYTE_SIZE)])

    return INT_TO_DTYPE[dtype_int], shape


if __name__ == "__main__":
    DTYPE = np.int32
    SHAPE = (10,)

    init_msg = encode_init_message(DTYPE, SHAPE)
    dtype, shape = decode_init_message(init_msg)

    print(dtype, shape)
