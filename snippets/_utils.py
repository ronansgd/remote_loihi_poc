import typing as ty

import numpy as np


def get_array_bytes_len(dtype: np.dtype, shape: tuple) -> int:
    '''
    Args:
        dtype: np.dtype - Array data type
        shape: tuple - Array shape
    Returns:
        An integer containing the byte size of an array of data type 'dtype' and shape 'shape'
    '''
    return np.dtype(dtype).itemsize * np.prod(shape, dtype=np.int64)


def encode_int_iterable(integers: ty.Iterable[int], bytes_per_int: int = 4, endianness: str = 'little') -> bytes:
    '''
    Args:
        integers: Iterable[int] - A series of int
        bytes_per_int: int = 4 - Number of bytes per int
        endianness: str = 'little' - Endianness of the byte encoding
    Returns:
        A byte string containing a concatenation of the byte encoding
        of the elements of 'integers'.
    '''
    return b"".join((i.to_bytes(bytes_per_int, endianness) for i in integers))


def decode_int_series(int_series: bytes, bytes_per_int: int = 4, endianness: str = 'little') -> ty.Tuple[int]:
    '''
    Args:
        int_series: bytes - A series of int encoded in bytes
            NOTE: Typically an output of encode_int_iterable
        bytes_per_int: int = 4 - Number of bytes per int
        endianness: str = 'little' - Endianness of the byte encoding
    Returns:
        A tuple of integers containing the decoded series of integers.
    '''
    int_count, should_be_zero = divmod(len(int_series), bytes_per_int)
    assert should_be_zero == 0, "Length of bytes doesn't match the number of bytes per int"

    return tuple((int.from_bytes(int_series[idx * bytes_per_int: (idx + 1) * bytes_per_int], endianness)
                  for idx in range(int_count)))


# int encoding
BYTES_PER_INT = 4  # bytes
ENDIANNESS = "little"

# init message encoding
# TODO: split the message in 2 to allow arbitrary ndim
MAX_NDIM = 8
# dtype, ndim, *shape
INIT_MESSAGE_LEN = BYTES_PER_INT + BYTES_PER_INT + MAX_NDIM * BYTES_PER_INT

DTYPE_TO_INT = {dtype: i for i, dtype in enumerate(
    (bool, np.int8, np.int16, np.int32, np.int64, np.float32, np.float64))}
INT_TO_DTYPE = {v: k for k, v in DTYPE_TO_INT.items()}


def encode_init_message(dtype: np.dtype, shape: tuple) -> str:
    # dtype, ndim, *shape
    int_byte_str = encode_int_iterable(
        (DTYPE_TO_INT[dtype], len(shape), *shape), BYTES_PER_INT, ENDIANNESS)
    assert len(int_byte_str) <= INIT_MESSAGE_LEN

    # complete with zeros
    # TODO: remove by splitting init message in two
    int_byte_str += b" " * (INIT_MESSAGE_LEN - len(int_byte_str))

    return int_byte_str


def decode_init_message(byte_str: str) -> tuple:
    dtype_int, ndim = decode_int_series(
        byte_str[:2 * BYTES_PER_INT], BYTES_PER_INT, ENDIANNESS)

    shape = decode_int_series(
        byte_str[2 * BYTES_PER_INT: (ndim + 2) * BYTES_PER_INT], BYTES_PER_INT, ENDIANNESS)

    return INT_TO_DTYPE[dtype_int], shape


if __name__ == "__main__":
    DTYPE = np.int32
    SHAPE = (10,)

    init_msg = encode_init_message(DTYPE, SHAPE)
    dtype, shape = decode_init_message(init_msg)

    print(dtype, shape)
