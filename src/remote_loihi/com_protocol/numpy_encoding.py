# Copyright (C) 2023 Ronan Sangouard, fortiss - Neuromorphic Computing group
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

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


def encode_int_iterable(integers: ty.Union[ty.Iterable[int], int], bytes_per_int: int = 4, endianness: str = 'little') -> bytes:
    '''
    Args:
        integers: Iterable[int] - A series of int
        bytes_per_int: int = 4 - Number of bytes per int
        endianness: str = 'little' - Endianness of the byte encoding
    Returns:
        A byte string containing a concatenation of the byte encoding
        of the elements of 'integers'.
    '''
    if isinstance(integers, int):
        integers = integers,
    return b"".join((i.to_bytes(bytes_per_int, endianness) for i in integers))


def decode_int_series(int_series: bytes, bytes_per_int: int = 4, endianness: str = 'little') -> ty.Union[ty.Tuple[int], int]:
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

    if int_count == 1:
        return int.from_bytes(int_series, endianness)
    else:
        return tuple((int.from_bytes(int_series[idx * bytes_per_int: (idx + 1) * bytes_per_int], endianness)
                      for idx in range(int_count)))
