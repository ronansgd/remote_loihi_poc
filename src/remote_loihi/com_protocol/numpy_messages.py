# Copyright (C) 2023 Ronan Sangouard, fortiss - Neuromorphic Computing group
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np

from . import numpy_encoding as npe

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
    int_byte_str = npe.encode_int_iterable(
        (DTYPE_TO_INT[dtype], len(shape), *shape), BYTES_PER_INT, ENDIANNESS)
    assert len(int_byte_str) <= INIT_MESSAGE_LEN

    # complete with zeros
    # TODO: remove by splitting init message in two
    int_byte_str += b" " * (INIT_MESSAGE_LEN - len(int_byte_str))

    return int_byte_str


def decode_init_message(byte_str: str) -> tuple:
    dtype_int, ndim = npe.decode_int_iterable(
        byte_str[:2 * BYTES_PER_INT], BYTES_PER_INT, ENDIANNESS)

    shape = npe.decode_int_iterable(
        byte_str[2 * BYTES_PER_INT: (ndim + 2) * BYTES_PER_INT], BYTES_PER_INT, ENDIANNESS)

    return INT_TO_DTYPE[dtype_int], shape
