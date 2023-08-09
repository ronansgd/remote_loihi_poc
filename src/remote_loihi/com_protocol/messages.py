# Copyright (C) 2023 Ronan Sangouard, fortiss - Neuromorphic Computing group
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import typing as ty
import numpy as np

from . import numpy_encoding as npe

# int encoding
BYTES_PER_INT: ty.Final[int] = 4  # bytes
ENDIANNESS: ty.Final[str] = "little"

# dtype is encoded as an int
DTYPE_TO_INT: ty.Final[dict] = {dtype: i for i, dtype in enumerate(
    (bool, np.int8, np.int16, np.int32, np.int64, np.float32, np.float64))}
INT_TO_DTYPE: ty.Final[dict] = {v: k for k, v in DTYPE_TO_INT.items()}

# dtype, ndim
INIT_MSG_LEN: ty.Final[int] = 2 * BYTES_PER_INT


def encode_dtype_ndim(dtype: np.dtype, ndim: int) -> bytes:
    return npe.encode_int_iterable((DTYPE_TO_INT[dtype], ndim), BYTES_PER_INT, ENDIANNESS)


def decode_dtype_ndim(byte_str: bytes) -> ty.Tuple[np.dtype, int]:
    dtype_int, ndim = npe.decode_int_series(
        byte_str, BYTES_PER_INT, ENDIANNESS)

    return INT_TO_DTYPE[dtype_int], ndim


def encode_shape(shape: ty.Tuple[int]) -> bytes:
    return npe.encode_int_iterable(shape, BYTES_PER_INT, ENDIANNESS)


def decode_shape(shape_bytes: bytes) -> ty.Tuple[int]:
    return npe.decode_int_series(shape_bytes, BYTES_PER_INT, ENDIANNESS)


def encode_init_message(dtype: np.dtype, shape: tuple) -> str:
    # dtype, ndim, *shape
    int_byte_str = npe.encode_int_iterable(
        (DTYPE_TO_INT[dtype], len(shape), *shape), BYTES_PER_INT, ENDIANNESS)
    assert len(int_byte_str) <= INIT_MSG_LEN

    # complete with zeros
    # TODO: remove by splitting init message in two
    int_byte_str += b" " * (INIT_MSG_LEN - len(int_byte_str))

    return int_byte_str


def decode_init_message(byte_str: str) -> tuple:
    dtype_int, ndim = npe.decode_int_series(
        byte_str[:2 * BYTES_PER_INT], BYTES_PER_INT, ENDIANNESS)

    shape = npe.decode_int_series(
        byte_str[2 * BYTES_PER_INT: (ndim + 2) * BYTES_PER_INT], BYTES_PER_INT, ENDIANNESS)

    return INT_TO_DTYPE[dtype_int], shape
