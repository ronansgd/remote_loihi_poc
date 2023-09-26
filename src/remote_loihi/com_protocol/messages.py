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


def encode_dtype_ndims(dtype: np.dtype, in_ndim: int, out_ndim: int) -> bytes:
    return npe.encode_int_iterable((DTYPE_TO_INT[dtype], in_ndim, out_ndim), BYTES_PER_INT, ENDIANNESS)


def decode_dtype_ndims(byte_str: bytes) -> ty.Tuple[np.dtype, int]:
    dtype_int, in_ndim, out_ndim = npe.decode_int_series(
        byte_str, BYTES_PER_INT, ENDIANNESS)

    return INT_TO_DTYPE[dtype_int], in_ndim, out_ndim


def encode_shape(shape: ty.Tuple[int]) -> bytes:
    return npe.encode_int_iterable(shape, BYTES_PER_INT, ENDIANNESS)


def decode_shape(shape_bytes: bytes) -> ty.Tuple[int]:
    return npe.decode_int_series(shape_bytes, BYTES_PER_INT, ENDIANNESS)
