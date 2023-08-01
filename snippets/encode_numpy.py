import numpy as np


if __name__ == "__main__":
    # init array
    shape, dtype = (10, ), np.int64
    arr = np.zeros(shape, dtype)
    print(arr)

    # encode to byte string & check size
    arr_bytes = arr.tobytes()
    theo_byte_len = np.dtype(dtype).itemsize * np.prod(shape, dtype=np.int64)
    assert len(arr_bytes) == theo_byte_len

    # reconstruct array from byte string
    rec_arr = np.frombuffer(arr_bytes, dtype=dtype).reshape(shape)
    print(rec_arr)
