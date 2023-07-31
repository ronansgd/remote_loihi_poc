import numpy as np

if __name__ == "__main__":

    arr = np.zeros(10)
    print(arr)
    arr_bytes = arr.tobytes()
    rec_arr = np.frombuffer(arr_bytes, dtype=arr.dtype).reshape(arr.shape)
    print(rec_arr)
