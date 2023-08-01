# https://realpython.com/python-sockets/
# echo-server.py

import socket
import numpy as np

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

# TODO: to be removed
SHAPE = (10,)
DTYPE = np.int32


def get_numpy_bytes_len(dtype: np.dtype, shape: tuple) -> int:
    return np.dtype(dtype).itemsize * np.prod(shape, dtype=np.int64)


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()

    while True:
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                arr_bytes = conn.recv(get_numpy_bytes_len(DTYPE, SHAPE))
                if not arr_bytes:
                    break

                arr = np.frombuffer(arr_bytes, dtype=DTYPE).reshape(SHAPE)
                print(f"Received array {arr}")

                conn.sendall(arr.tobytes())
