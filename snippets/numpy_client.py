# https://realpython.com/python-sockets/
# echo-client.py

import socket

import numpy as np

HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 65432  # The port used by the server

SHAPE = (10,)
DTYPE = np.int32

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))

    arr = np.zeros(SHAPE, DTYPE)
    arr_bytes = arr.tobytes()
    s.sendall(arr_bytes)
    print(f"Sent {arr}")

    rec_arr_bytes = s.recv(len(arr_bytes))
    rec_arr = np.frombuffer(rec_arr_bytes, dtype=DTYPE).reshape(SHAPE)


print(f"Received {rec_arr}")
