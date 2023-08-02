import socket

import numpy as np

import _utils

HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 65432  # The port used by the server

SHAPE = (10,)
DTYPE = np.int32

if __name__ == "__main__":
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))

        # send init message
        init_msg = _utils.encode_init_message(DTYPE, SHAPE)
        s.sendall(init_msg)
        print(f"Sent init msg")

        array_msg_len = _utils.get_array_bytes_len(DTYPE, SHAPE)
        for i in range(5):
            arr = np.full(SHAPE, i, DTYPE)
            arr_bytes = arr.tobytes()
            s.sendall(arr_bytes)
            print(f"Sent {arr}")

            rec_arr_bytes = s.recv(array_msg_len)
            rec_arr = np.frombuffer(rec_arr_bytes, dtype=DTYPE).reshape(SHAPE)
            print(f"Received {rec_arr}")
