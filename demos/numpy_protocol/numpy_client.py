import socket

import numpy as np

from remote_loihi import (
    com_protocol,
    routing
)

SHAPE = (10,)
DTYPE = np.int32

if __name__ == "__main__":
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((routing.LOCAL_HOST, routing.PORT))

        # send init message
        init_msg = com_protocol.encode_init_message(DTYPE, SHAPE)
        s.sendall(init_msg)
        print(f"Sent init msg")

        # TODO: use proper function for decoding
        array_msg_len = com_protocol.get_array_bytes_len(DTYPE, SHAPE)
        for i in range(5):
            arr = np.full(SHAPE, i, DTYPE)
            arr_bytes = arr.tobytes()
            s.sendall(arr_bytes)
            print(f"Sent {arr}")

            rec_arr_bytes = s.recv(array_msg_len)
            rec_arr = np.frombuffer(rec_arr_bytes, dtype=DTYPE).reshape(SHAPE)
            print(f"Received {rec_arr}")
