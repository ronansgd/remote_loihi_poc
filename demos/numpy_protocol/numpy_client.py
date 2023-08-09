import argparse
import socket

import numpy as np

from remote_loihi import (
    com_protocol,
    routing
)

DTYPE = np.int32
SHAPE = (10,)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=None)

    args = parser.parse_args()
    port = args.port
    assert port is not None and \
        isinstance(port, int), "Please provide a valid port via the --port flag"

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((routing.LOCAL_HOST, port))

        # send init message
        dtype_ndim_bytes = com_protocol.encode_dtype_ndim(
            DTYPE, len(SHAPE))
        s.sendall(dtype_ndim_bytes)

        shape_bytes = com_protocol.encode_shape(SHAPE)
        s.sendall(shape_bytes)
        print(f"Sent dtype & shape")

        array_msg_len = com_protocol.get_array_bytes_len(DTYPE, SHAPE)
        for i in range(5):
            arr = np.full(SHAPE, i, DTYPE)
            arr_bytes = arr.tobytes()
            s.sendall(arr_bytes)
            print(f"Sent {arr}")

            rec_arr_bytes = s.recv(array_msg_len)
            rec_arr = np.frombuffer(rec_arr_bytes, dtype=DTYPE).reshape(SHAPE)
            print(f"Received {rec_arr}")
