import argparse
import socket

import numpy as np

from remote_loihi import (
    com_protocol,
    routing
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=None)

    args = parser.parse_args()
    port = args.port
    assert port is not None and \
        isinstance(port, int), "Please provide a valid port via the --port flag"

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((routing.LOCAL_HOST, port))
        s.listen()

        while True:
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")

                # recv init message
                dtype_ndim_bytes = conn.recv(3 * com_protocol.BYTES_PER_INT)
                dtype, ndim, _ = com_protocol.decode_dtype_ndims(
                    dtype_ndim_bytes)

                shape_bytes = conn.recv(ndim * com_protocol.BYTES_PER_INT)
                shape = com_protocol.decode_shape(shape_bytes)

                array_msg_len = com_protocol.get_array_bytes_len(dtype, shape)

                print(f"Decoded init message: {dtype} {shape}")

                while True:
                    arr_bytes = conn.recv(array_msg_len)
                    if not arr_bytes:
                        break

                    arr = np.frombuffer(arr_bytes, dtype=dtype).reshape(shape)
                    print(f"Received array {arr}")
                    conn.sendall(arr.tobytes())
