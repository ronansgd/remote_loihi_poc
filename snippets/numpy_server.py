import socket
import numpy as np

import _utils

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)


if __name__ == "__main__":
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()

        while True:
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")

                # recv init message
                init_msg = conn.recv(_utils.INIT_MESSAGE_LEN)
                dtype, shape = _utils.decode_init_message(init_msg)
                array_msg_len = _utils.get_array_bytes_len(dtype, shape)

                print(f"Decoded init message: {dtype} {shape}")

                while True:
                    arr_bytes = conn.recv(array_msg_len)
                    if not arr_bytes:
                        break

                    arr = np.frombuffer(arr_bytes, dtype=dtype).reshape(shape)
                    print(f"Received array {arr}")
                    conn.sendall(arr.tobytes())
