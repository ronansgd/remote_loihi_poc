import socket
import numpy as np

from remote_loihi import (
    com_protocol,
    routing
)

if __name__ == "__main__":
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((routing.LOCAL_HOST, routing.PORT))
        s.listen()

        while True:
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")

                # recv init message
                init_msg = conn.recv(com_protocol.INIT_MESSAGE_LEN)
                dtype, shape = com_protocol.decode_init_message(init_msg)
                array_msg_len = com_protocol.get_array_bytes_len(dtype, shape)

                print(f"Decoded init message: {dtype} {shape}")

                while True:
                    arr_bytes = conn.recv(array_msg_len)
                    if not arr_bytes:
                        break

                    arr = np.frombuffer(arr_bytes, dtype=dtype).reshape(shape)
                    print(f"Received array {arr}")
                    conn.sendall(arr.tobytes())
