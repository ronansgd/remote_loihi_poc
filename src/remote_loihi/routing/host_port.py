import socket
import typing as ty

import numpy as np

LOCAL_HOST: ty.Final[str] = '127.0.0.1'  # The server's hostname or IP address


def is_port_in_use(port: int) -> bool:
    # credits: https://stackoverflow.com/questions/2470971/fast-way-to-test-if-a-port-is-in-use-using-python
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


VALID_PORTS_RANGE: ty.Final[tuple] = (1024, 65535)


def get_unused_port() -> int:
    while True:
        port = np.random.randint(*VALID_PORTS_RANGE)

        if not is_port_in_use(port):
            return port
