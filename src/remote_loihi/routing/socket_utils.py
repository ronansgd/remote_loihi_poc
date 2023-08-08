import socket
import time

import numpy as np


def wait_for_server(client_sock: socket.socket, host: str, port: int, wait_dt: float = 1.) -> None:
    '''
    Args:
        client_sock: socket - Client socket
        host: str - Host name
        port: int - Port
        wait_dt: float - Time between each connection try (in seconds)
    Side-effect:
        Tries to connect the client socket to the remote host every 'wait_dt' seconds.
        Returns when the connection is established.
    '''
    while True:
        try:
            client_sock.connect((host, port))
            break
        except ConnectionRefusedError:
            print("Spinning one more time waiting for the server...")
            time.sleep(wait_dt)
            continue
