import socket

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.variable import Var
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
import numpy as np

from remote_loihi import (
    com_protocol,
    routing
)


class ServerProcess(AbstractProcess):
    def __init__(self, port: int) -> None:
        # open management socket and wait for connection
        self.mgmt_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.mgmt_sock.bind((routing.LOCAL_HOST, routing.MGMT_PORT))
        self.mgmt_sock.listen()

        conn, addr = self.mgmt_sock.accept()
        print(f"Management connection from {addr}")

        with conn:
            # read dtype & shape from input connection
            init_msg = conn.recv(com_protocol.INIT_MESSAGE_LEN)
            dtype, shape = com_protocol.decode_init_message(init_msg)
            print(f"Received dtype & shape: {dtype} {shape}")

        self.mgmt_sock.shutdown(socket.SHUT_RDWR)
        self.mgmt_sock.close()

        super().__init__(host=routing.LOCAL_HOST, port=port, dtype=dtype, shape=shape)

        # we use the shape with a first message
        self.data = Var(shape=shape, init=0)
        self.inp = InPort(shape=shape)


@implements(proc=ServerProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyServerProcess(PyLoihiProcessModel):
    # TODO: add ports to communicate with other lava processes
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    data: np.ndarray = LavaPyType(np.ndarray, float)

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)

        # unpack proc params
        self.dtype, self.shape = (
            self.proc_params[k] for k in ('dtype', 'shape'))
        self.array_msg_len = com_protocol.get_array_bytes_len(
            self.dtype, self.shape)

        # init data socket & wait for connection
        self.data_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.data_sock.bind((routing.LOCAL_HOST, routing.DATA_PORT))
        self.data_sock.listen()

        # wait for first client
        self.wait_for_new_client()

        # NOTE: could move the connection logic to run_spk in a later version

    def run_spk(self) -> None:
        arr = self.read_from_client()

        if arr is None:
            print(f"The current client terminated the connection")

            # connect to next client
            self.close_current_client_socket()
            self.wait_for_new_client()

            # re-run run spike with new client
            self.run_spk()
        else:
            print(f"Received array {arr}")

            # send back array
            # TODO: instead, do meaningful computations with another Lava process
            self.send_to_client(arr)

    def send_to_client(self, array: np.ndarray) -> None:
        self.in_conn.sendall(array.tobytes())

    def wait_for_new_client(self) -> None:
        self.in_conn, addr = self.data_sock.accept()
        print(f"Data connection from {addr}")

    def read_from_client(self) -> np.ndarray:
        arr_bytes = self.in_conn.recv(self.array_msg_len)
        if not arr_bytes:

            return None
        else:
            return np.frombuffer(arr_bytes, dtype=self.dtype).reshape(self.shape)

    def close_current_client_socket(self) -> None:
        self.in_conn.shutdown(socket.SHUT_RDWR)
        self.in_conn.close()

    def close_data_socket(self) -> None:
        self.data_sock.shutdown(socket.SHUT_RDWR)
        self.data_sock.close()

    def _req_rs_stop(self) -> None:
        super()._req_rs_stop()

        # close the socket
        # TODO: it seems that it is not called for now
        print("Closing the socket")
        self.close_data_socket()

        if hasattr(self, "in_conn"):
            self.close_current_client_socket()
