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


class ClientProcess(AbstractProcess):
    def __init__(self, shape: tuple, dtype: np.dtype, port: int, **kwargs) -> None:
        '''
        Kwargs:
            send_init_msg: bool = True
                Flag indicating whether an initialization message should be sent to the server process
        NOTE: the management & data socket currently work on the same port. It will be necessary
            to change that it they were to run concurrently.
        '''
        # TODO: factorize proc_params in a single dictionnary
        super().__init__(shape=shape, dtype=dtype, host=routing.LOCAL_HOST, port=port)
        self.data = Var(shape=shape, init=0)
        self.inp = InPort(shape=shape)

        if kwargs.get("send_init_msg", True):
            # init & connect management socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as mgmt_sock:
                routing.wait_for_server(mgmt_sock, routing.LOCAL_HOST, port)

                # send desired shape & dtype using management socket
                # NOTE: could later be extended to send other info dynamically
                init_msg = com_protocol.encode_init_message(dtype, shape)
                mgmt_sock.sendall(init_msg)
                print(f"Sent dtype & shape: {dtype} {shape}")


@implements(proc=ClientProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyClientProcess(PyLoihiProcessModel):
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    data: np.ndarray = LavaPyType(np.ndarray, float)

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)

        # unpack proc params
        self.dtype, self.shape = (
            self.proc_params[k] for k in ('dtype', 'shape'))
        self.array_msg_len = com_protocol.get_array_bytes_len(
            self.dtype, self.shape)

        # init & connect data socket
        self.data_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host_port = tuple((self.proc_params[k] for k in ('host', 'port')))
        routing.wait_for_server(self.data_sock, *host_port)

    def run_spk(self) -> None:
        # send dummy data
        # TODO: plug with meaningful spike generator
        arr = np.full(self.shape, self.time_step, self.dtype)
        self.send_to_server(arr)
        print(f"Sent {arr}")

        # read returned data
        read_arr = self.read_from_server()
        print(f"Received: {read_arr}")

    def send_to_server(self, array: np.ndarray) -> None:
        self.data_sock.sendall(array.tobytes())

    def read_from_server(self) -> np.ndarray:
        arr_bytes = self.data_sock.recv(self.array_msg_len)

        return np.frombuffer(arr_bytes, dtype=self.dtype).reshape(self.shape)

    def close_socket(self) -> None:
        self.data_sock.shutdown(socket.SHUT_RDWR)
        self.data_sock.close()

    def _req_rs_stop(self) -> None:
        super()._req_rs_stop()

        # close the socket
        # TODO: it seems that it is not called for now
        print("Closing the socket")
        self.close_socket()
