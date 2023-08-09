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
    def __init__(self, local_port: int) -> None:
        '''
        Args:
            local_port: int - Id of the local port to which the server socket will be bound
        '''
        # the server socket is in charge of accepting new connections
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.bind((routing.LOCAL_HOST, local_port))
        server_sock.listen()

        # the first connection should be a management connection
        mgmt_conn, mgmt_addr = server_sock.accept()
        print(f"Management connection from {mgmt_addr}")

        # read dtype & shape from management connection
        init_msg = mgmt_conn.recv(com_protocol.INIT_MESSAGE_LEN)
        dtype, shape = com_protocol.decode_init_message(init_msg)
        print(f"Received dtype & shape: {dtype} {shape}")

        # NOTE: one could sustain the connection for dynamical reconfig at runtime
        mgmt_conn.close()

        # use received config to set the process parameters
        super().__init__(dtype=dtype, shape=shape, server_sock=server_sock)
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

        # unpack process params
        self.dtype, self.shape, self.server_sock = (self.proc_params[k]
                                                    for k in ("dtype", "shape", "server_sock"))
        self.array_msg_len = com_protocol.get_array_bytes_len(
            self.dtype, self.shape)

        self.wait_for_data_conn()

    def run_spk(self) -> None:
        arr_bytes = self.data_conn.recv(self.array_msg_len)

        if not arr_bytes:
            print(f"The current client terminated the connection")
            self.data_conn.close()

            self.wait_for_data_conn()

            # re-run run spike with new data connection
            self.run_spk()
        else:
            arr = np.frombuffer(
                arr_bytes, dtype=self.dtype).reshape(self.shape)
            print(f"Received array {arr}")

            # send back array
            # TODO: instead, do meaningful computations with another Lava process
            self.data_conn.sendall(arr.tobytes())

    def wait_for_data_conn(self) -> None:
        self.data_conn, addr = self.server_sock.accept()
        print(f"New data connection from {addr}")

    def _req_rs_stop(self) -> None:
        # NOTE: it seems that this callback is not called on .stop()

        super()._req_rs_stop()

        print("Closing open sockets")
        self.server_sock.close()
        if hasattr(self, "data_conn"):
            self.data_conn.close()
