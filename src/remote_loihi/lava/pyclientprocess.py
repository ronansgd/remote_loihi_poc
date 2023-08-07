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

from remote_loihi import com_protocol


class ClientProcess(AbstractProcess):
    def __init__(self, shape: tuple, dtype: np.dtype, host: str, port: int) -> None:
        # TODO: factorize proc_params in a single dictionnary
        super().__init__(shape=shape, dtype=dtype, host=host, port=port)
        self.data = Var(shape=shape, init=0)
        self.inp = InPort(shape=shape)

        self.host = host
        self.port = port


@implements(proc=ClientProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyClientProcess(PyLoihiProcessModel):
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    data: np.ndarray = LavaPyType(np.ndarray, float)

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)
        # init & connect socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(
            tuple((self.proc_params[k] for k in ('host', 'port'))))

        # send configuration message
        self.dtype, self.shape = (
            self.proc_params[k] for k in ('dtype', 'shape'))
        self.array_msg_len = com_protocol.get_array_bytes_len(
            self.dtype, self.shape)
        init_msg = com_protocol.encode_init_message(self.dtype, self.shape)
        self.sock.sendall(init_msg)
        print(f"Sent init msg")

    def run_spk(self) -> None:
        # TODO: de-hardcode the number of steps for the termination condition
        if self.sock is None:
            return
        elif self.time_step >= 10:
            self.close_socket()
            self.sock = None
            return
        else:
            # send dummy data
            # TODO: plug with meaningful spike generator
            arr = np.full(self.shape, self.time_step, self.dtype)
            self.send_to_server(arr)
            print(f"Sent {arr}")

            # read returned data
            read_arr = self.read_from_server()
            print(f"Received: {read_arr}")

    def send_to_server(self, array: np.ndarray) -> None:
        arr_bytes = array.tobytes()
        self.sock.sendall(arr_bytes)

    def read_from_server(self) -> np.ndarray:
        arr_bytes = self.sock.recv(self.array_msg_len)

        return np.frombuffer(arr_bytes, dtype=self.dtype).reshape(self.shape)

    def close_socket(self) -> None:
        self.sock.shutdown(socket.SHUT_RDWR)
        self.sock.close()
