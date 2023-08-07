import socket
import typing as ty

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

# TODO: define in another package?
LOCAL_HOST: ty.Final[str] = '127.0.0.1'
# TODO: get the port number dynamically
PORT: ty.Final[int] = 12345


class ClientProcess(AbstractProcess):
    def __init__(self, shape: tuple, host: str = LOCAL_HOST, port: int = PORT) -> None:
        super().__init__(shape=shape, host=host, port=port)
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
        super().__init__()
        self.port = self.proc.proc_params['host']
        self.host = self.proc.proc_params['port']
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))

    def run_spk(self) -> None:
        if self.t >= 10:
            self.sock.shutdown()
            self.sock.close()
        self.send_to_server()

    def send_to_server(self, sock):
        bytes = self.data.tobytes()
        sock.sendall(bytes)
        reply = sock.recv(1024)
        print(f"Received {reply!r}")
