

import socket
import numpy as np

from typing import Tuple

from lava.magma.core.process.variable import Var
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort

from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort


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
