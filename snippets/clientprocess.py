# https://realpython.com/python-sockets/
# echo-client.py

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


class ClientProcess(AbstractProcess):
    def __init__(self, shape: Tuple, host='127.0.0.1', port=12345) -> None:
        super().__init__(shape=shape, host=host, port=port)
        self.data = Var(shape=shape, init=0)
        self.inp = InPort(shape=shape)

        self.host = host
        self.port = port


if __name__ == '__main__':
    from lava.magma.core.run_conditions import RunSteps
    from lava.magma.core.run_configs import Loihi2SimCfg
    
    client = ClientProcess(shape=(10,))
    client.run(condition=RunSteps(10), run_cfg=Loihi2SimCfg())
    client.stop()
