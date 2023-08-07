# Copyright (C) 2023 Ronan Sangouard, fortiss, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import logging
import socket
import numpy as np

from typing import Tuple, Optional

from lava.magma.core.process.variable import Var
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort


class SocketProcess(AbstractProcess):
    def __init__(self, in_shape: Tuple[int], out_shape: Tuple[int],
                 host='127.0.0.1', port=12345, is_server=False) -> None:
        super().__init__(in_shape=in_shape, out_shape=out_shape,
                         host=host, port=port, is_server=is_server)
        self.in_port = InPort(shape=in_shape)
        self.out_port = OutPort(shape=out_shape)
        self.host = host
        self.port = port
