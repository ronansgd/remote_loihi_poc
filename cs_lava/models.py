# Copyright (C) 2023 Ronan Sangouard, fortiss, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import logging
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

from cs_lava.socket_process import SocketProcess


@implements(proc=SocketProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PySocket(PyLoihiProcessModel):
    in_port: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    out_port: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self.host = proc_params['host']
        self.port = proc_params['port']
        self.in_shape = proc_params['in_shape']
        self.out_shape = proc_params['out_shape']
        self.in_bytes = np.zeros(self.in_shape, dtype=float).nbytes
        self.out_bytes = np.zeros(self.out_shape, dtype=float).nbytes
        self.is_server = proc_params['is_server']
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn = None
        if self.is_server:
            self.connect_as_server()
        else:
            self.connect_as_client()

    def connect_as_client(self):
        print(f'Connecting client to {self.host}:{self.port}')
        self.sock.connect((self.host, self.port))

    def connect_as_server(self):
        print(f'Server listening on {self.host}:{self.port}')
        self.sock.bind((self.host, self.port))
        self.sock.listen()
        self.conn, addr = self.sock.accept()
        print(f'Server accepted connection from {addr}')

    def join(self):
        try:
            # self.sock.shutdown()
            if self.conn:
                self.conn.close()
            if self.sock:
                self.sock.close()
        except Exception as e:
            print(f'Unable to close socket: {e}')
        super().join()

    def run_spk(self):
        if self.is_server:
            self.socket_to_port(self.conn, self.in_bytes, self.in_shape)
            self.port_to_socket(self.conn)
        else:
            self.port_to_socket(self.sock)
            self.socket_to_port(self.sock, self.out_bytes, self.out_shape)

    def port_to_socket(self, socket):
        data = self.in_port.recv()
        bytes = data.tobytes()
        socket.sendall(bytes)

    def socket_to_port(self, socket, nbytes, shape):
        bytes = socket.recv(nbytes)
        data = np.frombuffer(bytes, dtype=float).reshape(shape)
        self.out_port.send(data)
