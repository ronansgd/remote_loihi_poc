# Copyright (C) 2023 Ronan Sangouard, fortiss, Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import argparse
import matplotlib.pyplot as plt

from scipy.sparse import csr_array

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg, Loihi2HwCfg
from lava.magma.core.process.process import LogConfig
from lava.proc.lif.process import LIF
from lava.proc.sparse.process import DelaySparse
from lava.proc.io.injector import Injector
from lava.proc.io.extractor import Extractor

from cs_lava.socket_process import SocketProcess


def connect_sparse_random(left, right, prob_connect, min_w, max_w, min_d, max_d):
    conn_shape = (np.prod(left.s_out.shape), np.prod(right.a_in.shape))
    mask = np.random.rand(*conn_shape) < (1 - prob_connect)
    weights = np.random.randint(min_w, max_w, size=conn_shape, dtype=int)
    weights[mask] = 0
    weights = csr_array(weights)
    delays = np.random.randint(min_d, max_d, size=conn_shape, dtype=int)
    delays[mask] = 0
    delays = csr_array(delays)
    conn = DelaySparse(weights=weights, delays=delays)
    left.s_out.connect(conn.s_in)
    conn.a_out.connect(right.a_in)
    return conn

def connect_sparse_diagonal_inputs(neurons, w_spk):
    weights = w_spk * np.eye(np.prod(neurons.a_in.shape), dtype=int)
    weights = csr_array(weights)
    delays = np.eye(np.prod(neurons.a_in.shape), dtype=int)
    delays = csr_array(delays)
    conn = DelaySparse(weights=weights, delays=delays)
    return conn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='demo_socket',
        description='Run a demo of SocketProcess')
    parser.add_argument('-c', '--client', action='store_true', help='Run as a client socket process.')
    parser.add_argument('-s', '--server', action='store_true', help='Run as a server socket process.')
    parser.add_argument('--use_loihi_2', action='store_true', help='Use a Loihi 2 Hw Config.')
    parser.add_argument('host', default='127.0.0.1', help=('For a client, the host on which ' +
                        'the server socket process is running. For a server, the host address ' +
                        'which the server socket process should listen for clients.'))
    parser.add_argument('port', default=12345, type=int, help='The port to connect.')
    args = parser.parse_args()

    shape = (10,)
    num_steps = 10
    p_spike = 0.25

    socket = SocketProcess(in_shape=shape, out_shape=shape, host=args.host,
                           port=args.port, is_server=args.server)

    if args.server:
        lif_exc = LIF(shape=shape, vth=10, du=4, dv=4)
        lif_inh = LIF(shape=shape, vth=10, du=4, dv=4)
        in_exc = connect_sparse_diagonal_inputs(lif_exc, 128)
        exc_exc = connect_sparse_random(lif_exc, lif_exc, 0.1, 4, 32, 2, 8)
        exc_inh = connect_sparse_random(lif_exc, lif_inh, 0.1, 4, 32, 2, 8)
        inh_exc = connect_sparse_random(lif_inh, lif_exc, 0.1, -32, -4, 2, 8)
        if args.use_loihi_2:
            from lava.proc.embedded_io.spike import PyToNxAdapter, NxToPyAdapter
            in_adapter = PyToNxAdapter(shape=shape)
            out_adapter = NxToPyAdapter(shape=shape)
            socket.out_port.connect(in_adapter.inp)
            in_adapter.out.connect(in_exc.s_in)
            lif_exc.s_out.connect(out_adapter.inp)
            out_adapter.out.connect(socket.in_port)
            run_cfg = Loihi2HwCfg()
        else:
            socket.out_port.connect(in_exc.s_in)
            lif_exc.s_out.connect(socket.in_port)
            run_cfg = Loihi2SimCfg()
        u_extractor = Extractor(shape=shape)
        lif_exc.v
        socket.run(condition=RunSteps(num_steps), run_cfg=run_cfg)
    else:
        spikes = np.array([[0, 0]])
        plt.figure(figsize=(4, 4), dpi=120)
        plt.plot(spikes, '.')
        plt.ion()
        plt.show()
        injector = Injector(shape=shape)
        extractor = Extractor(shape=shape)
        injector.out_port.connect(socket.in_port)
        socket.out_port.connect(extractor.in_port)
        socket.run(condition=RunSteps(num_steps, blocking=False), run_cfg=Loihi2SimCfg())
        for i in range(num_steps):
            s_in = (np.random.rand(*shape) < p_spike).astype(int)
            injector.send(s_in)
            s_out = extractor.receive()
            print(s_in)
            print(s_out)
        socket.wait()
        plt.show(block=True)
    socket.stop()
