import argparse

from lava.magma.core.run_conditions import RunContinuous
from lava.magma.core.run_configs import Loihi2HwCfg, Loihi2SimCfg
from lava.proc.dense.process import Dense
import numpy as np

from remote_loihi import (
    lava as _lava,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=None)

    args = parser.parse_args()
    port = args.port
    assert port is not None and \
        isinstance(port, int), "Please provide a valid port via the --port flag"

    if args.sim is None:
        print("No --sim flag provided; running on simulated hardware by default")
        args.sim = 1

    real_loihi = args.sim <= 0
    if real_loihi:
        print("Running on real Loihi 2...")
        run_cfg = Loihi2HwCfg()
    else:
        # run on simulated hardware
        # TODO: add possibility to pick floating point
        print("Running on simulated Loihi 2...")
        run_cfg = Loihi2SimCfg(select_tag="fixed_pt")

    server = _lava.ServerProcess(port)
    dense_shape = tuple((int(np.prod(s))
                        for s in (server.inp.shape, server.out.shape)))
    input_procs = [server]

    if real_loihi:
        # wrap server with nx2py & py2nx adapters
        from lava.proc.embedded_io.spike import PyToNxAdapter, NxToPyAdapter
        # TODO: align port names to allow factorization
        py2nx = PyToNxAdapter(shape=server.out.shape)
        server.out.connect(py2nx.inp)
        input_procs.append(py2nx)

        nx2py = NxToPyAdapter(shape=server.inp.shape)
        nx2py.out.connect(server.inp)
        input_procs.insert(0, nx2py)

    # create dense connection to close the loop
    # NOTE: the dense process will cause a latency of one step
    dense_weights = np.zeros(dense_shape, dtype=server.dtype)
    min_size = min(*dense_shape)
    dense_weights[np.arange(min_size), np.arange(min_size)] = 2
    dense_proc = Dense(weights=dense_weights)

    input_procs[-1].out.reshape(new_shape=dense_shape[1:]
                                ).connect(dense_proc.s_in)
    dense_proc.a_out.reshape(
        new_shape=server.inp.shape).connect(input_procs[0].inp)

    server.run(condition=RunContinuous(),
               run_cfg=run_cfg)
