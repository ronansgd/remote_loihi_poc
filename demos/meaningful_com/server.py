import argparse

from lava.magma.core.run_conditions import RunContinuous
from lava.magma.core.run_configs import Loihi2SimCfg
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

    server = _lava.ServerProcess(port)

    dense_shape = (int(np.prod(server.inp.shape)),
                   int(np.prod(server.outp.shape)))

    # TODO: create dense process...
    dense_weights = np.zeros(dense_shape, dtype=server.dtype)
    min_size = min(*dense_shape)
    dense_weights[:min_size, :min_size] = 1

    dense_proc = Dense(weights=dense_weights)

    # connect server process and dense
    server.outp.reshape(new_shape=dense_shape[1:]).connect(dense_proc.s_in)
    dense_proc.a_out.reshape(new_shape=server.inp.shape).connect(server.inp)

    server.run(condition=RunContinuous(), run_cfg=Loihi2SimCfg())
