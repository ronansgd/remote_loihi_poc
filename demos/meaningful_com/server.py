import argparse

from lava.magma.core.run_conditions import RunContinuous
from lava.magma.core.run_configs import Loihi2SimCfg
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

    # TODO: create dense process...
    dense_weights = np.zeros(())

    server.run(condition=RunContinuous(), run_cfg=Loihi2SimCfg())
