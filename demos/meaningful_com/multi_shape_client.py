import argparse

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
import numpy as np

from remote_loihi import (
    lava as _lava,
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # should we send shape & dtype to the remote server?
    parser.add_argument('--remote-init', dest="remote_init",
                        action="store_true")
    parser.add_argument("--no-remote-init", dest="remote_init",
                        action="store_false")
    parser.set_defaults(remote_init=True)
    # which port should we try to connect to?
    parser.add_argument('--port', type=int, default=None)

    args = parser.parse_args()
    send_init_msg = args.remote_init
    port = args.port
    assert port is not None and \
        isinstance(port, int), "Please provide a valid port via the --port flag"

    dtype = np.int32
    in_shape = (10,)
    client = _lava.ClientProcess(
        port, dtype, in_shape, send_init_msg=send_init_msg)

    n_runs, n_steps = 4, 10
    for _ in range(n_runs):
        client.run(condition=RunSteps(n_steps), run_cfg=Loihi2SimCfg())
    client.stop()
