import argparse

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
import numpy as np

from remote_loihi import (
    lava as _lava,
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Should we init the remote server?
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

    SHAPE = (10,)
    DTYPE = np.int32

    NUM_STEPS = 10

    client = _lava.ClientProcess(
        SHAPE, DTYPE, port, send_init_msg=send_init_msg)
    for _ in range(4):
        client.run(condition=RunSteps(NUM_STEPS), run_cfg=Loihi2SimCfg())
    client.stop()
