import argparse

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
import numpy as np

from remote_loihi import (
    lava as _lava,
    routing
)


if __name__ == '__main__':
    # Should we init the remote server?
    parser = argparse.ArgumentParser()
    parser.add_argument('--remote_init', action="store_true")
    parser.add_argument("--no_remote_init",
                        dest="remote_init", action="store_false")
    parser.set_defaults(remote_init=True)
    send_init_msg = parser.parse_args().remote_init

    SHAPE = (10,)
    DTYPE = np.int32

    NUM_STEPS = 10

    client = _lava.ClientProcess(
        SHAPE, DTYPE, routing.DATA_PORT, send_init_msg=send_init_msg)
    for _ in range(4):
        client.run(condition=RunSteps(NUM_STEPS), run_cfg=Loihi2SimCfg())
    client.stop()
