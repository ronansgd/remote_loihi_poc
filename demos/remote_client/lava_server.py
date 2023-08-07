from lava.magma.core.run_conditions import RunContinuous
from lava.magma.core.run_configs import Loihi2SimCfg
import numpy as np

from remote_loihi import (
    lava as _lava,
    routing
)

if __name__ == "__main__":
    server = _lava.ServerProcess(
        routing.DATA_PORT)
    server.run(condition=RunContinuous(), run_cfg=Loihi2SimCfg())

    # TODO: this line is not reachable
    server.stop()
