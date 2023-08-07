from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
import numpy as np

from remote_loihi import (
    lava as _lava,
    routing
)


if __name__ == '__main__':
    SHAPE = (10,)
    DTYPE = np.int32

    # TODO: de-hardcode the number of steps
    NUM_STEPS = 20

    client = _lava.ClientProcess(
        SHAPE, DTYPE, routing.LOCAL_HOST, routing.PORT)
    client.run(condition=RunSteps(NUM_STEPS), run_cfg=Loihi2SimCfg())
    client.stop()
