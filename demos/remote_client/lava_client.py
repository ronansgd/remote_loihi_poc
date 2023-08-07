from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
import numpy as np

from remote_loihi import (
    lava as _lava
)


if __name__ == '__main__':
    SHAPE = (10,)
    DTYPE = np.int32

    # TODO: de-hardcode the number of steps
    NUM_STEPS = 10

    client = _lava.ClientProcess(shape=SHAPE)
    # client.run(condition=RunSteps(10), run_cfg=Loihi2SimCfg())
    # client.stop()
