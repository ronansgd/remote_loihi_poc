import argparse

from lava.magma.core.run_conditions import RunContinuous
from lava.magma.core.run_configs import Loihi2SimCfg

from remote_loihi import (
    lava as _lava,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dport', type=int, default=None)
    parser.add_argument('--mport', type=int, default=None)

    args = parser.parse_args()
    data_port = args.dport
    assert data_port is not None and \
        isinstance(
            data_port, int), "Please provide a valid data port via the --dport flag"
    mgmt_port = args.mport
    assert mgmt_port is not None and \
        isinstance(
            mgmt_port, int), "Please provide a valid management port via the --mport flag"

    server = _lava.ServerProcess(mgmt_port, data_port)
    server.run(condition=RunContinuous(), run_cfg=Loihi2SimCfg())

    # server.stop()
