import argparse

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.proc.io.extractor import Extractor
from lava.proc.io.injector import Injector
from lava.proc.io.utils import ChannelConfig, SendFull
import numpy as np

from remote_loihi import (
    lava as _lava,
)


if __name__ == '__main__':
    # init client process
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
    in_shape, out_shape = (10,), (5,)
    client = _lava.ClientProcess(
        port, dtype, in_shape, out_shape, send_init_msg=send_init_msg)

    # connect data injector & extractor
    buffer_size = 1
    channel_config = ChannelConfig(send_full=SendFull.BLOCKING)

    injector = Injector(shape=in_shape, buffer_size=buffer_size,
                        channel_config=channel_config)
    injector.out_port.connect(client.in_port)

    extractor = Extractor(shape=out_shape, buffer_size=buffer_size,
                          channel_config=channel_config)
    client.out_port.connect(extractor.in_port)

    # run the network
    n_steps = 10
    for i in range(n_steps):
        # TODO: put meaningful data here
        in_array = np.zeros(in_shape, dtype)
        in_array[i % out_shape[0]] = 1

        print(f"{i}: sent {in_array}")

        # set data in injector & run
        # TODO: de-hardcode tag
        injector.send(in_array)
        injector.run(condition=RunSteps(1),
                     run_cfg=Loihi2SimCfg(select_tag="fixed_pt"))
        out_array = extractor.receive()
        print(f"{i}: received {out_array}")

    for p in (injector, client, extractor):
        p.stop()
