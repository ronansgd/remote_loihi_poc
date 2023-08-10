from lava.proc.io.injector import Injector
from lava.proc.io.extractor import Extractor
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.proc.io.utils import ChannelConfig, SendFull, ReceiveEmpty, ReceiveNotEmpty
import numpy as np

if __name__ == "__main__":
    chan_conf = ChannelConfig(
        SendFull.NON_BLOCKING_DROP, ReceiveEmpty.BLOCKING, ReceiveNotEmpty.ACCUMULATE)

    data_shape = (10,)
    buffer_size = 1
    channel_config = ChannelConfig(send_full=SendFull.BLOCKING)
    num_steps = 1

    injector = Injector(shape=data_shape, buffer_size=buffer_size,
                        channel_config=channel_config)

    extractor = Extractor(shape=data_shape, buffer_size=buffer_size,
                          channel_config=channel_config)

    injector.out_port.connect(extractor.in_port)

    dtype = np.int32
    n_steps = 10
    for i in range(n_steps):
        # TODO: put meaningful data here
        # NOTE: only float values are allowed so far
        in_array = np.zeros(data_shape, dtype)
        in_array[i % data_shape[0]] = 1
        print(f"{i}: in array {in_array}")

        # set data in data injector
        injector.send(in_array)
        injector.run(condition=RunSteps(1),
                     run_cfg=Loihi2SimCfg(select_tag="fixed_pt"))

        out_array = extractor.receive()
        print(f"{i}: out array {out_array}")
