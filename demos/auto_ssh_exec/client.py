import argparse
import subprocess

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.proc.io.extractor import Extractor
from lava.proc.io.injector import Injector
from lava.proc.io.utils import ChannelConfig, SendFull
import numpy as np

from remote_loihi import (
    lava as _lava,
    routing
)


if __name__ == '__main__':
    # this version is not working for now
    raise NotImplementedError

    # init client process
    parser = argparse.ArgumentParser()
    # should we send shape & dtype to the remote server?
    parser.add_argument("--user", type=str, default=None)
    parser.add_argument("--vm", type=str, default="ncl-com")

    args = parser.parse_args()
    assert args.user is not None, "Please provide a user name via the --user flag"
    print(f"Attempting connection to {args.vm} with username {args.user}")

    # generate random port
    # TODO: check that this port is actually free
    PORT_RANGE = (1024, 65535)
    port = np.random.randint(*PORT_RANGE)

    ssh_process = subprocess.Popen(["ssh", "-tt", "-v",
                                    "-L", f"{port}:{routing.LOCAL_HOST}:{port}",
                                   f"{args.user}@{args.vm}.research.intel-research.net"],
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   universal_newlines=True,
                                   bufsize=0)

    # NOTE: here write the command sequence to be executed remotely
    server_start_cmds = (
        f"export LOIHI_PORT={port}\n",
        "cd repos/remote_loihi_poc/\n",
        "git checkout ronan/single_port_solution\n",
        "git pull\n",
        "source env/bin/activate\n",
        f"python demos/basic_com/server.py --port {port}\n"
    )
    for cmd in server_start_cmds:
        ssh_process.stdin.write(cmd)

    for line in ssh_process.stdout:
        print(line, end="")
        if line.endswith(f"--port {port}\n"):
            break

    dtype = np.int32
    in_shape, out_shape = (10,), (5,)
    client = _lava.ClientProcess(port, dtype, in_shape, out_shape)

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
