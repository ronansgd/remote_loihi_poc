from lava.proc.io.injector import Injector
from lava.proc.io.extractor import Extractor
from lava.proc.io.utils import ChannelConfig, SendFull, ReceiveEmpty, ReceiveNotEmpty

if __name__ == "__main__":
    chan_conf = ChannelConfig(
        SendFull.NON_BLOCKING_DROP, ReceiveEmpty.BLOCKING, ReceiveNotEmpty.ACCUMULATE)
