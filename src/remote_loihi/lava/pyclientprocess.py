import socket

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
import numpy as np

from remote_loihi import (
    com_protocol,
    routing
)


class ClientProcess(AbstractProcess):
    SHAPE_KEYS = ("in_shape", "out_shape")

    def __init__(self, com_port: int, dtype: np.dtype, in_shape: tuple, out_shape: tuple = None, **kwargs) -> None:
        '''
        Kwargs:
            send_init_msg: bool = True
                Flag indicating whether an initialization message containing dtype & shape
                should be sent to the server process.
        '''
        out_shape = in_shape if out_shape is None else out_shape

        super().__init__(port=com_port, dtype=dtype, in_shape=in_shape, out_shape=out_shape)

        self.inp = InPort(shape=in_shape)
        self.outp = OutPort(shape=out_shape)

        if kwargs.get("send_init_msg", True):
            # init & connect management socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as mgmt_conn:
                routing.wait_for_server(
                    mgmt_conn, routing.LOCAL_HOST, com_port)

                # send desired shape & dtype using management socket
                # NOTE: could be extended to send other info dynamically
                # NOTE: we could one input dtype and one output dtype
                dtype_ndim_bytes = com_protocol.encode_dtype_ndims(
                    dtype, len(in_shape), len(out_shape))
                mgmt_conn.sendall(dtype_ndim_bytes)

                for shape in (in_shape, out_shape):
                    shape_bytes = com_protocol.encode_shape(shape)
                    mgmt_conn.sendall(shape_bytes)

                print(f"Sent dtype & shape: {dtype} {in_shape} {out_shape}")

                # wait for server being done
                mgmt_conn.recv(1)


@implements(proc=ClientProcess, protocol=LoihiProtocol)
@requires(CPU)
@tag('fixed_pt')
class PyClientProcess(PyLoihiProcessModel):
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32)
    outp: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32)

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)

        # unpack proc params
        port, self.dtype, self.in_shape, self.out_shape = (
            self.proc_params[k] for k in ("port", "dtype", "in_shape", "out_shape"))
        self.array_msg_len = com_protocol.get_array_bytes_len(
            self.dtype, self.in_shape)

        # init & connect data socket
        self.data_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        routing.wait_for_server(self.data_conn, routing.LOCAL_HOST, port)

    def run_spk(self) -> None:
        # send dummy data
        # TODO: plug with meaningful spike generator
        arr = np.zeros(self.in_shape, self.dtype)
        arr[:self.time_step % self.in_shape[0]] = 1

        self.data_conn.sendall(arr.tobytes())
        print(f"Sent {arr}")

        # read returned data
        arr_bytes = self.data_conn.recv(self.array_msg_len)
        read_arr = np.frombuffer(
            arr_bytes, dtype=self.dtype).reshape(self.out_shape)
        print(f"Received: {read_arr}")

    # def _req_rs_stop(self) -> None:
    #     # NOTE: it seems that this callback is not called on .stop()

    #     super()._req_rs_stop()

    #     print("Closing the socket")
    #     self.data_conn.close()
