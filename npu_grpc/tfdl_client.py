# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the GRPC helloworld.Greeter client."""

from __future__ import print_function

import logging
import numpy as np
import grpc
import tfdlserver_pb2
import tfdlserver_pb2_grpc


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.

    MAX_MESSAGE_LENGTH = 100 * 1024 * 1024
    print("Will try to greet world ...")
    with grpc.insecure_channel(
        "localhost:50051",
        options=[
            ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
            ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
        ],
    ) as channel:
        result = np.loadtxt("test_input.txt")
        stub = tfdlserver_pb2_grpc.TFDLServerStub(channel)
        input = [tfdlserver_pb2.TensorData(shape=[1, 3, 640, 640], data=result)]
        reply = stub.forward(tfdlserver_pb2.ForwardRequest(model="yolov5s_fp32", input=input))
        print("reply: ", reply.model)
        print("reply data size: : ", len(reply.output[0].data))
        np.savetxt("test_output.txt", reply.output[0].data)


if __name__ == "__main__":
    logging.basicConfig()
    run()
