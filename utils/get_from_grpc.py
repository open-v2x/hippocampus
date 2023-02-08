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

import grpc
from npu_grpc import tfdlserver_pb2, tfdlserver_pb2_grpc

MAX_MESSAGE_LENGTH = 100 * 1024 * 1024


def get_data(model, shape, input):
    with grpc.insecure_channel(
        "localhost:50051",
        options=[
            ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
            ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
        ],
    ) as channel:
        stub = tfdlserver_pb2_grpc.TFDLServerStub(channel)
        input = [tfdlserver_pb2.TensorData(shape=shape, data=input)]
        reply = stub.forward(tfdlserver_pb2.ForwardRequest(model=model, input=input))
    return reply
