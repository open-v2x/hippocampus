from google.protobuf.internal import containers as _containers  # type:ignore
from google.protobuf import descriptor as _descriptor  # type:ignore
from google.protobuf import message as _message  # type:ignore
from typing import (
    ClassVar as _ClassVar,
    Iterable as _Iterable,
    Mapping as _Mapping,
    Optional as _Optional,
    Union as _Union,
)

DESCRIPTOR: _descriptor.FileDescriptor

class ForwardRequest(_message.Message):
    __slots__ = ["id", "input", "model"]
    ID_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    id: str
    input: _containers.RepeatedCompositeFieldContainer[TensorData]
    model: str
    def __init__(
        self,
        model: _Optional[str] = ...,
        id: _Optional[str] = ...,
        input: _Optional[_Iterable[_Union[TensorData, _Mapping]]] = ...,
    ) -> None: ...

class ForwardResponse(_message.Message):
    __slots__ = ["id", "model", "output"]
    ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    id: str
    model: str
    output: _containers.RepeatedCompositeFieldContainer[TensorData]
    def __init__(
        self,
        model: _Optional[str] = ...,
        id: _Optional[str] = ...,
        output: _Optional[_Iterable[_Union[TensorData, _Mapping]]] = ...,
    ) -> None: ...

class TensorData(_message.Message):
    __slots__ = ["data", "shape"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[float]
    shape: _containers.RepeatedScalarFieldContainer[int]
    def __init__(
        self, shape: _Optional[_Iterable[int]] = ..., data: _Optional[_Iterable[float]] = ...
    ) -> None: ...
