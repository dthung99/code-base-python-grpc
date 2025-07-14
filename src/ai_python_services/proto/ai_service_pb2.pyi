from google.protobuf.descriptor import FileDescriptor
from google.protobuf.message import Message

DESCRIPTOR: FileDescriptor

class HealthRequest(Message):
    def __init__(self) -> None: ...

class HealthResponse(Message):
    message: str
    def __init__(self, *, message: str = ...) -> None: ...
