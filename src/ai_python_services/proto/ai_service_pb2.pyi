from google.protobuf.descriptor import FileDescriptor
from google.protobuf.message import Message

DESCRIPTOR: FileDescriptor

class HelloRequest(Message):
    name: str
    def __init__(self, *, name: str = ...) -> None: ...

class HelloResponse(Message):
    message: str
    def __init__(self, *, message: str = ...) -> None: ...
