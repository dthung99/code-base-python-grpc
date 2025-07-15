from google.protobuf.descriptor import FileDescriptor
from google.protobuf.message import Message

DESCRIPTOR: FileDescriptor

class HealthResponse(Message):
    message: str
    def __init__(self, *, message: str = ...) -> None: ...

class NoteGenerationResponseItem(Message):
    id: str
    label: str
    value: str
    def __init__(self, *, id: str = ..., label: str = ..., value: str = ...) -> None: ...

class NoteGenerationResponse(Message):
    NoteGenerationResponseItem: repeated
    def __init__(self, *, NoteGenerationResponseItem: repeated = ...) -> None: ...
