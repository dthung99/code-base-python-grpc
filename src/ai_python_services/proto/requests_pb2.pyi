from google.protobuf.descriptor import FileDescriptor
from google.protobuf.message import Message

DESCRIPTOR: FileDescriptor

class HealthRequest(Message):
    def __init__(self) -> None: ...

class NoteGenerationRequestItem(Message):
    id: str
    label: str
    guide: str
    sample: str
    def __init__(self, *, id: str = ..., label: str = ..., guide: str = ..., sample: str = ...) -> None: ...

class NoteGenerationRequest(Message):
    NoteGenerationRequestItem: repeated
    def __init__(self, *, NoteGenerationRequestItem: repeated = ...) -> None: ...
