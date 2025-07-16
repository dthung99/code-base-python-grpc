from google.protobuf.message import Message

class HealthRequest(Message):
    def __init__(self) -> None: ...

class NoteGenerationRequestItem(Message):
    id: str
    label: str
    guide: str
    sample: str
    def __init__(self, *, id: str = ..., label: str = ..., guide: str = ..., sample: str = ...) -> None: ...

class NoteGenerationRequest(Message):
    def __init__(self) -> None: ...
