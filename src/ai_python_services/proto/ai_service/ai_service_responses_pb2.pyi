from google.protobuf.message import Message

class HealthResponse(Message):
    message: str
    def __init__(self, *, message: str = ...) -> None: ...

class NoteGenerationResponseItem(Message):
    id: str
    label: str
    value: str
    def __init__(self, *, id: str = ..., label: str = ..., value: str = ...) -> None: ...

class NoteGenerationResponse(Message):
    def __init__(self) -> None: ...
