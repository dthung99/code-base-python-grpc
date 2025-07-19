from google.protobuf.message import Message

class HealthResponse(Message):
    message: str
    def __init__(self, *, message: str = ...) -> None: ...
