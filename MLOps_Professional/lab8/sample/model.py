from pydantic import BaseModel

class GenPayload(BaseModel):
    data: str
    user_input: str