from pydantic import BaseModel

class MaintenancePayload(BaseModel):
    temperature: int