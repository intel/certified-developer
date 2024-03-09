from pydantic import BaseModel

class MaintenancePayload(BaseModel):
    temperature: int
    hydraulic_pressure: int