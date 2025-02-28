from pydantic import BaseModel


class MaintenancePayload(BaseModel):
    """
    Model for representing maintenance data.

    Attributes:
        temperature (int): The temperature value.
    """

    temperature: int
