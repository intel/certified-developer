from pydantic import BaseModel


class GenPayload(BaseModel):
    """
    Data model for generation payload.

    Attributes:
        data (str): The data to be used for generation.
        user_input (str): The user input for the generation process.
    """

    data: str
    user_input: str
