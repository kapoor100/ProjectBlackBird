from pydantic import BaseModel


class ProviderFailure(BaseModel):
    provider: str
    error_type: str
    message: str