from datetime import datetime, timezone
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from blackbird.contracts.blackbird_result import BlackbirdResult


class TrialRecord(BaseModel):
    trial_id: UUID
    timestamp: datetime
    original_prompt: str = Field(min_length=1)
    provider_models: dict[str, str]
    elapsed_seconds: float = Field(ge=0.0)
    result: BlackbirdResult
    self_vote_count: int = Field(ge=0)

    @field_validator("timestamp")
    @classmethod
    def normalize_timestamp_to_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include timezone information")

        return value.astimezone(timezone.utc)
