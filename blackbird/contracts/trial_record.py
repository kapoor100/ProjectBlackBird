from datetime import datetime, timezone
from typing import Self
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

from blackbird.contracts.blackbird_result import BlackbirdResult


class TrialRecord(BaseModel):
    trial_id: UUID
    timestamp: datetime
    original_prompt: str = Field(min_length=1)
    provider_models: dict[str, str]
    elapsed_seconds: float = Field(ge=0.0)
    result: BlackbirdResult | None = None
    error_type: str | None = Field(default=None, min_length=1)
    error_message: str | None = Field(default=None, min_length=1)
    self_vote_count: int = Field(ge=0)

    @field_validator("timestamp")
    @classmethod
    def normalize_timestamp_to_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include timezone information")

        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        has_error = (
            self.error_type is not None
            and self.error_message is not None
        )
        has_partial_error = (
            (self.error_type is None)
            != (self.error_message is None)
        )

        if has_partial_error or (self.result is None) == (not has_error):
            raise ValueError(
                "trial must contain either a result or an error type/message"
            )

        if self.result is None and self.self_vote_count != 0:
            raise ValueError("failed trials must have self_vote_count=0")

        return self
