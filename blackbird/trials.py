from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from blackbird.contracts.blackbird_result import BlackbirdResult
from blackbird.contracts.trial_record import TrialRecord


def calculate_self_vote_count(result: BlackbirdResult) -> int:
    return sum(
        ballot.voter
        == result.candidates[ballot.candidate_id].provider
        for ballot in result.ballots
        if ballot.candidate_id in result.candidates
    )


def build_trial_record(
    *,
    original_prompt: str,
    provider_models: dict[str, str],
    elapsed_seconds: float,
    result: BlackbirdResult,
) -> TrialRecord:
    return TrialRecord(
        trial_id=uuid4(),
        timestamp=datetime.now(timezone.utc),
        original_prompt=original_prompt,
        provider_models=provider_models,
        elapsed_seconds=elapsed_seconds,
        result=result,
        self_vote_count=calculate_self_vote_count(result),
    )


def build_failed_trial_record(
    *,
    original_prompt: str,
    provider_models: dict[str, str],
    elapsed_seconds: float,
    error: Exception,
) -> TrialRecord:
    return TrialRecord(
        trial_id=uuid4(),
        timestamp=datetime.now(timezone.utc),
        original_prompt=original_prompt,
        provider_models=provider_models,
        elapsed_seconds=elapsed_seconds,
        error_type=type(error).__name__,
        error_message=str(error) or repr(error),
        self_vote_count=0,
    )


class TrialRecorder:
    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)

    def append(self, record: TrialRecord) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with self.output_path.open("a", encoding="utf-8") as output_file:
            output_file.write(record.model_dump_json())
            output_file.write("\n")
