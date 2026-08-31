import json
import tempfile
import unittest
from datetime import timezone
from pathlib import Path
from uuid import UUID

from blackbird.contracts import (
    BlackbirdResult,
    ProviderBallot,
    ReasoningResponse,
    ReasoningRound,
    TrialRecord,
)
from blackbird.trials import (
    TrialRecorder,
    build_trial_record,
    calculate_self_vote_count,
)


def fake_result() -> BlackbirdResult:
    candidates = {
        "A": ReasoningResponse(
            provider="openai",
            self_confidence=0.8,
            response="Alpha",
        ),
        "B": ReasoningResponse(
            provider="anthropic",
            self_confidence=0.9,
            response="Beta",
        ),
        "C": ReasoningResponse(
            provider="gemini",
            self_confidence=0.7,
            response="Gamma",
        ),
    }
    reasoning_round = ReasoningRound(
        round_number=2,
        responses=list(candidates.values()),
        selected_response=candidates["B"],
        quorum_met=True,
        threshold_met=True,
    )

    return BlackbirdResult(
        rounds=[reasoning_round],
        selected_response=candidates["C"],
        candidates=candidates,
        ballots=[
            ProviderBallot(
                voter="openai",
                candidate_id="A",
                selection_confidence=0.9,
                rationale="Selected Alpha.",
            ),
            ProviderBallot(
                voter="anthropic",
                candidate_id="C",
                selection_confidence=0.8,
                rationale="Selected Gamma.",
            ),
            ProviderBallot(
                voter="gemini",
                candidate_id="C",
                selection_confidence=0.7,
                rationale="Selected Gamma.",
            ),
        ],
        vote_counts={"A": 1, "B": 0, "C": 2},
        winning_candidate_id="C",
        voting_quorum_met=True,
        quorum_met=True,
        threshold_met=False,
    )


def fake_record() -> TrialRecord:
    return build_trial_record(
        original_prompt="Trial prompt",
        provider_models={
            "openai": "openai-model",
            "anthropic": "anthropic-model",
            "gemini": "gemini-model",
        },
        elapsed_seconds=1.25,
        result=fake_result(),
    )


class TrialRecordTests(unittest.TestCase):
    def test_builds_complete_record(self) -> None:
        record = fake_record()

        self.assertIsInstance(record.trial_id, UUID)
        self.assertEqual(record.timestamp.tzinfo, timezone.utc)
        self.assertEqual(record.original_prompt, "Trial prompt")
        self.assertEqual(record.elapsed_seconds, 1.25)
        self.assertEqual(record.result, fake_result())
        self.assertEqual(record.self_vote_count, 2)

    def test_calculates_self_votes_from_candidate_authors(self) -> None:
        self.assertEqual(calculate_self_vote_count(fake_result()), 2)

    def test_serializes_as_valid_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "nested" / "trials.jsonl"
            TrialRecorder(output_path).append(fake_record())

            lines = output_path.read_text(encoding="utf-8").splitlines()
            payload = json.loads(lines[0])

        self.assertEqual(len(lines), 1)
        self.assertEqual(payload["original_prompt"], "Trial prompt")
        self.assertEqual(payload["self_vote_count"], 2)
        self.assertIn("rounds", payload["result"])

    def test_appends_two_independently_parseable_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "trials.jsonl"
            recorder = TrialRecorder(output_path)
            first_record = fake_record()
            second_record = fake_record()

            recorder.append(first_record)
            recorder.append(second_record)

            lines = output_path.read_text(encoding="utf-8").splitlines()
            payloads = [json.loads(line) for line in lines]

        self.assertEqual(len(payloads), 2)
        self.assertEqual(payloads[0]["trial_id"], str(first_record.trial_id))
        self.assertEqual(payloads[1]["trial_id"], str(second_record.trial_id))
        self.assertNotEqual(payloads[0]["trial_id"], payloads[1]["trial_id"])


if __name__ == "__main__":
    unittest.main()
