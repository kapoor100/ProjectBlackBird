# Project Blackbird

Project Blackbird sends the same prompt to one or more AI providers concurrently
and normalizes their answers into a shared response contract.

## Architecture

- `blackbird/providers/` contains vendor-specific adapters.
- `blackbird/contracts/` contains provider-neutral data models.
- `blackbird/coordinator.py` owns concurrent orchestration.
- `main.py` is a thin command-line entry point.

Provider adapters depend inward on the shared contract. The coordinator depends
only on the abstract provider interface, so adding another provider does not
require changing orchestration.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add the keys for the providers you intend to
use. Local `.env` files are ignored by version control.

## Usage

Query all providers:

```bash
.venv/bin/python main.py "Explain dependency inversion"
```

Query only one provider:

```bash
.venv/bin/python main.py --provider openai "Explain dependency inversion"
```

The command prints a JSON result containing both reasoning rounds, the selected
response, and whether it met the confidence threshold. Provider/API errors fail
the command rather than silently returning a partial comparison.

## Tests

Tests use fake providers and do not make network requests:

```bash
.venv/bin/python -m unittest discover -s tests
```
## Automated review

Pull requests receive an advisory review from Codex. Human maintainers retain final approval and merge authority.
