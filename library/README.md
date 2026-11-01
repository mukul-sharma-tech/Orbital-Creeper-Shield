# Anudesh Kavach

Python library API for the finalized prompt-safety detector.

## Install from this repository

From the repository root, install the package into the active Python environment:

```powershell
python -m pip install -e .\library
```

The package requires Python 3.10 or later. Its first call downloads the
DeBERTa prompt-injection model and MiniLM sentence-embedding model, so network
access is required initially.

## Usage

```python
import anudesh_kavach

safe = anudesh_kavach.isPromptSafe(
    "Explain integration by parts in Python."
)

print(safe)  # True or False
```

The model is initialized lazily on the first call and reused for later calls.
The function returns the detector's `is_safe` boolean. Invalid non-string
arguments raise `TypeError`; model initialization and inference errors are
propagated rather than converted to a safety verdict.
