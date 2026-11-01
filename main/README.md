# Final Prompt Detector

This folder contains the project's standalone finalized prompt detector and
command-line test tools. The detector combines a pretrained prompt-injection
classifier with semantic similarity checks against benign and threat profiles.

> **Important:** This detector is a research prototype, not a guarantee of
> safety. A `PASSED` result should not be treated as authorization to execute a
> command or as a replacement for independent security controls.

## `main.py` architecture

The detector is implemented by `ExplainableSpaceGuardrail` in `main.py`. It
has two detection layers and produces a structured analysis rather than
executing or changing the submitted prompt.

```text
Prompt
  │
  ├─ Layer 1: DeBERTa prompt-injection classifier
  │    └─ Structural injection label and classifier score
  │
  ├─ Layer 2: MiniLM semantic similarity
  │    ├─ Compare with benign-intent anchor phrases
  │    └─ Compare with threat-profile phrases by category
  │
  ├─ Decision policy
  │    ├─ Allow a strong benign match to override a structural flag
  │    ├─ Otherwise block structural injection
  │    ├─ Block a sufficiently strong threat match over benign intent
  │    └─ Pass when none of the blocking conditions apply
  │
  └─ If blocked: word-removal attribution estimates
       └─ Return decision, reason, similarity scores, and attributions
```

### Initialization

Creating `ExplainableSpaceGuardrail()` loads two pretrained models:

1. `protectai/deberta-v3-base-prompt-injection-v2` is loaded through
   Transformers and wrapped in a text-classification pipeline. The model is
   converted to `float16`.
2. `sentence-transformers/all-MiniLM-L6-v2` is loaded through Sentence
   Transformers on CPU.

The class contains five threat categories: trajectory/orbit sabotage,
credential/privilege theft, command uplink spoofing, telemetry/communications
disruption, and satellite/cyber exploits. Each has two text anchors. It also
defines five benign-intent anchors for educational, programming, technical,
and defensive-security prompts.

During initialization, MiniLM encodes the threat and benign anchors once and
keeps their embeddings for subsequent calls. The first initialization may
download model files and therefore requires network access.

### Per-prompt evaluation

Call `analyze_prompt(prompt: str)` to evaluate one string:

1. **Structural check:** DeBERTa classifies the prompt. The result is treated
   as a structural injection when its label contains `INJECTION` or
   `LABEL_1` (case-insensitive).
2. **Semantic check:** MiniLM encodes the prompt. Cosine similarity is
   calculated against the benign anchors and the threat anchors. The detector
   records the maximum benign similarity and the strongest threat similarity
   and its category.
3. **Decision policy:** The checks are combined in this order:
   - If the structural layer flags the prompt, it is allowed only when the
     benign similarity is more than `0.15` higher than the strongest threat
     similarity **and** the threat similarity is below `0.40`.
   - Otherwise, a structural injection is blocked.
   - If the structural layer does not flag it, the prompt is blocked when the
     strongest threat similarity is above `0.40` and exceeds benign similarity
     by more than `0.10`.
   - Prompts not meeting a blocking condition pass.
4. **Attribution:** For a blocked prompt only, the classifier is run on the
   original text and on versions with each word removed. The difference in
   classifier unsafety scores is returned as a word attribution estimate.

The threat vector is reported as `Nominal Operations` when the strongest
threat similarity is at most `0.38`; otherwise the highest-scoring category
is reported. This display threshold is separate from the block thresholds.
Correlation values are formatted cosine similarities, not calibrated
probabilities.

### `analyze_prompt()` result

The method returns a dictionary with this shape:

```python
{
    "input_prompt": "...",
    "is_safe": True,  # bool
    "security_summary": {
        "verdict": "PASSED",  # or "BLOCKED"
        "reason": "...",
        "primary_threat_vector": "...",
        "threat_correlation": "23.45%",
        "benign_correlation": "74.12%",
    },
    "shap_token_attributions": {},  # empty when passed
}
```

Despite the `shap_token_attributions` field name, these values are
word-removal estimates, not values calculated by the SHAP library. Repeated
words share a dictionary key, and the estimates should be treated as
diagnostic hints rather than proof of model reasoning.

## Requirements

- Python 3.10 or later
- Internet access the first time the detector runs, to download its models
- PyTorch, Transformers, Sentence Transformers, and SentencePiece

The model dependencies can be installed from the package manifest in the
repository's `library` folder. This also installs the importable
`anudesh_kavach` API.

## Setup

From the repository root, install the library and its runtime dependencies into
the active virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .\library
```

If you have not created `.venv` yet:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .\library
```

## Run the standalone examples

Run two built-in example prompts (one benign and one adversarial):

```powershell
.\.venv\Scripts\python.exe .\main\main.py
```

This initializes the pretrained models and may take time on its first run.

## Test prompts interactively

From the repository root:

```powershell
.\.venv\Scripts\python.exe .\main\testPrompt.py
```

Enter prompts at `Prompt>`. The script displays the verdict, safety decision,
threat vector, correlation scores, and any token-attribution estimates. Type
`exit` or `quit` to end the session.

The same script can be run as a module:

```powershell
.\.venv\Scripts\python.exe -m main.testPrompt
```

## Use the installed library API

For applications that only need a boolean answer, use the library API:

```python
import anudesh_kavach

safe = anudesh_kavach.isPromptSafe(
    "Explain integration by parts in Python."
)
print(safe)  # True or False
```

See [`../library/README.md`](../library/README.md) for installation details,
API behavior, and library tests.

## Files

| File | Purpose |
|---|---|
| `main.py` | Detector implementation and built-in two-prompt example |
| `testPrompt.py` | Interactive prompt testing and detailed result display |
| `test.py` | Minimal example using the installed `anudesh_kavach` package |
| `output.txt` | Captured detector output; not needed to run the detector |
| `library.txt` | A short note with library setup instructions |

## Detector result details

The standalone `ExplainableSpaceGuardrail.analyze_prompt(prompt)` method
returns a dictionary containing:

- `is_safe`: boolean safety decision
- `security_summary`: verdict, reason, threat category, and benign/threat
  similarity correlations
- `shap_token_attributions`: word-removal attribution estimates when a prompt
  is flagged

The attribution values are estimates generated by perturbing the input; they
are not definitive explanations of model behavior. Detector decisions can
also produce false positives and false negatives. Validate behavior on
representative data before relying on it.
