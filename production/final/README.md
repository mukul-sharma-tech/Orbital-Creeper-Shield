# Anudesh Kavach: Core Mission Command Gateway

Anudesh Kavach is a high-performance, multi-layered security gateway designed to protect satellite constellation telemetry pipelines and ground stations, such as ISRO NavIC, Gaganyaan, and CubeSat swarms, from adversarial prompt injections, malware, and self-replicating polymorphic AI worms.

By combining classical machine-learning heuristics with transformer-based models, the framework aims to provide **zero-trust security enforcement** alongside efficient CPU inference.

---

## 🛰️ Architectural Design & Multi-Layer Cascade

The core innovation is a **Cascading Hybrid Guardrail Stack**. Instead of passing raw user commands directly to a large language model (LLM) or executing them immediately on an On-Board Computer (OBC), inputs traverse two defensive layers sequentially.

### System Architecture

```text
┌────────────────────────────────────────────────────┐
│ Incoming Ground / Inter-Satellite Command (TC)     │
└─────────────────────────┬──────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────┐
│ LAYER 1: HEURISTICS & LINEAR SVM ENGINE             │
│                                                    │
│ • Character / Hex / Base64 Density Checks          │
│ • High-Entropy Payload Indicators                  │
│ • Lexical and Structural Feature Analysis          │
└─────────────────────────┬──────────────────────────┘
                          │
              ┌───────────┴────────────┐
              │                        │
        [MALICIOUS]                 [PASSED]
              │                        │
              ▼                        ▼
      ┌──────────────┐  ┌────────────────────────────────┐
      │   BLOCKED    │  │ LAYER 2: DUAL-CORE TRANSFORMER │
      └──────────────┘  │             STACK              │
                        │                                │
                        │ ┌────────────────┐ ┌─────────┐ │
                        │ │ DeBERTa-v3     │ │ MiniLM  │ │
                        │ │ Structural     │ │ Semantic│ │
                        │ │ Checks         │ │ Analysis│ │
                        │ └────────────────┘ └─────────┘ │
                        └────────────────┬───────────────┘
                                         │
                             ┌───────────┴───────────┐
                             │                       │
                       [MALICIOUS]                [PASSED]
                             │                       │
                             ▼                       ▼
                     ┌──────────────┐   ┌──────────────────────┐
                     │    BLOCKED   │   │ Independent Command  │
                     └──────────────┘   │ Validation & Policy  │
                                        │ Enforcement          │
                                        └──────────┬───────────┘
                                                   │
                                                   ▼
                                  ┌────────────────────────────┐
                                  │ Authorized Telecommand     │
                                  │ Execution Gateway          │
                                  └────────────────────────────┘
```

> **Security principle:** Passing the detection layers does not automatically authorize execution. Every command must also pass independent authentication, authorization, validation, and mission-safety checks.

---

## Layer 1: Heuristics & Linear SVM Engine (`svm.py`)

### Objective

Intercept suspicious code payloads, obfuscation attempts, and known malicious patterns before invoking computationally expensive deep-learning models.

### Mechanics

The engine extracts structural and lexical features, including:

- Shannon entropy.
- Character-distribution spacing, such as `d i s r e g a r d`.
- Base64 and hexadecimal character-density ratios.
- Indicators of suspicious propagation syntax.
- Character-level and word-level TF-IDF features.

These numerical features are combined and passed to a calibrated **Linear Support Vector Machine (`LinearSVC`)** for classification.

### Production Advantage

The lightweight screening layer is designed to reduce unnecessary deep-learning inference and identify suspicious inputs early.

Actual latency depends on the feature-extraction pipeline, model configuration, and target hardware. Microsecond-level execution should be claimed only after benchmarking.

---

## Layer 2: Dual-Core Explainable Deep Learning Head (`deepLearning.py`)

Inputs that pass Layer 1 undergo contextual analysis using two complementary transformer-based components. Inference can be performed under `torch.no_grad()` to disable gradient calculation and reduce unnecessary memory consumption.

### 1. Structural Check: DeBERTa-v3

**Model:** `protectai/deberta-v3-base-prompt-injection-v2`

The DeBERTa-based classifier analyzes linguistic patterns associated with:

- Prompt injection.
- Context hijacking.
- Developer-mode role-play.
- Instruction and output-prefix overrides.

The model may use half-precision (`float16`) to reduce memory consumption where the hardware and inference runtime support it.

### 2. Semantic Intent Check: all-MiniLM-L6-v2

**Model:** `all-MiniLM-L6-v2`

The MiniLM sentence encoder maps input text into a **384-dimensional embedding space**.

The system computes cosine similarity between the input embedding and reference embeddings representing aerospace-related threat categories, including:

- Unauthorized thruster firings.
- Trajectory alteration.
- Telemetry suppression.
- Suspicious command-execution requests.

Cosine similarity provides a semantic screening signal. It is not, by itself, a calibrated probability of maliciousness or proof that a command is unsafe.

### 3. Explainability Matrix: SHAP Approximation

When an input is flagged, a localized word-perturbation routine can estimate how individual words influence the model's prediction.

The resulting importance scores can help explain why an input was flagged and support debugging or security analysis.

**Note:** Word-perturbation scores should be described as SHAP-style approximations unless a genuine SHAP implementation is used. These explanations are diagnostic estimates rather than definitive proof of a model's reasoning.

---

## 📁 Workspace Structure

The project directory follows this structure:

```text
Orbital-Creeper-Shield/
├── svm.py    # Classical Scikit-learn heuristic pipeline
├── deepLearning.py             # DeBERTa / MiniLM models and explainability
├── main.py              # Central orchestrator and pipeline entry point
└── README.md            # Project documentation
```

### File Responsibilities

| File | Responsibility |
|---|---|
| `svm.py` | Feature extraction, TF-IDF processing, and SVM-based screening |
| `deepLearning.py` | Transformer classification, semantic similarity, and optional explanations |
| `main.py` | Pipeline orchestration, decision handling, and execution flow |
| `README.md` | Architecture, installation, usage, and operational documentation |

---

## 🚀 Quickstart & Verification Matrix

### 1. Installation Requirements

Create and activate a Python virtual environment, then install the required dependencies.

```bash
pip install torch transformers sentence-transformers scikit-learn numpy pandas joblib
```

### 2. Execution

Run the central orchestration script from the project root:

```bash
python main.py
```

The orchestrator should initialize the required models, evaluate the input, combine the detection results according to the configured policy, and return a structured decision.

To enter prompts interactively and inspect each detection result, run the test interface from this directory:

```bash
python test.py
```

Alternatively, run it as a package from the repository root:

```bash
python -m production.final.test
```

Type `exit` or `quit` to stop the interactive session.

### 3. Expected Output Characteristics

A successful run should demonstrate the system's ability to distinguish routine inputs from potentially adversarial payloads.

- **Routine Telemetry / Mathematical Requests:** Inputs not flagged by the detection layers can proceed to independent command validation. They may be logged as `Nominal Operations (No Threat Profile Match)`.
- **Obfuscated / Suspicious Inputs:** Inputs flagged by Layer 1 or Layer 2 should be blocked or quarantined according to the configured security policy.
- **Model or Runtime Failures:** Commands must not proceed to execution if required security checks cannot be completed.

The actual classification accuracy, false-positive rate, false-negative rate, and latency must be measured using a representative verification dataset.

---

## 🛡️ Production Operations & Failure Modes

### 1. Fail-Closed Isolation Strategy

The pipeline should handle failures in each layer explicitly. If a model fails to load, data processing encounters a critical error, or a required dependency becomes unavailable, the system must prevent the affected command from reaching the execution gateway.

A fail-closed implementation should:

- Reject or quarantine the affected command.
- Record the failure and relevant diagnostic information.
- Prevent downstream execution on every exception path.
- Require successful recovery before normal command processing resumes.

Using `try/except` blocks alone does not guarantee fail-closed behavior. The execution gateway should independently enforce authorization and safety policies.

### 2. CPU Optimization Envelope

The system can be optimized for constrained ground-station processors and edge servers through:

- Benchmarking inference on the intended CPU architecture.
- Using a precision format supported by the target hardware and runtime.
- Precomputing and caching static semantic threat embeddings.
- Limiting input length and resource