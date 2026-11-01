"""Prompt-safety implementation from the project's finalized detector."""

import os
from typing import TypedDict

import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
MODEL_ID = "protectai/deberta-v3-base-prompt-injection-v2"


class SecuritySummary(TypedDict):
    verdict: str
    reason: str
    primary_threat_vector: str
    threat_correlation: str
    benign_correlation: str


class PromptAnalysis(TypedDict):
    input_prompt: str
    is_safe: bool
    security_summary: SecuritySummary
    shap_token_attributions: dict[str, str]


class ExplainableSpaceGuardrail:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.struct_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID
        ).to(dtype=torch.float16)
        self.classifier = pipeline(
            "text-classification",
            model=self.struct_model,
            tokenizer=self.tokenizer,
        )
        self.semantic_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )

        self.threat_profiles = {
            "Trajectory & Orbit Sabotage": [
                "initiate unscheduled thruster burn or command unauthorized satellite maneuvers and orbit alterations",
                "modify satellite orbital trajectory altitude perigee apogee stationkeeping telemetry",
            ],
            "Credential & Privilege Theft": [
                "stealing session tokens exfiltrating credentials capturing API keys or escalating profile admin permissions",
                "dump database user passwords unauthorized privilege escalation root access tokens",
            ],
            "Command Uplink Spoofing": [
                "bypass ground station command verification or execute raw hex telecommands directly to onboard computer",
                "spoof telecommand frame counter replay attack ground station payload injection",
            ],
            "Telemetry & Comm Disruption": [
                "deactivate transponder downlinks mute telemetry data streams wipe system logs",
                "jam communication links kill beacon signals disable telemetry downlink streams",
            ],
            "Satellite & Cyber Exploits": [
                "hacking satellites ground stations space communications executing unauthorized penetration payloads",
                "generate cyber attack exploits penetration payloads zero-day satellite firmware hacks",
            ],
        }
        self.benign_anchors = [
            "teaching math calculus integration physics academic concepts equations formulas",
            "asking standard coding questions algorithm explanations writing benign python software",
            "roleplay as a teacher professor tutor academic guide code reviewer proofreader",
            "explaining general technical knowledge computer science history databases SQL commands theoretically",
            "discussing theoretical cybersecurity concepts security trade-offs pros and cons of defensive security",
        ]

        self.threat_categories = list(self.threat_profiles)
        self.threat_embeddings = {
            category: self.semantic_model.encode(phrases, convert_to_tensor=True)
            for category, phrases in self.threat_profiles.items()
        }
        self.benign_embeddings = self.semantic_model.encode(
            self.benign_anchors,
            convert_to_tensor=True,
        )

    def _calculate_shap_approximation(self, text: str) -> dict[str, float]:
        """Estimate token contributions by removing each word in turn."""
        words = text.split()
        if not words:
            return {}

        with torch.no_grad():
            base_output = self.classifier(text)
            prediction = base_output[0] if isinstance(base_output, list) else base_output
            base_score = (
                prediction["score"]
                if prediction["label"] == "INJECTION"
                else 1 - prediction["score"]
            )

            shap_values = {}
            for index in range(len(words)):
                perturbed_text = " ".join(words[:index] + words[index + 1 :])
                if not perturbed_text.strip():
                    continue

                perturbed_output = self.classifier(perturbed_text)
                perturbed_prediction = (
                    perturbed_output[0]
                    if isinstance(perturbed_output, list)
                    else perturbed_output
                )
                perturbed_score = (
                    perturbed_prediction["score"]
                    if perturbed_prediction["label"] == "INJECTION"
                    else 1 - perturbed_prediction["score"]
                )
                shap_values[words[index]] = float(base_score - perturbed_score)

        return shap_values

    def analyze_prompt(self, prompt: str) -> PromptAnalysis:
        """Evaluate prompt safety using structural and semantic checks."""
        with torch.no_grad():
            classifier_outputs = self.classifier(prompt)
            prediction = (
                classifier_outputs[0]
                if isinstance(classifier_outputs, list)
                else classifier_outputs
            )
            is_structural_injection = (
                "INJECTION" in prediction["label"].upper()
                or "LABEL_1" in prediction["label"].upper()
            )
            structural_score = prediction["score"]

            prompt_embedding = self.semantic_model.encode(
                prompt,
                convert_to_tensor=True,
            )
            benign_similarities = util.cos_sim(
                prompt_embedding,
                self.benign_embeddings,
            )
            max_benign_score = float(benign_similarities.max())

            threat_scores = {
                category: float(
                    util.cos_sim(prompt_embedding, embeddings).max()
                )
                for category, embeddings in self.threat_embeddings.items()
            }
            top_threat_category = max(threat_scores, key=threat_scores.get)
            max_threat_score = threat_scores[top_threat_category]

            if (
                is_structural_injection
                and max_benign_score > max_threat_score + 0.15
                and max_threat_score < 0.40
            ):
                is_safe = True
                reason = "Passed (Layer 2 Benign Intent Overrode Layer 1 FP)"
            elif is_structural_injection:
                is_safe = False
                reason = (
                    f"Blocked (Layer 1 Structural Injection - "
                    f"{structural_score * 100:.1f}%)"
                )
            elif (
                max_threat_score > 0.40
                and max_threat_score > max_benign_score + 0.10
            ):
                is_safe = False
                reason = f"Blocked (Layer 2 Threat Match: {top_threat_category})"
            else:
                is_safe = True
                reason = "Passed (Nominal Execution)"

            word_attributions = (
                self._calculate_shap_approximation(prompt) if not is_safe else {}
            )
            threat_label = (
                top_threat_category
                if max_threat_score > 0.38
                else "Nominal Operations"
            )

            return {
                "input_prompt": prompt,
                "is_safe": is_safe,
                "security_summary": {
                    "verdict": "PASSED" if is_safe else "BLOCKED",
                    "reason": reason,
                    "primary_threat_vector": threat_label,
                    "threat_correlation": f"{max_threat_score * 100:.2f}%",
                    "benign_correlation": f"{max_benign_score * 100:.2f}%",
                },
                "shap_token_attributions": {
                    word: f"+{value:.4f}" if value > 0 else f"{value:.4f}"
                    for word, value in sorted(
                        word_attributions.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                },
            }
