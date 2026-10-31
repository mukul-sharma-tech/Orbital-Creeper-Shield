import os
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer, util

# Ensure structural environment tracking is clean
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
model_id = "protectai/deberta-v3-base-prompt-injection-v2"

class ExplainableSpaceGuardrail:
    def __init__(self):
        # Initialize core models in float16 for CPU efficiency
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.struct_model = AutoModelForSequenceClassification.from_pretrained(model_id).to(dtype=torch.float16)
        self.classifier = pipeline("text-classification", model=self.struct_model, tokenizer=self.tokenizer)
        self.semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        
        # Categorized aerospace threat profiles
        self.THREAT_MAPPING = {
            "Trajectory & Orbit Sabotage": "initiate unscheduled thruster burn or command unauthorized satellite maneuvers and orbit alterations",
            "Credential & Privilege Theft": "stealing session tokens, exfiltrating credentials, capturing API keys, or escalating profile admin permissions",
            "Command Uplink Spoofing": "bypass ground station command verification or execute raw hex telecommands directly to the onboard computer",
            "Telemetry & Comm Disruption": "deactivate transponder downlinks, mute telemetry data streams, or wipe system logs"
        }
        
        self.labels = list(self.THREAT_MAPPING.keys())
        self.intent_embeddings = self.semantic_model.encode(list(self.THREAT_MAPPING.values()), convert_to_tensor=True)

    def _calculate_shap_approximation(self, text: str) -> dict:
        """
        Computes a fast local Shapley-value approximation for text tokens
        without requiring a heavy external SHAP dependency library on CPU.
        """
        words = text.split()
        if not words:
            return {}
            
        with torch.no_grad():
            base_output = self.classifier(text)
            prediction = base_output[0] if isinstance(base_output, list) else base_output
            base_score = prediction['score'] if prediction['label'] == 'INJECTION' else 1 - prediction['score']
            
            shap_values = {}
            # Perturbation loop: drop one word at a time to measure marginal contribution
            for i, word in enumerate(words):
                perturbed_words = words[:i] + words[i+1:]
                perturbed_text = " ".join(perturbed_words)
                
                if not perturbed_text.strip():
                    continue
                    
                p_output = self.classifier(perturbed_text)
                p_prediction = p_output[0] if isinstance(p_output, list) else p_output
                p_score = p_prediction['score'] if p_prediction['label'] == 'INJECTION' else 1 - p_prediction['score']
                
                # Marginal contribution of the word to the unsafety score
                shap_values[word] = float(base_score - p_score)
                
        return shap_values

    def analyze_prompt(self, prompt: str) -> dict:
        """Evaluates a prompt and generates a comprehensive SHAP + Intent audit log."""
        with torch.no_grad():
            # 1. Evaluate Structure via Layer 1
            l1_outputs = self.classifier(prompt)
            prediction = l1_outputs[0] if isinstance(l1_outputs, list) else l1_outputs
            is_structural_injection = "INJECTION" in prediction['label'].upper() or "LABEL_1" in prediction['label'].upper()

            # 2. Evaluate Semantic Intent via Layer 2
            prompt_embedding = self.semantic_model.encode(prompt, convert_to_tensor=True)
            cosine_scores = util.cos_sim(prompt_embedding, self.intent_embeddings)
            
            scores_vector = cosine_scores[0] if len(cosine_scores.shape) > 1 else cosine_scores
            max_sim_idx = torch.argmax(scores_vector).item()
            max_similarity = scores_vector[max_sim_idx].item()

            # Clean Mapping Fix: Only label the threat category if it meets a minimum baseline check
            if max_similarity > 0.40:
                matched_intent_category = self.labels[max_sim_idx]
            else:
                matched_intent_category = "Nominal Operations (No Threat Profile Match)"

            # Determine security boundary status
            is_safe = not (is_structural_injection or max_similarity > 0.52)
            
            # 3. Compute SHAP attributions if flagged as unsafe
            word_attributions = {}
            if not is_safe:
                word_attributions = self._calculate_shap_approximation(prompt)

            # 4. Generate Structured Audit Summary
            return {
                "input_prompt": prompt,
                "is_safe": is_safe,
                "security_summary": {
                    "verdict": "PASSED" if is_safe else "BLOCKED",
                    "primary_threat_vector": matched_intent_category,
                    "semantic_intent_correlation": f"{max_similarity * 100:.2f}%"
                },
                "shap_token_attributions": {
                    word: f"+{val:.4f}" if val > 0 else f"{val:.4f}" 
                    for word, val in sorted(word_attributions.items(), key=lambda item: item[1], reverse=True)
                }
            }
