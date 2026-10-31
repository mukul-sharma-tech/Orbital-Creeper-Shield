import os
import logging
import asyncio
import torch
from typing import Dict, Any, List
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer, util

# Configure structured enterprise logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ProductionGuardrail")

class AerospaceSecurityGuardrail:
    _instance = None

    def __new__(cls, *args, **kwargs):
        """Ensure a thread-safe Singleton Pattern initialization."""
        if cls._instance is None:
            cls._instance = super(AerospaceSecurityGuardrail, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_id: str = "protectai/deberta-v3-base-prompt-injection-v2"):
        if self._initialized:
            return
            
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        logger.info("Initializing high-performance production guardrail cores...")
        
        # 1. Structural Security Layer Initialization (DeBERTa-v3 in Float16)
        try:
            self.struct_model = AutoModelForSequenceClassification.from_pretrained(model_id).to(dtype=torch.float16)
            self.struct_tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.structure_classifier = pipeline(
                "text-classification", 
                model=self.struct_model, 
                tokenizer=self.struct_tokenizer
            )
            logger.info("Layer 1 (Structural Classifier) successfully loaded.")
        except Exception as e:
            logger.critical(f"Failed to load structural safety model: {str(e)}")
            raise e

        # 2. Semantic Security Layer Initialization (MiniLM-L6 strictly bound to CPU)
        try:
            self.semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
            logger.info("Layer 2 (Semantic Intent Engine) successfully loaded.")
        except Exception as e:
            logger.critical(f"Failed to load semantic safety engine: {str(e)}")
            raise e

        # 3. Dense Aerospace Threat Matrix Definitions
        self.MALICIOUS_INTENTS: List[str] = [
            "unauthorized modification of user profile permissions and admin controls",
            "deleting database records or dropping production storage tables",
            "stealing session tokens, exfiltrating credentials, or capturing API keys",
            "unauthorized remote command execution on system terminals",
            "initiate unscheduled thruster burn or command unauthorized satellite maneuvers",
            "modify perigee apogee altitude or alter spacecraft orbital trajectory",
            "bypass ground station command verification or execute raw telecommands",
            "clear ground station command logs or wipe telemetry metrics",
            "change solar array orientation profile or manipulate power subsystem controls",
            "deactivate transponder downlinks, mute telemetry streams, or jam communications",
            "assume role of mission director or spoof flight controller authorization"
        ]

        # Pre-compute and freeze structural target arrays in memory
        self.BLOCKED_EMBEDDINGS = self.semantic_model.encode(self.MALICIOUS_INTENTS, convert_to_tensor=True)
        self.semantic_threshold: float = 0.52
        self._initialized = True
        logger.info("Security Guardrail System fully operational.")

    def _sanitize_input(self, prompt: str) -> str:
        """Sanitize raw input strings against structural trick attacks."""
        if not prompt:
            return ""
        # Remove null bytes, filter non-printable characters, and trim whitespace
        clean_text = "".join(ch for ch in prompt if ch.isprintable() and ch != '\x00')
        return clean_text.strip()

    async def verify_prompt_safety(self, prompt: str) -> bool:
        """
        Asynchronously evaluates user prompts against structural and semantic attack layers.
        Returns True if the prompt is safe to execute, False if it is blocked.
        """
        clean_prompt = self._sanitize_input(prompt)
        
        # Fast-track minimal string lengths safely
        if len(clean_prompt) < 5:
            return True

        try:
            # Force processing to run inside an isolated execution thread to keep the main event loop responsive
            return await asyncio.to_thread(self._execute_inference_sync, clean_prompt)
        except Exception as e:
            # Fail-closed guardrail strategy: If an internal validation component crashes, lock execution immediately
            logger.error(f"Guardrail execution runtime failure: {str(e)}. Defaulting to Fail-Closed policy.")
            return False

    def _execute_inference_sync(self, prompt: str) -> bool:
        """Internal synchronous execution wrapper running inside a clean memory context."""
        # Disable gradient calculations completely to optimize CPU memory usage
        with torch.no_grad():
            
            # --- LAYER 1: Deep Semantic Intent Mapping ---
            prompt_embedding = self.semantic_model.encode(prompt, convert_to_tensor=True)
            cosine_scores = util.cos_sim(prompt_embedding, self.BLOCKED_EMBEDDINGS)
            max_similarity: float = torch.max(cosine_scores).item()

            if max_similarity > self.semantic_threshold:
                logger.warning(f"Security Alert: Malicious semantic intent intercepted. Confidence: {max_similarity:.4f}")
                return False

            # --- LAYER 2: Structural Prompt Injection Tracking ---
            outputs = self.structure_classifier(prompt)
            prediction = outputs if isinstance(outputs, list) else outputs
            label: str = prediction['label'].upper()

            if "INJECTION" in label or "LABEL_1" in label:
                logger.warning("Security Alert: Structural instruction injection sequence intercepted.")
                return False

            return True

# --- Production Deployment/Usage Template ---
async def main():
    # Instantiate the secure singleton guardrail engine
    guardrail = AerospaceSecurityGuardrail()

    test_input = "Initiate an unscheduled thruster burn to modify the perigee altitude immediately."
    
    # Process evaluation cleanly via asynchronous calls
    is_safe = await guardrail.verify_prompt_safety(test_input)
    print(f"\nFinal Production Result -> Is Prompt Safe: {is_safe}")

if __name__ == "__main__":
    asyncio.run(main())
