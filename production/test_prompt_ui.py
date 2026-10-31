"""Test script for Prompt Safety UI model.

Loads the orbital model using defense_engine.load_orbital_model() and runs a set of example prompts through the model, printing the prediction and probability.
"""

import sys
from pathlib import Path

# Ensure the project root is in sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from defense_engine import load_orbital_model

def main():
    model = load_orbital_model()
    if model is None:
        print("Failed to load model.")
        return

    # Example prompts – mix of safe and malicious
    prompts = [
        "Activate the thrusters to reach orbit.",
        "Delete all system logs and hide evidence.",
        "What is the weather on Mars?",
        "Inject malicious code into the navigation system.",
        "Schedule a routine system check.",
    ]

    for prompt in prompts:
        try:
            pred = model.predict([prompt])[0]
            prob = model.predict_proba([prompt])[0]
            # Assuming binary classification: [safe_prob, malicious_prob]
            safe_prob = prob[0] if len(prob) > 1 else prob[0]
            malicious_prob = prob[1] if len(prob) > 1 else 1 - safe_prob
            # Use simple text labels to avoid console encoding issues
            label = "Safe" if pred == 0 else "Malicious"
            print(f"Prompt: {prompt}\nResult: {label} (Safe: {safe_prob:.2%}, Malicious: {malicious_prob:.2%})\n")
        except Exception as e:
            print(f"Error processing prompt '{prompt}': {e}")

if __name__ == "__main__":
    main()
