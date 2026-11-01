"""Interactively test prompts with the integrated mission guardrail."""

if __package__:
    from .main import IntegratedMissionGuardrail
else:
    from main import IntegratedMissionGuardrail


def main() -> None:
    guardrail = IntegratedMissionGuardrail()
    print("Prompt detector ready. Enter a prompt, or type 'exit' to quit.")

    while True:
        try:
            prompt = input("\nPrompt> ")
        except EOFError:
            print()
            break

        if prompt.strip().lower() in {"exit", "quit"}:
            break

        result = guardrail.audit_incoming_prompt(prompt)
        summary = result["audit_summary"]
        threat_vector = summary.get("threat_vector") or summary.get("primary_threat_vector")
        intent_correlation = summary.get("intent_correlation") or summary.get(
            "semantic_intent_correlation"
        )

        print(f"Verdict: {result['verdict']}")
        print(f"Safe: {'Yes' if result['is_safe'] else 'No'}")
        print(f"Triggered layer: {result['triggered_layer']}")
        print(f"Threat vector: {threat_vector or 'Not reported'}")
        print(f"Intent correlation: {intent_correlation or 'Not reported'}")

        attributions = result["attribution_metrics"]
        if attributions:
            print("Top attribution signals:")
            for token, weight in list(attributions.items())[:5]:
                print(f"  {token}: {weight}")


if __name__ == "__main__":
    main()
