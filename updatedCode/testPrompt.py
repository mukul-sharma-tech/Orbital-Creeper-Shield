"""Interactively test prompts with the production guardrail."""

if __package__:
    from .main import ExplainableSpaceGuardrail
else:
    from main import ExplainableSpaceGuardrail


def main() -> None:
    guardrail = ExplainableSpaceGuardrail()
    print("Prompt detector ready. Enter a prompt, or type 'exit' to quit.")

    while True:
        try:
            prompt = input("\nPrompt> ")
        except EOFError:
            print()
            break

        if prompt.strip().lower() in {"exit", "quit"}:
            break

        analysis = guardrail.analyze_prompt(prompt)
        summary = analysis["security_summary"]

        print(f"Verdict: {summary['verdict']}")
        print(f"Safe: {'Yes' if analysis['is_safe'] else 'No'}")
        print(f"Threat vector: {summary['primary_threat_vector']}")

        threat_correlation = summary.get(
            "threat_correlation", summary.get("semantic_intent_correlation")
        )
        if threat_correlation is not None:
            print(f"Threat correlation: {threat_correlation}")

        benign_correlation = summary.get("benign_correlation")
        if benign_correlation is not None:
            print(f"Benign correlation: {benign_correlation}")

        attributions = analysis["shap_token_attributions"]
        if attributions:
            print("Top attribution signals:")
            for token, score in list(attributions.items())[:5]:
                print(f"  {token}: {score}")


if __name__ == "__main__":
    main()
