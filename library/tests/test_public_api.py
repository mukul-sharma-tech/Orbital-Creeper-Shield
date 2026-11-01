import sys
import types
import unittest
from unittest.mock import patch

import anudesh_kavach


class FakeGuardrail:
    instances = 0

    def __init__(self) -> None:
        type(self).instances += 1

    def analyze_prompt(self, prompt: str) -> dict[str, bool]:
        return {"is_safe": prompt == "safe prompt"}


class PublicApiTests(unittest.TestCase):
    def setUp(self) -> None:
        anudesh_kavach._guardrail = None
        FakeGuardrail.instances = 0

    def test_returns_detector_boolean_and_reuses_detector(self) -> None:
        detector_module = types.ModuleType("anudesh_kavach._guardrail")
        detector_module.ExplainableSpaceGuardrail = FakeGuardrail

        with patch.dict(sys.modules, {"anudesh_kavach._guardrail": detector_module}):
            self.assertIs(anudesh_kavach.isPromptSafe("safe prompt"), True)
            self.assertIs(anudesh_kavach.isPromptSafe("unsafe prompt"), False)

        self.assertEqual(FakeGuardrail.instances, 1)

    def test_rejects_non_string_input(self) -> None:
        with self.assertRaises(TypeError):
            anudesh_kavach.isPromptSafe(123)  # type: ignore[arg-type]

        self.assertIsNone(anudesh_kavach._guardrail)


if __name__ == "__main__":
    unittest.main()
