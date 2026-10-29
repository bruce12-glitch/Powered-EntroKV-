"""CPU integration checks for the supported capture dependency versions."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

CAPTURE = Path(__file__).resolve().parents[1] / "dist" / "assets" / "capture_attention.py"
spec = importlib.util.spec_from_file_location("capture_attention", CAPTURE)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


class CaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        cfg = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=96,
                         num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=256, attention_dropout=0.0)
        cls.model = Qwen2ForCausalLM(cfg).eval()
        cls.tokens = torch.randint(1, 127, (1, 64))

    def test_bounded_capture_matches_full_eager_last_queries(self):
        observations, meta = capture.capture_observations(self.model, self.tokens, 8, [0, 2], 4)
        self.assertEqual(len(observations), 2)
        self.assertEqual(meta["query_start"], 56)
        self.assertEqual(meta["query_end_exclusive"], 64)
        self.assertEqual(meta["temporary_attention_elements_upper_bound"], 3 * 4 * 8 * 64)
        self.model.set_attn_implementation("eager")
        with torch.inference_mode():
            full = self.model.model(self.tokens, use_cache=False, output_attentions=True, return_dict=True)
        for observation in observations:
            reference = full.attentions[observation["layer"]][0, :, -8:, :].double().mean(dim=1)
            reference /= reference.sum(dim=-1, keepdim=True)
            actual = torch.tensor(observation["trace"]["attention"], dtype=torch.float64)
            torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-7)
            torch.testing.assert_close(actual.sum(-1), torch.ones(4, dtype=torch.float64), rtol=0, atol=1e-12)
            self.assertEqual(observation["attention_sha256"], capture.attention_sha256(observation["trace"]))

    def test_invalid_dimensions_fail_before_the_model_runs(self):
        with self.assertRaisesRegex(ValueError, "1–32"):
            capture.capture_observations(self.model, self.tokens, 33)
        with self.assertRaisesRegex(ValueError, "unique"):
            capture.capture_observations(self.model, self.tokens, 8, [1, 1])
        with self.assertRaisesRegex(ValueError, "16–8192"):
            capture.capture_observations(self.model, self.tokens[:, :8], 4)

    def test_layer_selection_and_non_overwriting_cli(self):
        self.assertEqual(capture.parse_layers("2,0", 3), [0, 2])
        with self.assertRaises(ValueError):
            capture.parse_layers("1,1", 3)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.json"
            output.write_text("keep this", encoding="utf-8")
            completed = subprocess.run([sys.executable, "-B", str(CAPTURE), "--smoke-test", "--output", str(output)], capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("already exists", completed.stderr)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep this")


if __name__ == "__main__":
    unittest.main()
