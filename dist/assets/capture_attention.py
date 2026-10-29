"""Bounded Qwen2 attention capture for Page EntroKV v0.2 (MIT).

Run locally: python capture_attention.py --prompt-file prompt.txt --output capture.json
Install requirements-capture.txt first. No remote-code execution or serving endpoint.
The output contains attention observations, not generation-quality measurements.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import struct
import subprocess
import sys

VERSION = "0.2.0"
MAX_ENTRIES = 1048576


def attention_sha256(trace):
    payload = bytearray(b"PageEntroKV/attention-v2\0")
    payload.extend(struct.pack("<IIII", trace["query_heads"], trace["kv_heads"],
                               trace["sink_size"], len(trace["attention"][0])))
    for row in trace["attention"]:
        payload.extend(struct.pack("<" + "d" * len(row), *row))
    return hashlib.sha256(payload).hexdigest()


def parse_layers(value, count):
    if value == "all":
        return list(range(count))
    try:
        layers = [int(v.strip()) for v in value.split(",")]
    except ValueError as error:
        raise ValueError("--layers must be 'all' or comma-separated layer indices.") from error
    if not layers or len(set(layers)) != len(layers) or any(v < 0 or v >= count for v in layers):
        raise ValueError(f"Layer indices must be unique and in [0, {count - 1}].")
    return sorted(layers)


def capture_observations(model, input_ids, observation_window=8, layers=None, sink_size=4):
    """Prefill with SDPA, then capture only the final Q query rows with eager attention.

    All layers execute; only selected layers are exported. Temporary attention storage is
    O(L * H_Q * Q * T), with Q <= 32, instead of materializing full T * T rows.
    Raw cast/mean row sums are recorded; normalization to a probability vector is explicit.
    """
    import torch

    cfg = model.config
    if cfg.model_type != "qwen2" or not hasattr(model, "set_attn_implementation"):
        raise ValueError("v0.2 capture supports Qwen2 models with runtime attention switching.")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Capture requires exactly one unpadded token sequence.")
    tokens = input_ids.shape[1]
    if not 16 <= tokens <= 8192:
        raise ValueError("Capture context must contain 16–8192 tokens.")
    if not 1 <= observation_window <= min(32, tokens):
        raise ValueError("Observation window must be 1–32 and at most the token count.")
    if not 0 <= sink_size < tokens:
        raise ValueError("sink_size must be less than the context length.")
    chosen = list(range(cfg.num_hidden_layers)) if layers is None else layers
    if not chosen or any(type(i) is not int or i < 0 or i >= cfg.num_hidden_layers for i in chosen):
        raise ValueError("Invalid selected layers.")
    if len(set(chosen)) != len(chosen):
        raise ValueError("Selected layers must be unique.")
    if len(chosen) * cfg.num_attention_heads * tokens > MAX_ENTRIES:
        raise ValueError("Export exceeds 1,048,576 entries. Reduce --max-tokens or select fewer --layers.")
    if cfg.num_attention_heads > 64 or cfg.num_attention_heads % cfg.num_key_value_heads:
        raise ValueError("Unsupported GQA head geometry.")
    # Refuse excessive temporary attention tensors before running model forwards.
    temporary_elements = cfg.num_hidden_layers * cfg.num_attention_heads * observation_window * tokens
    if temporary_elements > 67108864:
        raise ValueError("Attention window is too large. Reduce --observation-window or --max-tokens.")
    prefix = tokens - observation_window
    backbone = model.model
    cache = None
    model.eval()
    with torch.inference_mode():
        if prefix:
            model.set_attn_implementation("sdpa")
            warm = backbone(input_ids=input_ids[:, :prefix],
                            attention_mask=torch.ones((1, prefix), dtype=torch.long, device=input_ids.device),
                            use_cache=True, output_attentions=False, return_dict=True)
            cache = warm.past_key_values
            del warm
        model.set_attn_implementation("eager")
        observed = backbone(input_ids=input_ids[:, prefix:], past_key_values=cache,
                            attention_mask=torch.ones((1, tokens), dtype=torch.long, device=input_ids.device),
                            position_ids=torch.arange(prefix, tokens, device=input_ids.device).unsqueeze(0),
                            use_cache=True, output_attentions=True, return_dict=True)
        if not observed.attentions or len(observed.attentions) != cfg.num_hidden_layers:
            raise RuntimeError("The eager backend did not return all layer attention observations.")
        result, errors = [], []
        for layer in sorted(chosen):
            tensor = observed.attentions[layer]
            expected = (1, cfg.num_attention_heads, observation_window, tokens)
            if tensor is None or tuple(tensor.shape) != expected:
                raise RuntimeError(f"Layer {layer}: expected attention shape {expected}; got {getattr(tensor, 'shape', None)}.")
            rows = tensor[0].to(dtype=torch.float64).mean(dim=1)
            sums = rows.sum(dim=-1, keepdim=True)
            if not torch.isfinite(rows).all() or (rows < 0).any() or (sums <= 0).any():
                raise RuntimeError(f"Layer {layer}: invalid attention probabilities.")
            maximum_error = (sums - 1).abs().max().item()
            if maximum_error > 0.02:
                raise RuntimeError(f"Layer {layer}: raw attention sum error {maximum_error} exceeds 0.02.")
            rows = rows / sums
            trace = {"schema_version": "1.0", "query_heads": cfg.num_attention_heads,
                     "kv_heads": cfg.num_key_value_heads, "sink_size": sink_size,
                     "attention": rows.cpu().tolist(),
                     "metadata": {"model": getattr(cfg, "_name_or_path", "qwen2"),
                                  "layer": str(layer), "description": f"Mean of final {observation_window} query rows"}}
            result.append({"id": f"layer-{layer:03d}", "layer": layer, "trace": trace,
                           "attention_sha256": attention_sha256(trace)})
            errors.append({"layer": layer, "raw_mean_row_sum_max_abs_error": maximum_error})
        del observed
    return result, {"query_start": prefix, "query_end_exclusive": tokens,
                    "aggregation": "arithmetic_mean_over_query_rows",
                    "prefix_backend": "sdpa" if prefix else "none", "observation_backend": "eager",
                    "row_normalization": "Explicit float64 normalization after averaging cast attention rows",
                    "normalization_diagnostics": errors,
                    "temporary_attention_elements_upper_bound": temporary_elements}


def code_revision():
    try:
        # A downloaded or edited script must not claim that HEAD identifies its bytes.
        subprocess.check_output(["git", "ls-files", "--error-unmatch", str(Path(__file__).resolve())],
                                cwd=Path(__file__).parent, stderr=subprocess.DEVNULL)
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", str(Path(__file__).resolve())],
                                        cwd=Path(__file__).parent, stderr=subprocess.DEVNULL)
        if dirty.strip():
            return None
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--revision", default="main", help="Prefer an immutable model commit SHA")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--output", type=Path, default=Path("capture.json"))
    parser.add_argument("--max-tokens", type=int, default=1024, choices=range(16, 8193), metavar="16..8192")
    parser.add_argument("--observation-window", type=int, default=8)
    parser.add_argument("--layers", default="all", help="all, or zero-indexed comma-separated layer indices")
    parser.add_argument("--sink-size", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-token-text", action="store_true", help="Optional; exports decoded token text from your prompt")
    parser.add_argument("--smoke-test", action="store_true", help="Tiny random-weight Qwen2, CPU, no model download; explicitly labeled")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists. Choose a new path to preserve the previous capture.")
    if not args.smoke_test and args.prompt_file is None:
        parser.error("--prompt-file is required unless --smoke-test is selected.")
    if not 1 <= args.observation_window <= 32:
        parser.error("--observation-window must be between 1 and 32.")
    if not 0 <= args.seed <= 2147483647:
        parser.error("--seed must be an integer in [0, 2147483647].")
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable. Use --device cpu or a configured GPU environment.")
        tokenizer = None
        if args.smoke_test:
            cfg = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=96,
                              num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                              max_position_embeddings=256, attention_dropout=0.0)
            cfg._name_or_path = "random-weight-qwen2-smoke-test"
            model = Qwen2ForCausalLM(cfg).to("cpu").eval()
            input_ids = torch.randint(1, 127, (1, min(args.max_tokens, 64)))
            name, source = "Random-weight Qwen2 smoke test", "random_weight_model"
            prompt_sha, truncated, original_tokens = None, False, input_ids.shape[1]
        else:
            if args.device == "cpu" and args.dtype == "float16":
                raise ValueError("Use float32 for the supported CPU capture path.")
            prompt_bytes = args.prompt_file.read_bytes()
            if len(prompt_bytes) > 4194304:
                raise ValueError("Prompt file exceeds 4 MiB.")
            prompt = prompt_bytes.decode("utf-8")
            tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
            original_tokens = encoded.shape[1]
            truncated = original_tokens > args.max_tokens
            input_ids = encoded[:, :args.max_tokens].to(args.device)
            if input_ids.shape[1] < 16:
                raise ValueError("The prompt must tokenize to at least 16 tokens.")
            model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
                        trust_remote_code=False, torch_dtype=getattr(torch, args.dtype), attn_implementation="sdpa").to(args.device).eval()
            name, source = f"{args.model} attention capture", "model_capture"
            prompt_sha = hashlib.sha256(prompt_bytes).hexdigest()
        layers = parse_layers(args.layers, model.config.num_hidden_layers)
        observations, capture = capture_observations(model, input_ids, args.observation_window, layers, args.sink_size)
        capture.update({"max_tokens": args.max_tokens, "original_token_count": original_tokens,
                        "truncation": "prefix_keep_first_tokens" if truncated else "none",
                        "prompt_format": "raw_text_with_tokenizer_special_tokens", "prompt_file_sha256": prompt_sha})
        device = str(next(model.parameters()).device)
        bundle = {"schema_version": "2.0", "kind": "attention_bundle", "name": name, "source": source,
                  "token_ids": input_ids[0].cpu().tolist(), "observations": observations,
                  "metadata": {"model": getattr(model.config, "_name_or_path", args.model),
                    "requested_revision": None if args.smoke_test else args.revision,
                    "resolved_model_revision": getattr(model.config, "_commit_hash", None),
                    "resolved_tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash") if tokenizer else None,
                    "capture_tool_version": VERSION, "capture_code_revision": code_revision(),
                    "capture_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "seed": args.seed, "capture": capture,
                    "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "transformers": transformers.__version__, "device": device,
                        "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None,
                        "model_dtype": str(next(model.parameters()).dtype),
                        "determinism": "eval + seeded RNG; cross-device bitwise determinism is not asserted"}}}
        if args.include_token_text and tokenizer:
            bundle["token_text"] = [tokenizer.decode([i], clean_up_tokenization_spaces=False)[:256] for i in bundle["token_ids"]]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive create: never overwrite a previous scientific artifact.
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(bundle, handle, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
        print(f"Captured {len(observations)} layers, {input_ids.shape[1]} tokens. Source: {source}. Saved {args.output}.")
        if truncated:
            print("Notice: prompt was truncated to its first --max-tokens tokens.", file=sys.stderr)
    except ImportError as error:
        parser.exit(2, f"Missing capture dependencies: {error}. Install requirements-capture.txt first.\n")
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f"Capture failed: {error}\n")


if __name__ == "__main__":
    main()
