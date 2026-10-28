"""Page EntroKV numerical reference (MIT). Python 3.10+, standard library only.

One layer / one observation. Head order is contiguous within each KV group.
This selects page indices; it does not execute model inference or free KV memory.
Example: python page_entrokv.py trace.json --retention 20 --block-size 16
"""
import argparse
import json
import math
import sys

ENGINE_VERSION = "0.1.0"
EPSILON = 1e-12


def validate(trace):
    if not isinstance(trace, dict) or trace.get("schema_version") != "1.0":
        raise ValueError('Expected a trace object with schema_version "1.0".')
    hq, hkv = trace.get("query_heads"), trace.get("kv_heads")
    if type(hq) is not int or not 1 <= hq <= 64:
        raise ValueError("query_heads must be an integer in [1, 64].")
    if type(hkv) is not int or not 1 <= hkv <= hq or hq % hkv:
        raise ValueError("kv_heads must divide query_heads.")
    rows = trace.get("attention")
    if not isinstance(rows, list) or len(rows) != hq or not isinstance(rows[0], list):
        raise ValueError("Expected query_heads attention rows.")
    tokens = len(rows[0])
    if not 16 <= tokens <= 8192 or tokens * hq > 262144:
        raise ValueError("Trace exceeds the supported token/entry limits.")
    sink = trace.get("sink_size", 4)
    if type(sink) is not int or not 0 <= sink < tokens:
        raise ValueError("sink_size must be an integer in [0, tokens).")
    for i, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != tokens:
            raise ValueError(f"Head {i}: inconsistent row length.")
        if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in row):
            raise ValueError(f"Head {i}: expected finite nonnegative probabilities.")
        if abs(sum(row) - 1) > 1e-5:
            raise ValueError(f"Head {i}: probability sum is not within 1e-5 of one.")
    return rows, hq, hkv, sink


def topk(scores, k):
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]


def pages(row, block_size):
    return [max(row[i:i + block_size]) for i in range(0, len(row), block_size)]


def select_pages(trace, block_size=16, retention=20, tau=0.5, isolate=True):
    rows, hq, hkv, sink = validate(trace)
    if type(block_size) is not int or not 1 <= block_size <= 512:
        raise ValueError("block_size must be an integer in [1, 512].")
    if not math.isfinite(retention) or not 0 < retention <= 100:
        raise ValueError("retention must be in (0, 100].")
    if not math.isfinite(tau) or not 0.05 <= tau <= 10:
        raise ValueError("tau must be in [0.05, 10].")
    tokens, ratio = len(rows[0]), hq // hkv
    page_count = math.ceil(tokens / block_size)
    quota = max(1, math.floor(page_count * retention / 100))
    entropies = []
    for row in rows:
        residual = row[sink:] if isolate else row
        denominator = max(sum(residual), EPSILON)
        collision = sum((v / denominator) ** 2 for v in residual)
        entropies.append(-math.log(collision + EPSILON))
    groups = []
    for g in range(hkv):
        heads = rows[g * ratio:(g + 1) * ratio]
        entropy = entropies[g * ratio:(g + 1) * ratio]
        logits = [-h / tau for h in entropy]
        maximum = max(logits)
        exps = [math.exp(h - maximum) for h in logits]
        z = sum(exps)
        weights = [v / z for v in exps]
        pooled = [sum(weights[h] * heads[h][t] for h in range(ratio)) for t in range(tokens)]
        mean = [sum(heads[h][t] / ratio for h in range(ratio)) for t in range(tokens)]
        scores, mean_scores = pages(pooled, block_size), pages(mean, block_size)
        independent = [topk(pages(row, block_size), quota) for row in heads]
        union = sorted(set(p for selection in independent for p in selection))
        groups.append({"group": g, "entropies": entropy, "weights": weights,
                       "scores": scores, "selected": topk(scores, quota),
                       "meanSelected": topk(mean_scores, quota),
                       "independentUnion": union, "uor": len(union) / quota})
    return {"engine_version": ENGINE_VERSION, "tokens": tokens, "pages": page_count,
            "quota": quota, "realizedRetention": quota / page_count * 100,
            "groups": groups, "note": "Selection analysis only; no inference or allocation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", help="Normalized attention-trace JSON file")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--retention", type=float, default=20)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--raw-entropy", action="store_true", help="Disable sink isolation")
    args = parser.parse_args()
    try:
        with open(args.trace, encoding="utf-8") as handle:
            trace = json.load(handle)
        output = select_pages(trace, args.block_size, args.retention, args.tau, not args.raw_entropy)
        json.dump(output, sys.stdout, indent=2, allow_nan=False)
        sys.stdout.write("\n")
    except (ValueError, OSError, TypeError, OverflowError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
