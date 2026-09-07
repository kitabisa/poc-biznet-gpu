#!/usr/bin/env python3
"""Test vLLM inference server with a batch of system design questions."""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "http://YOUR_VLLM_HOST:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen3.8-27B"

SYSTEM_DESIGN_CASES = [
    {
        "id": 1,
        "title": "URL Shortener",
        "prompt": "Design a URL shortening service like bit.ly. Cover the API, ID generation strategy, data model, and how you would handle 50,000 redirects per second at read time.",
    },
    {
        "id": 2,
        "title": "Rate Limiter",
        "prompt": "Design a distributed rate limiter that can be used across multiple API gateway nodes. Compare token bucket vs sliding window log, and explain how you would keep counters consistent across nodes.",
    },
    {
        "id": 3,
        "title": "News Feed System",
        "prompt": "Design a social media news feed (like Twitter/X home timeline). Discuss fan-out-on-write vs fan-out-on-read, and how you would handle celebrity accounts with millions of followers.",
    },
    {
        "id": 4,
        "title": "Distributed Cache",
        "prompt": "Design a distributed in-memory cache system similar to Redis Cluster. Explain sharding strategy, replication, and cache invalidation approach.",
    },
    {
        "id": 5,
        "title": "Chat Application",
        "prompt": "Design a real-time one-on-one and group chat application like WhatsApp. Cover message delivery guarantees, offline message storage, and read receipts.",
    },
    {
        "id": 6,
        "title": "Payment Processing System",
        "prompt": "Design a payment processing system that must guarantee exactly-once charge semantics even under network retries. Discuss idempotency keys, double-entry ledger design, and reconciliation.",
    },
    {
        "id": 7,
        "title": "Ride-Hailing Dispatch",
        "prompt": "Design the driver-matching component of a ride-hailing app like Gojek/Grab. Explain how you'd index driver locations geospatially and match riders to nearby drivers at scale.",
    },
    {
        "id": 8,
        "title": "Video Streaming Platform",
        "prompt": "Design a video streaming platform like YouTube, focusing on video upload, transcoding pipeline, and adaptive bitrate delivery via CDN.",
    },
    {
        "id": 9,
        "title": "Distributed Job Scheduler",
        "prompt": "Design a distributed cron/job scheduler that can run millions of scheduled jobs reliably across a cluster of worker nodes, avoiding duplicate execution and handling worker failures.",
    },
    {
        "id": 10,
        "title": "Online Donation Platform",
        "prompt": "Design the backend for a donation/crowdfunding platform (like Kitabisa) that must handle campaign creation, real-time donation totals under high concurrency, and prevent overcounting or race conditions on the donation counter.",
    },
]


def call_chat_completion(base_url: str, model: str, prompt: str, max_tokens: int, temperature: float, timeout: int):
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a senior software architect. Answer system design questions concisely but completely, covering key components, data flow, and trade-offs."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    elapsed = time.monotonic() - start
    return body, elapsed


def main():
    parser = argparse.ArgumentParser(description="Run 10 system design test cases against a vLLM server")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="vLLM OpenAI-compatible base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name as registered on the server")
    parser.add_argument("--max-tokens", type=int, default=800, help="Max completion tokens per case")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--timeout", type=int, default=180, help="Per-request timeout in seconds")
    parser.add_argument("--cases", default="", help="Comma-separated case IDs to run (default: all 10)")
    parser.add_argument("--output", default="vllm_system_design_results.json", help="Path to write JSON results")
    args = parser.parse_args()

    cases = SYSTEM_DESIGN_CASES
    if args.cases:
        wanted = {int(x) for x in args.cases.split(",")}
        cases = [c for c in SYSTEM_DESIGN_CASES if c["id"] in wanted]

    print(f"Target: {args.base_url}  model={args.model}")
    print(f"Running {len(cases)} case(s)\n")

    results = []
    ok_count = 0
    fail_count = 0

    for case in cases:
        print(f"[{case['id']:02d}] {case['title']} ... ", end="", flush=True)
        try:
            body, elapsed = call_chat_completion(
                args.base_url, args.model, case["prompt"], args.max_tokens, args.temperature, args.timeout
            )
            choice = body["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
            usage = body.get("usage", {})
            print(f"OK  ({elapsed:.1f}s, {usage.get('completion_tokens', '?')} tokens, finish={finish_reason})")
            results.append({
                "id": case["id"],
                "title": case["title"],
                "prompt": case["prompt"],
                "status": "ok",
                "elapsed_seconds": round(elapsed, 2),
                "finish_reason": finish_reason,
                "usage": usage,
                "response": content,
            })
            ok_count += 1
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            print(f"HTTP ERROR {e.code}")
            results.append({
                "id": case["id"],
                "title": case["title"],
                "prompt": case["prompt"],
                "status": "error",
                "error": f"HTTP {e.code}: {err_body}",
            })
            fail_count += 1
        except Exception as e:
            print(f"FAILED ({type(e).__name__}: {e})")
            results.append({
                "id": case["id"],
                "title": case["title"],
                "prompt": case["prompt"],
                "status": "error",
                "error": str(e),
            })
            fail_count += 1

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone: {ok_count} ok, {fail_count} failed. Full results written to {args.output}")

    if fail_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
