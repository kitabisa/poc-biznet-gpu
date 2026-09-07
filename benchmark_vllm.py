#!/usr/bin/env python3
"""Benchmark vLLM inference: GPU VRAM usage, token spend, and time to complete.

Runs each system design test case against the vLLM OpenAI-compatible API while
sampling GPU memory/utilization on the remote host (via SSH + nvidia-smi) in a
background thread, then reports per-case and aggregate stats.
"""

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "http://YOUR_VLLM_HOST:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_SSH_HOST = "user@YOUR_VLLM_HOST"
DEFAULT_SSH_KEY = "~/.ssh/your_key.pem"

SYSTEM_DESIGN_CASES = [
    {"id": 1, "title": "URL Shortener", "prompt": "Design a URL shortening service like bit.ly. Cover the API, ID generation strategy, data model, and how you would handle 50,000 redirects per second at read time."},
    {"id": 2, "title": "Rate Limiter", "prompt": "Design a distributed rate limiter that can be used across multiple API gateway nodes. Compare token bucket vs sliding window log, and explain how you would keep counters consistent across nodes."},
    {"id": 3, "title": "News Feed System", "prompt": "Design a social media news feed (like Twitter/X home timeline). Discuss fan-out-on-write vs fan-out-on-read, and how you would handle celebrity accounts with millions of followers."},
    {"id": 4, "title": "Distributed Cache", "prompt": "Design a distributed in-memory cache system similar to Redis Cluster. Explain sharding strategy, replication, and cache invalidation approach."},
    {"id": 5, "title": "Chat Application", "prompt": "Design a real-time one-on-one and group chat application like WhatsApp. Cover message delivery guarantees, offline message storage, and read receipts."},
    {"id": 6, "title": "Payment Processing System", "prompt": "Design a payment processing system that must guarantee exactly-once charge semantics even under network retries. Discuss idempotency keys, double-entry ledger design, and reconciliation."},
    {"id": 7, "title": "Ride-Hailing Dispatch", "prompt": "Design the driver-matching component of a ride-hailing app like Gojek/Grab. Explain how you'd index driver locations geospatially and match riders to nearby drivers at scale."},
    {"id": 8, "title": "Video Streaming Platform", "prompt": "Design a video streaming platform like YouTube, focusing on video upload, transcoding pipeline, and adaptive bitrate delivery via CDN."},
    {"id": 9, "title": "Distributed Job Scheduler", "prompt": "Design a distributed cron/job scheduler that can run millions of scheduled jobs reliably across a cluster of worker nodes, avoiding duplicate execution and handling worker failures."},
    {"id": 10, "title": "Online Donation Platform", "prompt": "Design the backend for a donation/crowdfunding platform (like Kitabisa) that must handle campaign creation, real-time donation totals under high concurrency, and prevent overcounting or race conditions on the donation counter."},
]


class GpuSampler:
    """Polls `nvidia-smi` on a remote host over SSH at a fixed interval in a background thread."""

    def __init__(self, ssh_host: str, ssh_key: str, interval: float = 1.0):
        self.ssh_host = ssh_host
        self.ssh_key = ssh_key
        self.interval = interval
        self._samples = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def _query(self):
        cmd = [
            "ssh", "-i", self.ssh_key, "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=5", self.ssh_host,
            "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,power.draw "
            "--format=csv,noheader,nounits",
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            line = out.stdout.strip().splitlines()[0]
            mem_used, mem_total, util, power = [x.strip() for x in line.split(",")]
            return {
                "t": time.monotonic(),
                "mem_used_mib": float(mem_used),
                "mem_total_mib": float(mem_total),
                "gpu_util_pct": float(util),
                "power_draw_w": float(power),
            }
        except Exception:
            return None

    def _run(self):
        while not self._stop.is_set():
            sample = self._query()
            if sample:
                with self._lock:
                    self._samples.append(sample)
            self._stop.wait(self.interval)

    def start(self):
        self._samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop_and_collect(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        with self._lock:
            return list(self._samples)

    @staticmethod
    def summarize(samples):
        if not samples:
            return {"count": 0}
        mem = [s["mem_used_mib"] for s in samples]
        util = [s["gpu_util_pct"] for s in samples]
        power = [s["power_draw_w"] for s in samples]
        return {
            "count": len(samples),
            "mem_used_mib_min": min(mem),
            "mem_used_mib_max": max(mem),
            "mem_used_mib_avg": round(statistics.mean(mem), 1),
            "mem_total_mib": samples[0]["mem_total_mib"],
            "gpu_util_pct_min": min(util),
            "gpu_util_pct_max": max(util),
            "gpu_util_pct_avg": round(statistics.mean(util), 1),
            "power_draw_w_avg": round(statistics.mean(power), 1),
            "power_draw_w_max": max(power),
        }


def call_chat_completion(base_url, model, prompt, max_tokens, temperature, timeout):
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
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    elapsed = time.monotonic() - start
    return body, elapsed


def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM: GPU VRAM, token spend, time to complete")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ssh-host", default=DEFAULT_SSH_HOST, help="user@host for GPU SSH monitoring")
    parser.add_argument("--ssh-key", default=DEFAULT_SSH_KEY)
    parser.add_argument("--no-gpu-monitor", action="store_true", help="Skip SSH-based GPU sampling")
    parser.add_argument("--gpu-poll-interval", type=float, default=1.0, help="Seconds between nvidia-smi polls")
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--cases", default="", help="Comma-separated case IDs (default: all 10)")
    parser.add_argument("--output", default="benchmark_results.json")
    args = parser.parse_args()

    cases = SYSTEM_DESIGN_CASES
    if args.cases:
        wanted = {int(x) for x in args.cases.split(",")}
        cases = [c for c in SYSTEM_DESIGN_CASES if c["id"] in wanted]

    print(f"Target: {args.base_url}  model={args.model}")
    print(f"GPU monitoring: {'disabled' if args.no_gpu_monitor else args.ssh_host}")
    print(f"Running {len(cases)} case(s), max_tokens={args.max_tokens}\n")

    gpu_sampler = None
    idle_gpu_summary = None
    if not args.no_gpu_monitor:
        gpu_sampler = GpuSampler(args.ssh_host, args.ssh_key, args.gpu_poll_interval)
        idle_sample = gpu_sampler._query()
        if idle_sample is None:
            print("WARNING: could not reach GPU host via SSH; disabling GPU monitoring.\n")
            gpu_sampler = None
        else:
            idle_gpu_summary = idle_sample
            print(f"Idle VRAM before benchmark: {idle_sample['mem_used_mib']:.0f} / {idle_sample['mem_total_mib']:.0f} MiB\n")

    results = []
    ok_count = 0
    fail_count = 0
    bench_start = time.monotonic()

    for case in cases:
        print(f"[{case['id']:02d}] {case['title']} ... ", end="", flush=True)

        if gpu_sampler:
            gpu_sampler.start()

        try:
            body, elapsed = call_chat_completion(
                args.base_url, args.model, case["prompt"], args.max_tokens, args.temperature, args.timeout
            )
            gpu_samples = gpu_sampler.stop_and_collect() if gpu_sampler else []
            gpu_summary = GpuSampler.summarize(gpu_samples) if gpu_sampler else None

            choice = body["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
            usage = body.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            prompt_tokens = usage.get("prompt_tokens", 0)
            tok_per_sec = round(completion_tokens / elapsed, 1) if elapsed > 0 else None

            gpu_str = ""
            if gpu_summary and gpu_summary["count"]:
                gpu_str = f", VRAM peak {gpu_summary['mem_used_mib_max']:.0f}MiB, util avg {gpu_summary['gpu_util_pct_avg']:.0f}%"
            print(f"OK  ({elapsed:.1f}s, {completion_tokens} tok, {tok_per_sec} tok/s, finish={finish_reason}{gpu_str})")

            results.append({
                "id": case["id"],
                "title": case["title"],
                "status": "ok",
                "elapsed_seconds": round(elapsed, 2),
                "finish_reason": finish_reason,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": usage.get("total_tokens"),
                "tokens_per_second": tok_per_sec,
                "gpu": gpu_summary,
                "response_chars": len(content),
            })
            ok_count += 1
        except urllib.error.HTTPError as e:
            if gpu_sampler:
                gpu_sampler.stop_and_collect()
            err_body = e.read().decode("utf-8", errors="replace")
            print(f"HTTP ERROR {e.code}")
            results.append({"id": case["id"], "title": case["title"], "status": "error", "error": f"HTTP {e.code}: {err_body}"})
            fail_count += 1
        except Exception as e:
            if gpu_sampler:
                gpu_sampler.stop_and_collect()
            print(f"FAILED ({type(e).__name__}: {e})")
            results.append({"id": case["id"], "title": case["title"], "status": "error", "error": str(e)})
            fail_count += 1

    total_elapsed = time.monotonic() - bench_start

    ok_results = [r for r in results if r["status"] == "ok"]
    summary = {
        "cases_run": len(cases),
        "ok": ok_count,
        "failed": fail_count,
        "total_wall_time_seconds": round(total_elapsed, 2),
        "idle_vram_mib": idle_gpu_summary["mem_used_mib"] if idle_gpu_summary else None,
        "idle_vram_total_mib": idle_gpu_summary["mem_total_mib"] if idle_gpu_summary else None,
    }
    if ok_results:
        elapsed_list = [r["elapsed_seconds"] for r in ok_results]
        tok_list = [r["completion_tokens"] for r in ok_results]
        tps_list = [r["tokens_per_second"] for r in ok_results if r["tokens_per_second"]]
        summary.update({
            "time_per_case_avg_s": round(statistics.mean(elapsed_list), 2),
            "time_per_case_min_s": round(min(elapsed_list), 2),
            "time_per_case_max_s": round(max(elapsed_list), 2),
            "completion_tokens_total": sum(tok_list),
            "completion_tokens_avg": round(statistics.mean(tok_list), 1),
            "tokens_per_second_avg": round(statistics.mean(tps_list), 1) if tps_list else None,
            "finish_stop_count": sum(1 for r in ok_results if r["finish_reason"] == "stop"),
            "finish_length_count": sum(1 for r in ok_results if r["finish_reason"] == "length"),
        })
        peak_vrams = [r["gpu"]["mem_used_mib_max"] for r in ok_results if r.get("gpu") and r["gpu"]["count"]]
        if peak_vrams:
            summary["vram_peak_mib_overall"] = max(peak_vrams)
            summary["vram_peak_mib_avg_per_case"] = round(statistics.mean(peak_vrams), 1)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))

    output = {"summary": summary, "cases": results}
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results written to {args.output}")

    if fail_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
