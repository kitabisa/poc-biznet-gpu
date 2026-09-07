#!/usr/bin/env python3
"""Benchmark vLLM under concurrent load: GPU VRAM/util, aggregate throughput, and
per-request latency, at increasing levels of concurrency.

For each concurrency level N, fires N requests at once (one per test case, cycling
through the case list if N > number of cases) using a thread pool, while a
background thread samples nvidia-smi on the remote GPU host over SSH.
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
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    def __init__(self, ssh_host, ssh_key, interval=1.0):
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


def run_one(case, base_url, model, max_tokens, temperature, timeout):
    try:
        body, elapsed = call_chat_completion(base_url, model, case["prompt"], max_tokens, temperature, timeout)
        choice = body["choices"][0]
        usage = body.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        return {
            "case_id": case["id"],
            "title": case["title"],
            "status": "ok",
            "elapsed_seconds": round(elapsed, 2),
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": completion_tokens,
            "tokens_per_second_individual": round(completion_tokens / elapsed, 1) if elapsed > 0 else None,
        }
    except urllib.error.HTTPError as e:
        return {"case_id": case["id"], "title": case["title"], "status": "error", "error": f"HTTP {e.code}"}
    except Exception as e:
        return {"case_id": case["id"], "title": case["title"], "status": "error", "error": str(e)}


def run_concurrency_level(level, base_url, model, max_tokens, temperature, timeout, gpu_sampler):
    requests_to_fire = [SYSTEM_DESIGN_CASES[i % len(SYSTEM_DESIGN_CASES)] for i in range(level)]

    if gpu_sampler:
        gpu_sampler.start()

    batch_start = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=level) as pool:
        futures = [pool.submit(run_one, c, base_url, model, max_tokens, temperature, timeout) for c in requests_to_fire]
        for fut in as_completed(futures):
            results.append(fut.result())
    batch_elapsed = time.monotonic() - batch_start

    gpu_samples = gpu_sampler.stop_and_collect() if gpu_sampler else []
    gpu_summary = GpuSampler.summarize(gpu_samples) if gpu_sampler else None

    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] != "ok"]
    total_completion_tokens = sum(r["completion_tokens"] for r in ok)
    aggregate_tps = round(total_completion_tokens / batch_elapsed, 1) if batch_elapsed > 0 else None

    return {
        "concurrency": level,
        "requests_fired": level,
        "ok": len(ok),
        "failed": len(failed),
        "batch_wall_time_seconds": round(batch_elapsed, 2),
        "total_completion_tokens": total_completion_tokens,
        "aggregate_tokens_per_second": aggregate_tps,
        "per_request_latency_avg_s": round(statistics.mean([r["elapsed_seconds"] for r in ok]), 2) if ok else None,
        "per_request_latency_min_s": round(min([r["elapsed_seconds"] for r in ok]), 2) if ok else None,
        "per_request_latency_max_s": round(max([r["elapsed_seconds"] for r in ok]), 2) if ok else None,
        "gpu": gpu_summary,
        "requests": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM under concurrent load")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ssh-host", default=DEFAULT_SSH_HOST)
    parser.add_argument("--ssh-key", default=DEFAULT_SSH_KEY)
    parser.add_argument("--no-gpu-monitor", action="store_true")
    parser.add_argument("--gpu-poll-interval", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1500, help="Keep modest so concurrency sweep finishes in reasonable time")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--levels", default="1,2,4,8,16", help="Comma-separated concurrency levels to test")
    parser.add_argument("--output", default="benchmark_concurrency_results.json")
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",")]

    print(f"Target: {args.base_url}  model={args.model}")
    print(f"GPU monitoring: {'disabled' if args.no_gpu_monitor else args.ssh_host}")
    print(f"Concurrency levels: {levels}, max_tokens={args.max_tokens}\n")

    gpu_sampler = None
    if not args.no_gpu_monitor:
        gpu_sampler = GpuSampler(args.ssh_host, args.ssh_key, args.gpu_poll_interval)
        idle_sample = gpu_sampler._query()
        if idle_sample is None:
            print("WARNING: could not reach GPU host via SSH; disabling GPU monitoring.\n")
            gpu_sampler = None
        else:
            print(f"Idle VRAM before benchmark: {idle_sample['mem_used_mib']:.0f} / {idle_sample['mem_total_mib']:.0f} MiB\n")

    level_results = []
    for level in levels:
        print(f"--- Concurrency level: {level} ---")
        lvl = run_concurrency_level(level, args.base_url, args.model, args.max_tokens, args.temperature, args.timeout, gpu_sampler)
        gpu_str = ""
        if lvl["gpu"] and lvl["gpu"]["count"]:
            gpu_str = f", VRAM peak {lvl['gpu']['mem_used_mib_max']:.0f}MiB, util avg {lvl['gpu']['gpu_util_pct_avg']:.0f}%"
        print(
            f"  ok={lvl['ok']}/{lvl['requests_fired']}  batch_time={lvl['batch_wall_time_seconds']}s  "
            f"aggregate_tok/s={lvl['aggregate_tokens_per_second']}  "
            f"avg_latency={lvl['per_request_latency_avg_s']}s{gpu_str}\n"
        )
        level_results.append(lvl)

    print("=== Concurrency Sweep Summary ===")
    print(f"{'Level':>6} {'OK':>5} {'BatchTime(s)':>13} {'AggTok/s':>10} {'AvgLatency(s)':>14} {'VRAMPeak(MiB)':>14}")
    for lvl in level_results:
        vram_peak = lvl["gpu"]["mem_used_mib_max"] if lvl["gpu"] and lvl["gpu"]["count"] else None
        print(f"{lvl['concurrency']:>6} {lvl['ok']:>5} {lvl['batch_wall_time_seconds']:>13} {lvl['aggregate_tokens_per_second']:>10} {lvl['per_request_latency_avg_s']:>14} {vram_peak if vram_peak else 'n/a':>14}")

    with open(args.output, "w") as f:
        json.dump({"levels": level_results}, f, indent=2)
    print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    main()
