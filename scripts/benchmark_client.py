import argparse
import json
import time
import csv
import requests
import statistics
from datetime import datetime


def send_prompt(host, port, prompt, n_predict):
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n_predict,
        "stream": True
    }

    ttft = None
    token_times = []
    generated_text = ""
    start_time = time.perf_counter()

    try:
        with requests.post(url, json=payload, stream=True, timeout=300) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    line = line[6:]
                if line.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                now = time.perf_counter()

                if ttft is None:
                    ttft = now - start_time

                token_times.append(now)

                # OpenAI-compatible format
                choices = data.get("choices", [])
                if choices:
                    content = choices[0].get("delta", {}).get("content") or ""
                    generated_text += content
                    if choices[0].get("finish_reason") is not None:
                        break

    except Exception as e:
        print(f"  ERROR: {e}")
        return None

    end_time = time.perf_counter()
    total_time = end_time - start_time
    n_tokens = len(token_times)

    if n_tokens > 1:
        gaps = [token_times[i] - token_times[i-1] for i in range(1, n_tokens)]
        tpot = statistics.mean(gaps) * 1000
    else:
        tpot = 0.0

    throughput = n_tokens / total_time if total_time > 0 else 0.0

    return {
        "ttft_ms": round(ttft * 1000, 2) if ttft else 0.0,
        "tpot_ms": round(tpot, 2),
        "throughput_tok_s": round(throughput, 2),
        "total_time_s": round(total_time, 2),
        "n_tokens": n_tokens,
        "generated_text": generated_text[:200]
    }


def run_benchmark(host, port, prompts_file, n_predict, trials, output):
    with open(prompts_file) as f:
        prompts = json.load(f)

    all_prompts = []
    for category, prompt_list in prompts.items():
        for prompt in prompt_list:
            all_prompts.append((category, prompt))

    print(f"Loaded {len(all_prompts)} prompts ({trials} trials each)")
    print(f"Output: {output}")

    results = []

    for i, (category, prompt) in enumerate(all_prompts):
        print(f"[{i+1}/{len(all_prompts)}] Category: {category} | Prompt: {prompt[:50]}...")

        trial_results = []
        for trial in range(trials):
            print(f"  Trial {trial+1}/{trials}...", end=" ", flush=True)
            result = send_prompt(host, port, prompt, n_predict)
            if result:
                trial_results.append(result)
                print(f"TTFT={result['ttft_ms']}ms TPOT={result['tpot_ms']}ms {result['throughput_tok_s']}tok/s")
            else:
                print("FAILED")
            time.sleep(0.5)

        if trial_results:
            ttfts = [r["ttft_ms"] for r in trial_results]
            tpots = [r["tpot_ms"] for r in trial_results]
            throughputs = [r["throughput_tok_s"] for r in trial_results]

            results.append({
                "timestamp": datetime.now().isoformat(),
                "category": category,
                "prompt": prompt[:100],
                "trials": len(trial_results),
                "ttft_mean_ms": round(statistics.mean(ttfts), 2),
                "ttft_std_ms": round(statistics.stdev(ttfts) if len(ttfts) > 1 else 0.0, 2),
                "tpot_mean_ms": round(statistics.mean(tpots), 2),
                "tpot_std_ms": round(statistics.stdev(tpots) if len(tpots) > 1 else 0.0, 2),
                "throughput_mean_tok_s": round(statistics.mean(throughputs), 2),
                "throughput_std_tok_s": round(statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0, 2),
                "generated_text": trial_results[0]["generated_text"]
            })

    if results:
        fieldnames = results[0].keys()
        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n=== Done! Results saved to {output} ===")
    else:
        print("No results to save.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="llama.cpp benchmark client")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default="8080")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--n-predict", type=int, default=256)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_benchmark(
        host=args.host,
        port=args.port,
        prompts_file=args.prompts,
        n_predict=args.n_predict,
        trials=args.trials,
        output=args.output
    )
