#!/usr/bin/env python3
import csv
import os
import json
from glob import glob
from collections import defaultdict

RESULTS_DIR = "/projects/F202500010HPCVLABUMINHO/uminhocp008/ADED/project2/results"
OUTPUT_FILE = "processed_results.json"

def extract_metrics(csv_file):
    """Extract TTFT, TPOT, throughput from a CSV file (averages over trials)"""
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        return None
    
    # Use the first row as sample (CSV already has means from benchmark_client.py)
    # Columns: ttft_mean_ms, tpot_mean_ms, throughput_mean_tok_s
    return {
        'ttft_mean_ms': float(rows[0]['ttft_mean_ms']),
        'tpot_mean_ms': float(rows[0]['tpot_mean_ms']),
        'throughput_mean_tok_s': float(rows[0]['throughput_mean_tok_s'])
    }

def parse_filename(filename):
    """Extract config from filename"""
    base = os.path.basename(filename).replace('.csv', '')
    parts = base.split('_')
    
    config = {}
    # Detect model
    if 'SmolLM2' in base:
        config['model'] = 'SmolLM2'
        # Extract quantisation (Q2_K, Q4_K_M, Q5_K_M, Q8_0)
        for q in ['Q2_K', 'Q4_K_M', 'Q5_K_M', 'Q8_0']:
            if q in base:
                config['quant'] = q
                break
        else:
            config['quant'] = 'Q4_K_M'
    elif 'qwen' in base.lower():
        config['model'] = 'Qwen'
        config['quant'] = 'Q4_K_M'
    elif 'Meta-Llama' in base:
        config['model'] = 'Llama-8B'
        config['quant'] = 'Q4_K_M'
    else:
        config['model'] = 'Unknown'
        config['quant'] = 'Unknown'
    
    # Extract threads
    for t in ['t4', 't16', 't32', 't1', 't2', 't8', 't48']:
        if t in base:
            config['threads'] = int(t[1:])
            break
    else:
        config['threads'] = 16
    
    # Extract context size
    for ctx in ['ctx128', 'ctx512', 'ctx1024', 'ctx2048']:
        if ctx in base:
            config['ctx_size'] = int(ctx[3:])
            break
    else:
        config['ctx_size'] = 2048
    
    return config

def main():
    all_results = []
    csv_files = glob(f"{RESULTS_DIR}/*.csv")
    
    print(f"Found {len(csv_files)} CSV files")
    
    for csv_file in csv_files:
        config = parse_filename(csv_file)
        metrics = extract_metrics(csv_file)
        if metrics:
            result = {**config, **metrics, 'file': os.path.basename(csv_file)}
            all_results.append(result)
    
    # Group by configuration type
    baseline = [r for r in all_results if r['threads'] == 16 and r['ctx_size'] == 2048 and r['quant'] == 'Q4_K_M']
    quantisation = [r for r in all_results if r['model'] == 'SmolLM2' and r['threads'] == 16]
    threading = [r for r in all_results if r['ctx_size'] == 2048 and r['quant'] == 'Q4_K_M']
    context = [r for r in all_results if r['threads'] == 16 and r['quant'] == 'Q4_K_M']
    
    # Print tables
    print("\n" + "="*60)
    print("BASELINE (3 models, threads=16, ctx=2048, Q4_K_M)")
    print("="*60)
    print(f"{'Model':<12} {'TTFT (ms)':<12} {'TPOT (ms)':<12} {'Throughput (tok/s)':<20}")
    print("-"*60)
    for r in baseline:
        print(f"{r['model']:<12} {r['ttft_mean_ms']:<12.2f} {r['tpot_mean_ms']:<12.2f} {r['throughput_mean_tok_s']:<20.1f}")
    
    print("\n" + "="*60)
    print("QUANTISATION (SmolLM2, threads=16)")
    print("="*60)
    print(f"{'Quant':<8} {'TPOT (ms)':<12} {'Throughput (tok/s)':<20}")
    print("-"*40)
    for q in ['Q2_K', 'Q4_K_M', 'Q5_K_M', 'Q8_0']:
        r = next((x for x in quantisation if x.get('quant') == q), None)
        if r:
            print(f"{q:<8} {r['tpot_mean_ms']:<12.2f} {r['throughput_mean_tok_s']:<20.1f}")
    
    print("\n" + "="*60)
    print("THREADING (3 models, ctx=2048, Q4_K_M)")
    print("="*60)
    for model in ['SmolLM2', 'Qwen', 'Llama-8B']:
        print(f"\n{model}:")
        print(f"{'Threads':<8} {'TPOT (ms)':<12} {'Throughput (tok/s)':<20}")
        for t in [4, 16, 32]:
            r = next((x for x in threading if x['model'] == model and x['threads'] == t), None)
            if r:
                print(f"{t:<8} {r['tpot_mean_ms']:<12.2f} {r['throughput_mean_tok_s']:<20.1f}")
    
    # Save JSON
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()