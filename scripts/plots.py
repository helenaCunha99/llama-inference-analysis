import matplotlib.pyplot as plt
import numpy as np

# ============================================
# PLOT 1: Throughput vs Threads
# ============================================
threads = [4, 16, 32]
thru_smol = [51.3, 101.2, 69.7]
thru_qwen = [22.8, 31.8, 32.2]
thru_llama = [6.5, 10.9, 8.9]

plt.figure(figsize=(6, 4))
plt.plot(threads, thru_smol, 'o-', label='SmolLM2', linewidth=2, markersize=8, color='blue')
plt.plot(threads, thru_qwen, 's-', label='Qwen', linewidth=2, markersize=8, color='orange')
plt.plot(threads, thru_llama, '^-', label='Llama-8B', linewidth=2, markersize=8, color='green')
plt.xlabel('Number of Threads', fontsize=12)
plt.ylabel('Throughput (tok/s)', fontsize=12)
plt.title('Throughput vs Thread Count', fontsize=14)
plt.legend(fontsize=10, loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(threads, ['4', '16', '32'])
plt.tight_layout()
plt.savefig('plot_throughput_vs_threads.png', dpi=150)
plt.savefig('plot_throughput_vs_threads.pdf')
print("✓ Plot 1 saved: plot_throughput_vs_threads.png/pdf")

# ============================================
# PLOT 2: TTFT vs Context Length (com escala log)
# ============================================
ctx = [128, 512, 1024, 2048]
ttft_smol = [215.9, 216.8, 215.5, 228.3]
ttft_qwen = [476.5, 449.5, 451.6, 458.1]
ttft_llama = [2034.2, 2003.7, 2012.2, 2033.8]

plt.figure(figsize=(6, 4))
plt.plot(ctx, ttft_smol, 'o-', label='SmolLM2', linewidth=2, markersize=8, color='blue')
plt.plot(ctx, ttft_qwen, 's-', label='Qwen', linewidth=2, markersize=8, color='orange')
plt.plot(ctx, ttft_llama, '^-', label='Llama-8B', linewidth=2, markersize=8, color='green')
plt.xlabel('Context Length (tokens)', fontsize=12)
plt.ylabel('TTFT (ms) - Log Scale', fontsize=12)
plt.title('TTFT vs Context Length (Cold Cache)', fontsize=12)
plt.yscale('log')
plt.legend(fontsize=10, loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(ctx, ['128', '512', '1024', '2048'])
plt.tight_layout()
plt.savefig('plot_ttft_vs_context.png', dpi=150)
plt.savefig('plot_ttft_vs_context.pdf')
print("✓ Plot 2 saved: plot_ttft_vs_context.png/pdf")

# ============================================
# PLOT 3: Memory vs Quantisation
# ============================================
quant = ['Q2_K', 'Q4_K_M', 'Q5_K_M', 'Q8_0']
memory_gb = [0.219, 0.259, 0.290, 0.386]
throughput = [111.0, 80.2, 102.9, 77.8]

fig, ax1 = plt.subplots(figsize=(6, 4))
bars = ax1.bar(quant, memory_gb, color='steelblue', alpha=0.7, label='Model Size (GB)')
ax1.set_xlabel('Quantisation Level', fontsize=12)
ax1.set_ylabel('Model Size (GB)', fontsize=12, color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')

ax2 = ax1.twinx()
ax2.plot(quant, throughput, 'o-', color='darkorange', linewidth=2, markersize=8, label='Throughput (tok/s)')
ax2.set_ylabel('Throughput (tok/s)', fontsize=12, color='darkorange')
ax2.tick_params(axis='y', labelcolor='darkorange')

plt.title('Memory vs Quantisation (SmolLM2)', fontsize=14)
fig.tight_layout()
plt.savefig('plot_memory_vs_quantisation.png', dpi=150)
plt.savefig('plot_memory_vs_quantisation.pdf')
print("✓ Plot 3 saved: plot_memory_vs_quantisation.png/pdf")

# ============================================
# PLOT 4: Predicted vs Observed TPOT
# ============================================
models_names = ['SmolLM2', 'Qwen', 'Llama-8B']
tpot_obs = [2.8, 31.7, 92.1]
tpot_pred = [7.7, 28.5, 92.3]

plt.figure(figsize=(5.5, 5.5))
plt.scatter(tpot_obs, tpot_pred, color='red', s=120, zorder=5)
for i, name in enumerate(models_names):
    plt.annotate(name, (tpot_obs[i], tpot_pred[i]), 
                 xytext=(10, 10), textcoords='offset points', 
                 fontsize=11, fontweight='bold', ha='center', va='center')
max_val = max(max(tpot_obs), max(tpot_pred)) + 15
plt.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, label='Perfect prediction (y=x)')
plt.xlabel('Observed TPOT (ms)', fontsize=12)
plt.ylabel('Predicted TPOT (ms)', fontsize=12)
plt.title('Performance Model Validation', fontsize=14)
plt.legend(fontsize=10, loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, 110)
plt.ylim(0, 110)
plt.tight_layout()
plt.savefig('plot_predicted_vs_observed.png', dpi=150)
plt.savefig('plot_predicted_vs_observed.pdf')
print("✓ Plot 4 saved: plot_predicted_vs_observed.png/pdf")

print("\nAll 4 plots generated successfully!")
