import os
import matplotlib.pyplot as plt
import numpy as np
from model import RumorModel

os.makedirs("plots", exist_ok=True)

SEED = 42
N = 100
STEPS = 80
BETA = 0.10   # base spreading probability, from Vosoughi et al. (2018)
GAMMA = 0.05  # recovery rate

# Run simulation on both network types
model_ba = RumorModel(n_agents=N, beta=BETA, gamma=GAMMA, network_type="BA", seed=SEED)
data_ba = model_ba.run(steps=STEPS)

model_ws = RumorModel(n_agents=N, beta=BETA, gamma=GAMMA, network_type="WS", seed=SEED)
data_ws = model_ws.run(steps=STEPS)


# Plot full SIR curves for BA network
plt.figure(figsize=(8, 5))
plt.plot(data_ba["S"], label="Susceptible", color="steelblue")
plt.plot(data_ba["I"], label="Infected (spreading)", color="firebrick")
plt.plot(data_ba["R"], label="Recovered (stopped)", color="seagreen")
plt.xlabel("Time (days)")
plt.ylabel("Number of users (out of 100)")
plt.title("SIR Rumor Spreading — Barabási-Albert Network (β=0.10, γ=0.05)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/sir_BA.png", dpi=150)
plt.show()
plt.close()

# Plot full SIR curves for WS network
plt.figure(figsize=(8, 5))
plt.plot(data_ws["S"], label="Susceptible", color="steelblue")
plt.plot(data_ws["I"], label="Infected (spreading)", color="firebrick")
plt.plot(data_ws["R"], label="Recovered (stopped)", color="seagreen")
plt.xlabel("Time (days)")
plt.ylabel("Number of users (out of 100)")
plt.title("SIR Rumor Spreading — Watts-Strogatz Network (β=0.10, γ=0.05)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/sir_WS.png", dpi=150)
plt.show()
plt.close()

# Compare infected curves from networks on the same graph
plt.figure(figsize=(8, 5))
plt.plot(data_ba["I"], label="Barabási-Albert", color="firebrick")
plt.plot(data_ws["I"], label="Watts-Strogatz", color="steelblue")
plt.xlabel("Time (days)")
plt.ylabel("Users actively spreading the rumor (out of 100)")
plt.title("Infected Agents Over Time — BA vs WS (β=0.10, γ=0.05)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/infected_comparison.png", dpi=150)
plt.show()
plt.close()

# Bar chart comparing key metrics for each network
metrics = {
    "BA": {
        "Peak I":       data_ba["I"].max(),
        "Step of peak": data_ba["I"].idxmax(),
        "Final spread": data_ba["R"].iloc[-1],
    },
    "WS": {
        "Peak I":       data_ws["I"].max(),
        "Step of peak": data_ws["I"].idxmax(),
        "Final spread": data_ws["R"].iloc[-1],
    },
}

labels = ["Peak infected\n(users)", "Time to peak\n(days)", "Final spread\n(users)"]
ba_vals = [metrics["BA"]["Peak I"], metrics["BA"]["Step of peak"], metrics["BA"]["Final spread"]]
ws_vals = [metrics["WS"]["Peak I"], metrics["WS"]["Step of peak"], metrics["WS"]["Final spread"]]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, ba_vals, width, label="Barabási-Albert", color="firebrick", alpha=0.85)
ax.bar(x + width/2, ws_vals, width, label="Watts-Strogatz",  color="steelblue", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Value")
ax.set_title("Summary Metrics — BA vs WS (N=100, β=0.10, γ=0.05)")
ax.legend()
plt.tight_layout()
plt.savefig("plots/summary_metrics.png", dpi=150)
plt.show()
plt.close()

print("\n--- Summary Metrics ---")
print(f"{'Metric':<20} {'BA':>8} {'WS':>8}")
print("-" * 38)
for key in ["Peak I", "Step of peak", "Final spread"]:
    print(f"{key:<20} {metrics['BA'][key]:>8} {metrics['WS'][key]:>8}")
