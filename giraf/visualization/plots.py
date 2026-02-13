"""
Main visualization functions for GIRAF simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def extended_visualize_results(time_span, aggregate_risks, epistemic_risks, staleness_risks,
                               congestion_index, jitter, bt_true, bt_reported,
                               ping_violations, jitter_violations, fraud_detected):
    """
    Generate comprehensive visualization of GIRAF simulation results.
    
    Args:
        time_span: Time array
        aggregate_risks: Aggregate risk values
        epistemic_risks: Epistemic risk component
        staleness_risks: Staleness risk component
        congestion_index: Traffic congestion data
        jitter: Network jitter data
        bt_true: Ground truth confidence
        bt_reported: Reported confidence
        ping_violations: Ping violation flags
        jitter_violations: Jitter violation flags
        fraud_detected: Fraud detection flags
    """
    min_len = min(len(time_span), len(aggregate_risks), len(epistemic_risks), len(staleness_risks))
    t = time_span[:min_len]
    
    fig, axes = plt.subplots(6, 1, figsize=(12, 22), sharex=True)
    
    # Subplot 1: Aggregate Risk
    axes[0].plot(t, aggregate_risks[:min_len], color='black', label=r'Aggregate Risk ($R_t$)')
    axes[0].axhline(y=45, color='red', linestyle='--', label='Mitigation Threshold')
    axes[0].set_yscale('symlog', linthresh=100)
    axes[0].set_ylabel(r'Risk Index ($R_t$) [SymLog]')
    axes[0].set_title("GIRAF Governance: Risk Indexing")
    axes[0].legend(loc='upper right')
    
    # Subplot 2: Environmental Context
    ax_env = axes[1]
    ax_env.plot(t, congestion_index[:min_len], label='Traffic Jam Factor', color='brown', alpha=0.6)
    ax_env.set_ylabel('Congestion Index', color='brown')
    
    ax_jitter = ax_env.twinx()
    ax_jitter.plot(t, jitter[:min_len], label='Jitter (ms)', color='orange', linewidth=1.5)
    ax_jitter.set_ylabel('Jitter (ms)', color='orange')
    ax_jitter.set_ylim(0, max(jitter) * 1.2 if any(jitter) else 10)
    
    # Subplot 3: Risk Component Decomposition
    ax3 = axes[2]
    ax3.plot(t, epistemic_risks[:min_len], label='Epistemic (LLM)', color='blue', linewidth=1.5)
    ax3.set_ylabel('Epistemic Risk', color='blue')
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(t, staleness_risks[:min_len], label='Staleness (Latency)', color='green', alpha=0.5)
    ax3_twin.set_ylabel('Staleness Risk', color='green')
    ax3_twin.set_yscale('log')
    
    ax3.set_title("Risk Component Decomposition (Dual-Scaled)")
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc='upper left')
    
    # Subplot 4: Binary Incident Flags
    axes[3].step(t, ping_violations[:min_len], label='Ping Violation', where='post', color='red', alpha=0.5)
    axes[3].step(t, fraud_detected[:min_len], label='Fraud Detected', where='post', color='darkred', linewidth=2)
    axes[3].set_title("Binary Incident Flags")
    axes[3].legend()
    
    # Subplot 5: Agentic Confidence Alignment
    min_len_n = min(len(time_span), len(bt_true), len(bt_reported))
    t_plot = time_span[:min_len_n]
    
    axes[4].plot(t_plot, bt_true[:min_len_n], 'k--', label=r'Ground Truth ($B_T$)', alpha=0.8)
    axes[4].plot(t_plot, bt_reported[:min_len_n], 'g-', label=r'Reported ($B_R$)')
    axes[4].set_ylim(-0.05, 1.05)
    axes[4].set_ylabel("Confidence [0,1]")
    axes[4].set_title("Agentic Confidence Alignment")
    axes[4].legend(loc='upper right')
    
    # Subplot 6: Epistemic Dissonance
    bt_t = np.array(bt_true)
    bt_r = np.array(bt_reported)
    min_len_n = min(len(time_span), len(bt_t), len(bt_r))
    t = np.array(time_span)[:min_len_n]
    gap = np.abs(bt_t[:min_len_n] - bt_r[:min_len_n])
    
    axes[5].fill_between(t, gap, color='purple', alpha=0.3, label=r'Confidence Gap ($r_{epi}$)')
    axes[5].axhline(y=0.15, color='red', linestyle='--', label=r'Trust Boundary ($\Omega$)')
    axes[5].set_ylim(0, 1.0)
    axes[5].set_ylabel('Error')
    axes[5].legend()
    
    plt.tight_layout()
    plt.savefig("GIRAF_Simulation_Results.pdf", format='pdf', bbox_inches='tight', dpi=300)
    print("Plot successfully saved as GIRAF_Simulation_Results.pdf")
    plt.show()


def plot_risk_distribution_by_traffic_jam_factor(congestion_data, epistemic_risks, staleness_risks):
    """
    Visualize risk distributions by congestion level.
    
    Args:
        congestion_data: Traffic jam factor data
        epistemic_risks: Epistemic risk values
        staleness_risks: Staleness risk values
    """
    min_len = min(len(congestion_data), len(epistemic_risks), len(staleness_risks))
    
    if min_len == 0:
        print("Warning: No data available for risk distribution plot.")
        return
    
    tj = np.array(congestion_data[:min_len])
    er = np.array(epistemic_risks[:min_len])
    sr = np.array(staleness_risks[:min_len])
    
    valid_mask = ~(np.isnan(tj) | np.isnan(er) | np.isnan(sr))
    tj = tj[valid_mask]
    er = er[valid_mask]
    sr = sr[valid_mask]
    
    if len(tj) == 0:
        print("Warning: All data contains NaN values. Cannot plot.")
        return
    
    low = tj <= 3
    med = (tj > 3) & (tj <= 7)
    high = tj > 7
    
    if not (np.any(low) or np.any(med) or np.any(high)):
        print("Warning: No data falls into congestion categories.")
        return
    
    epi_means = [
        np.mean(er[low]) if np.any(low) and np.sum(low) > 0 else 0,
        np.mean(er[med]) if np.any(med) and np.sum(med) > 0 else 0,
        np.mean(er[high]) if np.any(high) and np.sum(high) > 0 else 0
    ]
    
    stal_means = [
        np.mean(sr[low]) if np.any(low) and np.sum(low) > 0 else 1,
        np.mean(sr[med]) if np.any(med) and np.sum(med) > 0 else 1,
        np.mean(sr[high]) if np.any(high) and np.sum(high) > 0 else 1
    ]
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    categories = ['Low\n(≤3)', 'Medium\n(3-7)', 'High\n(>7)']
    x = np.arange(len(categories))
    width = 0.35
    
    color1 = 'purple'
    bars1 = ax1.bar(x - width/2, epi_means, width, label='Epistemic Risk',
                    color=color1, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Congestion Level (Traffic Jam Factor)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Epistemic Risk (Uncertainty)', fontsize=12, color=color1, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    ax2 = ax1.twinx()
    color2 = 'goldenrod'
    bars2 = ax2.bar(x + width/2, stal_means, width, label='Staleness Risk',
                    color=color2, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Staleness Risk (Latency Penalty)', fontsize=12, color=color2, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_yscale('log')
    
    for bars, axis in [(bars1, ax1), (bars2, ax2)]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                axis.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}',
                         ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_title('Risk Distribution by Congestion Level\n(Epistemic vs. Staleness Components)',
                  fontsize=14, fontweight='bold', pad=20)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11, framealpha=0.9)
    
    fig.tight_layout()
    plt.savefig("risk_distribution_combined.pdf", format='pdf', bbox_inches='tight', dpi=300)
    print("Combined risk distribution plot saved as risk_distribution_combined.pdf")
    plt.show()


def plot_verification_staleness_dist(lv_history, smt_depths, sla_deadline=1.0):
    """
    Plot verification latency vs SMT depth.
    
    Args:
        lv_history: Latency values
        smt_depths: SMT depth values
        sla_deadline: SLA deadline threshold
    """
    plt.figure(figsize=(10, 6))
    
    lv = np.array(lv_history)
    phi = np.array(smt_depths)
    
    if np.all(phi == phi[0]):
        phi = phi + np.random.normal(0, 0.01, size=phi.shape)
    
    plt.scatter(phi, lv, alpha=0.5, color='royalblue', s=15, label='Empirical $L_v$')
    plt.axhline(y=sla_deadline, color='darkred', linestyle='--',
                label=fr'SLA Deadline ($\Delta t_{{req}}$ = {sla_deadline}ms)')
    
    plt.yscale('symlog', linthresh=2.0)
    plt.ylim(0.1, max(lv.max() * 1.1, 10))
    
    plt.xlabel(r"SMT Verification Depth ($\Phi$)", fontsize=12)
    plt.ylabel(r"Latency ($L_v$) [ms]", fontsize=12)
    plt.title(r"Empirical Verification Latency ($L_v$) vs. SMT Depth")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig("verified_latency_vs_SMTdepth.png")
    plt.show()
