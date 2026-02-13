"""
Reliability diagram visualization for calibration analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from scipy.signal import savgol_filter


def plot_reliability_diagram(y_true, prob_pred, n_bins=10):
    """
    Plot a reliability (calibration) diagram.
    
    Args:
        y_true: Ground truth labels (0/1)
        prob_pred: Predicted probabilities
        n_bins: Number of calibration bins
    """
    y_true = np.array(y_true)
    prob_pred = np.array(prob_pred)
    
    min_len = min(len(y_true), len(prob_pred))
    if min_len == 0:
        print("Reliability Diagram: no data provided.")
        return
    
    y_true = y_true[:min_len]
    prob_pred = prob_pred[:min_len]
    
    if not set(np.unique(y_true)).issubset({0, 1}):
        y_true_binary = (y_true > 0.5).astype(int)
    else:
        y_true_binary = y_true.astype(int)
    
    unique_classes = np.unique(y_true_binary)
    if len(unique_classes) < 2:
        print("Warning: Reliability Diagram -- y_true contains a single class.")
        bin_pred = np.linspace(0.0, 1.0, n_bins)
        bin_true = np.full_like(bin_pred, fill_value=float(unique_classes[0]))
    else:
        bin_true, bin_pred = calibration_curve(y_true_binary, prob_pred, n_bins=n_bins, strategy='uniform')
    
    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.plot(bin_pred, bin_true, marker='s', color='green', label='GIRAF Agent')
    plt.fill_between(bin_pred, bin_pred, bin_true, color='pink', alpha=0.25, label='Calibration Error')
    
    plt.xlabel(r'Reported Confidence ($B_R$)')
    plt.ylabel('Empirical Success Rate')
    plt.title('Reliability Diagram: SLA & Trust Alignment')
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def generate_comparative_reliability_diagram(y_true, giraf_preds, pretrained_preds):
    """
    Generate comparative reliability diagram for GIRAF vs baseline.
    
    Args:
        y_true: Ground truth labels
        giraf_preds: GIRAF model predictions
        pretrained_preds: Baseline model predictions
    """
    min_len = min(len(y_true), len(giraf_preds), len(pretrained_preds))
    y_true = np.array(y_true[:min_len]).astype(int)
    giraf_preds = np.array(giraf_preds[:min_len])
    pretrained_preds = np.array(pretrained_preds[:min_len])
    
    if len(np.unique(y_true)) < 2:
        y_true[0] = 0
    
    prob_true_g, prob_pred_g = calibration_curve(y_true, giraf_preds, n_bins=10)
    prob_true_p, prob_pred_p = calibration_curve(y_true, pretrained_preds, n_bins=10)
    
    window = 5
    poly = 2
    smooth_g = savgol_filter(prob_true_g, window, poly)
    smooth_p = savgol_filter(prob_true_p, window, poly)
    
    plt.figure(figsize=(9, 7))
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Ideal Calibration')
    
    plt.plot(prob_pred_p, smooth_p, 'ro-', label='Pretrained LLM (Baseline)', markersize=4)
    plt.fill_between(prob_pred_p, prob_pred_p, smooth_p, color='red', alpha=0.1)
    
    plt.plot(prob_pred_g, smooth_g, 'gs-', label='GIRAF-Aligned Agent', markersize=4)
    plt.fill_between(prob_pred_g, prob_pred_g, smooth_g, color='green', alpha=0.1)
    
    plt.xlabel('Reported Confidence ($B_R$)')
    plt.ylabel('Empirical Accuracy (Success Rate)')
    plt.title('Reliability Comparison: GIRAF vs. Pretrained Baseline')
    plt.legend()
    plt.xlim(0.3, 1.0)
    plt.ylim(0.3, 1.0)
    plt.grid(alpha=0.3, linestyle=':')
    plt.savefig("GIRAF_Smoothed_Reliability.pdf")
    plt.show()
