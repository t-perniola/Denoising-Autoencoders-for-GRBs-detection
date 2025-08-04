import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

# PLOT METRICS
def plot_conf_matrices(cm_f1, cm_perc):
  fig, axs = plt.subplots(1, 2, figsize=(15, 5))
  sns.heatmap(cm_f1, annot=True, fmt='d', cmap='Blues', ax=axs[0])
  axs[0].set_title("Confusion Matrix (F1 Threshold)")
  axs[0].set_xlabel("Predicted")
  axs[0].set_ylabel("True")
  axs[0].set_xticklabels(["normal", "anomaly"])
  axs[0].set_xticklabels(["normal", "anomaly"])

  sns.heatmap(cm_perc, annot=True, fmt='d', cmap='Oranges', ax=axs[1])
  axs[1].set_title("Confusion Matrix (99.8th Percentile)")
  axs[1].set_xlabel("Predicted")
  axs[1].set_ylabel("True")
  axs[1].set_xticklabels(["normal", "anomaly"])
  axs[1].set_xticklabels(["normal", "anomaly"])

  plt.tight_layout()
  plt.show()

def plot_ROC(y_true, y_scores):
  fpr, tpr, thresholds = roc_curve(y_true, y_scores)
  auc_score = roc_auc_score(y_true, y_scores)

  plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
  plt.plot([0, 1], [0, 1], linestyle='--')
  plt.xlabel("FP rate")
  plt.ylabel("TP rate")
  plt.title("ROC Curve")
  plt.legend()
  plt.grid()
  plt.show()

def plot_PR_curve(y_true, y_scores):
  precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

  # Compute F1-Score for each threshold
  f1_scores = 2 * (precision * recall) / (precision + recall)

  # Find the threshold that maximizes the F1-Score
  best_threshold_index = np.argmax(f1_scores)
  best_threshold = thresholds[best_threshold_index]
  best_f1_score = f1_scores[best_threshold_index]

  print(f"Best threshold (based on F1-score): {best_threshold:.4f}")
  print(f"Best F1-score: {best_f1_score:.4f}")

  # Plot Precision-Recall curve and mark the best threshold
  plt.figure(figsize=(8, 6))
  plt.plot(recall, precision, label='Precision-Recall curve')
  plt.scatter(recall[best_threshold_index], precision[best_threshold_index], color='red', label=f"Best threshold: {best_threshold:.4f}")
  plt.xlabel("Recall")
  plt.ylabel("Precision")
  plt.title("Precision-Recall Curve")
  plt.legend()
  plt.show()

# Plotting the likelihoods and the anomalies (both real and predicted)
def plot_lls_anomalies(likelihoods, mean_train_ll, gt_anomalies,
                       pred_anomalies, threshold, title="",
                       percentile=False, raw_lls=None):
    plt.figure(figsize=(12, 6))

    # Raw LLs if provided
    if raw_lls is not None:
        plt.plot(raw_lls, label='Raw LLs', color='red', alpha=0.4)

    plt.plot(likelihoods, label='Smoothed LLs' if raw_lls is not None else "LLs", color='red')

    # Anomalies
    plt.scatter(gt_anomalies, np.array(likelihoods)[gt_anomalies],
                label="GT anomalies", color="green", marker="x", s=80)
    plt.scatter(pred_anomalies, np.array(likelihoods)[pred_anomalies],
                label="Pred anomalies", color="blue", marker=".", s=50)

    # Plot thresholds
    if percentile:
        plt.axhline(threshold, color='green', linestyle='-', label='99.8th threshold')
        plt.fill_between(np.arange(len(likelihoods)), mean_train_ll, threshold,
                         color="red", alpha=0.2, label="Confidence band")
    else:
        plt.axhline(threshold, color='orange', linestyle='-', label='F1 threshold')

    plt.axhline(mean_train_ll, color='blue', linestyle='--', label='Training mean')
    plt.title(title)
    plt.xlabel("Test light curves")
    plt.ylabel("Likelihood")
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.tight_layout()
    plt.show()


# Showing history of the model
def show_history(history):
  plt.figure(figsize=(12, 5))
  for metric in history.history.keys():
      plt.plot(history.history[metric], label=metric)
  if len(history.history.keys()) > 0:
      plt.legend()
  plt.title('Training history')
  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.grid(linestyle=':')
  plt.tight_layout()
  plt.show()

# Showing lightcurves that account for bin duration
def show_box_lightcurves(data, selected_indices, size, label,
                         bin_duration=0.05, color="tab:blue",
                         overlay_data=None, overlay_color="gray", overlay_label="Background"):
    """
    Plot step-style (box) lightcurves with optional background overlay.

    Args:
        data: np.ndarray of shape (n_series, chunk_size), e.g. perturbed data
        selected_indices: list of indices to show
        size: number of subplots
        label: label for main signal
        bin_duration: float, duration of each time bin
        color: color for the main signal
        overlay_data: optional np.ndarray, same shape as `data`
            background data to overlay
        overlay_color: color of the overlay signal (default = gray)
        overlay_label: legend label for overlay (default = "Background")
    """
    if size == 1:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax = [ax]
    else:
        fig, ax = plt.subplots(size, 1, figsize=(13, 7))

    for i, (axis, idx) in enumerate(zip(ax, selected_indices)):
        counts = data[idx]
        time = np.arange(len(counts) + 1) * bin_duration
        counts_step = np.append(counts, counts[-1])

        axis.step(time, counts_step, where='post', color=color, label=label)

        if overlay_data is not None:
            overlay = overlay_data[idx]
            overlay_step = np.append(overlay, overlay[-1])
            axis.step(time, overlay_step, where='post', color=overlay_color, label=overlay_label)

        axis.set_xticks(np.arange(0, time[-1] + bin_duration, step=bin_duration * 10))
        axis.set_title(f"{label} light curves" if i == 0 else "")
        axis.grid(linestyle=':')
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Photon counts")
        axis.legend()

    plt.tight_layout()
    plt.show()