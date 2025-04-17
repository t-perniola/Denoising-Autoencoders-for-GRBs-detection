import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Apply bursts to the given data
def add_burst(data, anomaly_indices, anomaly_percentage, amplitudes,
    peak_time, burst_duration
):
  ## Parameters
  amplitude_lb = amplitudes["lb"]
  amplitude_ub = amplitudes["ub"]
  lambda_lb = 0.1
  lambda_ub = 0.2

  ## Apply anomalies only to a selected percentage of light curves
  perturbed_data = data.copy()
  for i in anomaly_indices:
      amplitude = np.random.randint(amplitude_lb, amplitude_ub)
      lambd = np.random.uniform(lambda_lb, lambda_ub)
      burst = amplitude * np.exp(-lambd * np.arange(burst_duration))
      perturbed_data[i, peak_time:burst_duration + peak_time] += burst  # Add the anomaly

  return perturbed_data, anomaly_indices


# Showing lightcurves
def show_lightcurves(data, num_samples, size, label, color = "blue"):
  #idx1, idx2 = np.random.randint(0, size, size=2)
  idx1, idx2 = np.random.choice(size, size=2, replace=False)
  fig, ax = plt.subplots(2, 1, figsize=(13, 7))

  # Plot each time series
  for i, (ax, idx) in enumerate(zip(ax, [idx1, idx2])):
      ax.plot(data[idx], label=f"series {idx}", color = color)

      # Set the ticks every 10 bins
      ax.set_xticks(np.arange(0, len(data[idx]), step=10))  # Ticks every 10 bins
      ax.set_xticklabels(np.arange(0, len(data[idx]), step=10))  # Label ticks with corresponding bin numbers

      ax.set_title(f"{label} light curves" if i == 0 else "")
      ax.legend()
      ax.grid(linestyle=':')
      ax.set_xlabel("Bins")
      ax.set_ylabel("Value")

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


# Plotting the likelihoods and the anomalies (both real and predicted)
def plot_lls_anomalies(likelihoods, mean_train_ll, gt_anomalies, pred_anomalies, threshold, thresh_95):
  plt.figure(figsize=(12, 6))

  # Plot test likelihoods
  plt.plot(likelihoods, label='Test likelihoods', color='red', zorder=1)

  # Plot anomalies
  ## Ground truths
  plt.scatter(gt_anomalies, np.array(likelihoods)[gt_anomalies],
              label="Ground truth anomalies", color="green", marker="x", s=80)
  ## Predicted
  plt.scatter(pred_anomalies, np.array(likelihoods)[pred_anomalies],
              label="Predicted anomalies", color="blue", marker=".", s=50)

  # Plot mean
  plt.axhline(mean_train_ll, color='blue', linestyle='--', label='Training mean')

  # Plot threshold
  plt.axhline(threshold, color='green', linestyle='--', label='Optimized threshold')
  plt.axhline(thresh_95, color='orange', linestyle='--', label='95th perc. threshold')

  # Plot confidence band
  plt.fill_between(np.arange(len(likelihoods)), mean_train_ll, thresh_95, color="red", alpha=0.2, label="Confidence band")

  # Customize plot
  plt.title("Test likelihoods with Confidence interval")
  plt.xlabel("Test light curves")
  plt.ylabel("Likelihood")
  plt.grid(True, linestyle=':')
  plt.legend()
  plt.tight_layout()
  plt.show()


# Plotting metrics
def evaluate_and_plot_per_strength(
    test_data, results, fixed_anomalies_idx, test_anomaly_perc,
    burst_strengths, scaler, compute_likelihood, ae, train_likelihoods,
    num_chunks, thresh_95, peak_time, burst_duration
):
    from sklearn.metrics import (
        roc_curve, roc_auc_score, precision_recall_curve,
        confusion_matrix, precision_score, recall_score
    )

    sns.set(style="whitegrid")

    # Plot setup
    fig_lls, axs_lls = plt.subplots(2, 2, figsize=(14, 8))
    fig_lls.suptitle("Likelihoods per burst strength", fontsize=16)

    fig_roc, axs_roc = plt.subplots(2, 2, figsize=(14, 8))
    fig_roc.suptitle("ROC curves", fontsize=16)

    fig_pr, axs_pr = plt.subplots(2, 2, figsize=(14, 8))
    fig_pr.suptitle("Precision-Recall Curves", fontsize=16)

    fig_cm, axs_cm = plt.subplots(2, 2, figsize=(14, 8))
    fig_cm.suptitle("Confusion Matrices", fontsize=16)

    for idx, (label, amplitudes) in enumerate(burst_strengths.items()):
        # Perturb test data
        test_data_perturbed, test_anomaly_indices = add_burst(
            test_data, fixed_anomalies_idx, test_anomaly_perc, amplitudes,
            peak_time, burst_duration
        )
        test_data_perturbed = scaler.transform(test_data_perturbed)

        # Likelihoods
        likelihoods = compute_likelihood(test_data_perturbed, ae)
        avg_ll, std_ll = np.mean(likelihoods), np.std(likelihoods)

        y_true = np.zeros(num_chunks)
        y_true[test_anomaly_indices] = 1
        y_scores = np.array(likelihoods)

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_score = roc_auc_score(y_true, y_scores)

        # PR curve and best F1 threshold
        precision, recall, pr_thresh = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = pr_thresh[best_idx]
        best_f1 = f1_scores[best_idx]

        # Predictions and metrics
        y_pred = (y_scores > best_threshold).astype(int)
        precision_final = precision_score(y_true, y_pred)
        recall_final = recall_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        anomalous_indices = np.where(y_scores > best_threshold)[0]

        # Save results
        results[label].update({
            "Avg_ll": avg_ll, "Std dev": std_ll, "AUC": auc_score,
            "Precision": precision_final, "Recall": recall_final,
            "F1": best_f1, "Best Threshold (F1)": best_threshold
        })

        # Plotting
        row, col = divmod(idx, 2)

        axs_lls[row, col].plot(likelihoods, label="Test Likelihoods", color="red")
        axs_lls[row, col].axhline(best_threshold, linestyle="--", color="green", label="Upper Threshold")
        axs_lls[row, col].axhline(thresh_95, linestyle="--", color="orange", label="95th Percentile")
        axs_lls[row, col].scatter(test_anomaly_indices, np.array(likelihoods)[test_anomaly_indices],
                                  color="green", marker="x", s=80, label="GT Anomalies")
        axs_lls[row, col].scatter(anomalous_indices, np.array(likelihoods)[anomalous_indices],
                                  color="blue", marker=".", s=50, label="Predicted Anomalies")
        axs_lls[row, col].set_title(label)
        axs_lls[row, col].legend()

        axs_roc[row, col].plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
        axs_roc[row, col].plot([0, 1], [0, 1], linestyle="--", color="gray")
        axs_roc[row, col].set_title(label)
        axs_roc[row, col].set_xlabel("FPR")
        axs_roc[row, col].set_ylabel("TPR")
        axs_roc[row, col].legend()

        axs_pr[row, col].plot(recall, precision, label=f"F1 = {best_f1:.2f}")
        axs_pr[row, col].set_title(label)
        axs_pr[row, col].set_xlabel("Recall")
        axs_pr[row, col].set_ylabel("Precision")
        axs_pr[row, col].legend()

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axs_cm[row, col])
        axs_cm[row, col].set_title(label)
        axs_cm[row, col].set_xlabel("Predicted")
        axs_cm[row, col].set_ylabel("True")

    # Final layout
    fig_lls.tight_layout(rect=[0, 0, 1, 0.95])
    fig_roc.tight_layout(rect=[0, 0, 1, 0.95])
    fig_pr.tight_layout(rect=[0, 0, 1, 0.95])
    fig_cm.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return results