import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import skewnorm
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, roc_curve, roc_auc_score,
    precision_recall_curve, classification_report,
    confusion_matrix, precision_score, recall_score
)

# Generate data sampled from a Skew-Normal
def generate_skewnorm_data(alpha, mu, sigma, num_samples): # size: num of distribution to output
  data = skewnorm.rvs(alpha, loc=mu, scale=sigma, size=num_samples, random_state=42)
  return data

# Reshape data
def reshape_data(data, chunk_size):
  num_chunks = len(data) // chunk_size
  reshaped_data = data[:num_chunks * chunk_size].reshape((num_chunks, chunk_size))
  return reshaped_data, num_chunks

# Custom class definition
tfd = tfp.distributions
class SkewNormal(tfd.Distribution):
    def __init__(self, loc, scale, skewness, validate_args=False, allow_nan_stats=True, name="SkewNormal"):
        parameters = dict(locals())
        self.loc = tf.convert_to_tensor(loc, dtype=tf.float32)
        self.scale = tf.convert_to_tensor(scale, dtype=tf.float32)
        self.skewness = tf.convert_to_tensor(skewness, dtype=tf.float32)
        self.normal = tfd.Normal(loc=0.0, scale=1.0)  # Standard normal

        super(SkewNormal, self).__init__(
            dtype=tf.float32,
            reparameterization_type=tfd.NOT_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name=name.replace(" ", "_"),
        )

    # Log-probability density function (PDF) of the Skew-Normal distribution.
    def _log_prob(self, value):
        z = (value - self.loc) / self.scale
        normal_log_prob = self.normal.log_prob(z)
        skew_factor = 2 * self.normal.cdf(self.skewness * z)

        # Ensure numerical stability
        skew_factor = tf.clip_by_value(skew_factor, 1e-6, 1.0)  # Avoid log(0)
        return normal_log_prob + tf.math.log(skew_factor) - tf.math.log(self.scale)

# Negative Log-Likelihood for the Skew-Normal distribution.
def skew_normal_nll(y_true, params):
  mu, sigma, alpha = params[:, 0], params[:, 1], params[:, 2]

  # reshape to (batch_size, 1)
  mu = tf.expand_dims(mu, axis=-1)
  sigma = tf.expand_dims(sigma, axis=-1)
  alpha = tf.expand_dims(alpha, axis=-1)

  sn = SkewNormal(loc=mu, scale=sigma, skewness=alpha)
  return -tf.reduce_mean(sn.log_prob(y_true))

# Apply bursts to the time-series, at fixed indices
def add_burst_fixed(data, anomaly_indices, amplitudes,
                    peak_time, burst_duration):
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

  return perturbed_data

# Add random perturbations, randomizing over burst duration and peak time
def add_random_burst(data, anomaly_indices, anomaly_perc, amplitudes, chunk_size=100):
  burst_locations = {}

  ## Parameters
  amplitude_lb = amplitudes["lb"]
  amplitude_ub = amplitudes["ub"]
  lambda_lb = 0.02
  lambda_ub = 0.4

  ## Apply anomalies only to a selected percentage of light curves
  perturbed_data = data.copy()
  for i in anomaly_indices:
      # Randomize burst params
      burst_duration = np.random.randint(5, 40)
      peak_time = int(np.random.beta(2, 3) * (chunk_size - burst_duration))
      amplitude = np.random.randint(amplitude_lb, amplitude_ub)
      lambd = np.random.uniform(lambda_lb, lambda_ub)

      # Update dict of burst loc
      burst_locations[int(i)] = (peak_time, burst_duration)

      # Add the anomaly
      burst = amplitude * np.exp(-lambd * np.arange(burst_duration))
      perturbed_data[i, peak_time:burst_duration + peak_time] += burst

  return perturbed_data, anomaly_indices, burst_locations

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

# Moving average (smoothing LLs)
def moving_average(x, w):
    pad = w // 2
    x = np.pad(x, pad, mode='edge')
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode='valid')

# Plotting metrics for each level of strength
def evaluate_and_plot_per_strength(
    test_data, results, model, anomaly_indices,
    burst_strengths, scaler, train_likelihoods,
    num_chunks, peak_time, burst_duration,
    window_size=1  # default is no smoothing
):
    sns.set(style="whitegrid")

    fig_lls, axs_lls = plt.subplots(2, 2, figsize=(14, 8))
    fig_lls.suptitle("Likelihoods per burst strength", fontsize=16)

    fig_roc, axs_roc = plt.subplots(2, 2, figsize=(14, 8))
    fig_roc.suptitle("ROC curves", fontsize=16)

    fig_pr, axs_pr = plt.subplots(2, 2, figsize=(14, 8))
    fig_pr.suptitle("Precision-Recall curves", fontsize=16)

    fig_cm, axs_cm = plt.subplots(2, 2, figsize=(14, 8))
    fig_cm.suptitle("Confusion matrices", fontsize=16)

    # Smooth training likelihoods if needed
    if window_size > 1:
        smoothed_train_lls = moving_average(train_likelihoods, w=window_size)
        thresh_perc = np.percentile(smoothed_train_lls, 99.8)
    else:
        thresh_perc = np.percentile(train_likelihoods, 99.8)

    for idx, (label, amplitudes) in enumerate(burst_strengths.items()):
        test_data_perturbed = add_burst_fixed(
            test_data, anomaly_indices, amplitudes,
            peak_time, burst_duration
        )
        test_data_perturbed = scaler.transform(test_data_perturbed)

        raw_likelihoods = compute_likelihoods(test_data_perturbed, model,
                                              num_chunks, skew_normal_nll)

        if window_size > 1:
            likelihoods = moving_average(raw_likelihoods, w=window_size)
        else:
            likelihoods = raw_likelihoods

        avg_ll, std_ll = np.mean(likelihoods), np.std(likelihoods)
        y_true = np.zeros(num_chunks)
        y_true[anomaly_indices] = 1
        y_scores = np.array(likelihoods)

        # ROC & AUC
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_score = roc_auc_score(y_true, y_scores)

        # PR curve
        precision, recall, pr_thresh = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        f1_scores = f1_scores[:-1]
        best_idx = np.argmax(f1_scores)
        best_threshold = pr_thresh[best_idx]
        best_f1 = f1_scores[best_idx]

        # Classification
        y_pred = (y_scores > best_threshold).astype(int)
        precision_final = precision_score(y_true, y_pred)
        recall_final = recall_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        anomalous_indices = np.where(y_scores > best_threshold)[0]

        results[model.name][label].update({
            "Avg ll": avg_ll, "Std dev": std_ll, "AUC": auc_score,
            "Precision": precision_final, "Recall": recall_final,
            "F1": best_f1, "Best Threshold (F1)": best_threshold
        })

        row, col = divmod(idx, 2)

        # --- Likelihoods subplot ---
        axs_lls[row, col].plot(raw_likelihoods, label="Raw LLs", color="red", alpha=0.4)
        axs_lls[row, col].plot(likelihoods, label="Smoothed LLs" if window_size > 1 else "LLs", color="red")
        axs_lls[row, col].axhline(best_threshold, linestyle="--", color="green", label="Best F1 th")
        axs_lls[row, col].axhline(thresh_perc, linestyle="--", color="orange", label="99.8th th")
        axs_lls[row, col].scatter(anomaly_indices, np.array(likelihoods)[anomaly_indices],
                                  color="green", marker="x", s=80, label="GT anomalies")
        axs_lls[row, col].scatter(anomalous_indices, np.array(likelihoods)[anomalous_indices],
                                  color="blue", marker=".", s=50, label="Pred anomalies")
        axs_lls[row, col].set_title(label)
        axs_lls[row, col].legend()

        # --- ROC subplot ---
        axs_roc[row, col].plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
        axs_roc[row, col].plot([0, 1], [0, 1], linestyle="--", color="gray")
        axs_roc[row, col].set_title(label)
        axs_roc[row, col].set_xlabel("FPR")
        axs_roc[row, col].set_ylabel("TPR")
        axs_roc[row, col].legend()

        # --- PR subplot ---
        axs_pr[row, col].plot(recall, precision, label=f"F1 = {best_f1:.2f}")
        axs_pr[row, col].set_title(label)
        axs_pr[row, col].set_xlabel("Recall")
        axs_pr[row, col].set_ylabel("Precision")
        axs_pr[row, col].legend()

        # --- Confusion matrix subplot ---
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axs_cm[row, col])
        axs_cm[row, col].set_title(label)
        axs_cm[row, col].set_xlabel("Predicted")
        axs_cm[row, col].set_ylabel("True")
        axs_cm[row, col].set_xticklabels(["normal", "anomaly"])
        axs_cm[row, col].set_yticklabels(["normal", "anomaly"])

    fig_lls.tight_layout(rect=[0, 0, 1, 0.95])
    fig_roc.tight_layout(rect=[0, 0, 1, 0.95])
    fig_pr.tight_layout(rect=[0, 0, 1, 0.95])
    fig_cm.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return results, thresh_perc

# Plotting metrics for each level of strength
def evaluate_and_plot_per_strength_ma(
    test_data, results, test_anomaly_perc, model, burst_strengths, scaler,
    train_likelihoods, num_chunks, burst_duration_range, lambda_range, window_size=1  # default is no smoothing
):
    sns.set(style="whitegrid")

    fig_lls, axs_lls = plt.subplots(2, 2, figsize=(14, 8))
    fig_lls.suptitle("Likelihoods per burst strength", fontsize=16)

    fig_roc, axs_roc = plt.subplots(2, 2, figsize=(14, 8))
    fig_roc.suptitle("ROC curves", fontsize=16)

    fig_pr, axs_pr = plt.subplots(2, 2, figsize=(14, 8))
    fig_pr.suptitle("Precision-Recall curves", fontsize=16)

    fig_cm, axs_cm = plt.subplots(2, 2, figsize=(14, 8))
    fig_cm.suptitle("Confusion matrices", fontsize=16)

    # Smooth training likelihoods if needed
    if window_size > 1:
        smoothed_train_lls = moving_average(train_likelihoods, w=window_size)
        thresh_perc = np.percentile(smoothed_train_lls, 99.8)
    else:
        thresh_perc = np.percentile(train_likelihoods, 99.8)

    for idx, (label, amplitudes) in enumerate(burst_strengths.items()):
        # Perturbing data
        test_data_perturbed, test_anomaly_mask, test_labels = inject_until_target(
                                                    test_data, test_anomaly_perc,
                                                    burst_duration_range, amplitudes,
                                                    lambda_range
                                                    )
        # Capture the anomalous indices
        anomaly_indices = np.where(test_labels == 1)[0]
        test_data_perturbed = scaler.transform(test_data_perturbed)

        raw_likelihoods = compute_likelihoods(test_data_perturbed, model,
                                              num_chunks, skew_normal_nll)
        if window_size > 1:
            likelihoods = moving_average(raw_likelihoods, w=window_size)
        else:
            likelihoods = raw_likelihoods

        avg_ll, std_ll = np.mean(likelihoods), np.std(likelihoods)
        y_true = np.zeros(num_chunks)
        y_true[anomaly_indices] = 1
        y_scores = np.array(likelihoods)

        # ROC & AUC
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_score = roc_auc_score(y_true, y_scores)

        # PR curve
        precision, recall, pr_thresh = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        f1_scores = f1_scores[:-1]
        best_idx = np.argmax(f1_scores)
        best_threshold = pr_thresh[best_idx]
        best_f1 = f1_scores[best_idx]

        # Classification
        y_pred = (y_scores > best_threshold).astype(int)
        precision_final = precision_score(y_true, y_pred)
        recall_final = recall_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        anomalous_indices = np.where(y_scores > best_threshold)[0]

        results[model.name][label].update({
            "Avg ll": avg_ll, "Std dev": std_ll, "AUC": auc_score,
            "Precision": precision_final, "Recall": recall_final,
            "F1": best_f1, "Best Threshold (F1)": best_threshold
        })

        row, col = divmod(idx, 2)

        # --- Likelihoods subplot ---
        axs_lls[row, col].plot(raw_likelihoods, label="Raw LLs", color="red", alpha=0.4)
        axs_lls[row, col].plot(likelihoods, label="Smoothed LLs" if window_size > 1 else "LLs", color="red")
        axs_lls[row, col].axhline(best_threshold, linestyle="--", color="green", label="Best F1 th")
        axs_lls[row, col].axhline(thresh_perc, linestyle="--", color="orange", label="99.8th th")
        axs_lls[row, col].scatter(anomaly_indices, np.array(likelihoods)[anomaly_indices],
                                  color="green", marker="x", s=80, label="GT anomalies")
        axs_lls[row, col].scatter(anomalous_indices, np.array(likelihoods)[anomalous_indices],
                                  color="blue", marker=".", s=50, label="Pred anomalies")
        axs_lls[row, col].set_title(label)
        axs_lls[row, col].legend()

        # --- ROC subplot ---
        axs_roc[row, col].plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
        axs_roc[row, col].plot([0, 1], [0, 1], linestyle="--", color="gray")
        axs_roc[row, col].set_title(label)
        axs_roc[row, col].set_xlabel("FPR")
        axs_roc[row, col].set_ylabel("TPR")
        axs_roc[row, col].legend()

        # --- PR subplot ---
        axs_pr[row, col].plot(recall, precision, label=f"F1 = {best_f1:.2f}")
        axs_pr[row, col].set_title(label)
        axs_pr[row, col].set_xlabel("Recall")
        axs_pr[row, col].set_ylabel("Precision")
        axs_pr[row, col].legend()

        # --- Confusion matrix subplot ---
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axs_cm[row, col])
        axs_cm[row, col].set_title(label)
        axs_cm[row, col].set_xlabel("Predicted")
        axs_cm[row, col].set_ylabel("True")
        axs_cm[row, col].set_xticklabels(["normal", "anomaly"])
        axs_cm[row, col].set_yticklabels(["normal", "anomaly"])

    fig_lls.tight_layout(rect=[0, 0, 1, 0.95])
    fig_roc.tight_layout(rect=[0, 0, 1, 0.95])
    fig_pr.tight_layout(rect=[0, 0, 1, 0.95])
    fig_cm.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return results, thresh_perc

# Generate & perturb data (aggregating function)
def prepare_test_data(distribution_params, num_samples, chunk_size,
                      test_anomaly_perc, burst_strengths, strength_lvl,
                      scaler, peak_time, burst_duration,
                      burst_type="fixed_bursts"):
    """
    Prepares test data by generating and perturbing background signals.

    Parameters:
        - distribution_params: (alpha, mu, sigma)
        - num_samples: total number of points to generate
        - chunk_size: number of bins per lightcurve
        - test_anomaly_perc: percentage of anomalous lightcurves
        - burst_strengths: dictionary of burst levels
        - strength_lvl: which burst level to apply
        - scaler: fitted StandardScaler
        - peak_time: start of burst (for fixed only)
        - burst_duration: duration of burst (for fixed only)
        - burst_type: "fixed" or "random"

    Returns:
        test_data: standardized background data
        test_data_perturbed: standardized perturbed data
        test_anomaly_indices: indices of perturbed samples
        burst_locations: dict for random bursts, None for fixed
    """
    alpha_fit, mu_fit, sigma_fit = distribution_params

    # Generate test data
    test_data = generate_skewnorm_data(alpha_fit, mu_fit, sigma_fit, num_samples // 4)
    test_data, _ = reshape_data(test_data, chunk_size)
    num_chunks_eval = test_data.shape[0]

    # Choose anomaly indices
    test_anomaly_indices = np.random.choice(
        num_chunks_eval, size=int(test_anomaly_perc * num_chunks_eval),
        replace=False
    )

    amplitudes = burst_strengths[strength_lvl]

    if burst_type == "fixed_bursts":
        test_data_perturbed = add_burst_fixed(
            test_data, test_anomaly_indices, amplitudes,
            peak_time, burst_duration
        )
        burst_locations = None

    elif burst_type == "random_bursts":
        test_data_perturbed, test_anomaly_indices, burst_locations = add_random_burst(
            test_data, test_anomaly_indices, test_anomaly_perc, amplitudes
        )

    else:
        raise ValueError(f"Invalid burst_type '{burst_type}', use 'fixed' or 'random'.")

    # Standardize
    test_data_perturbed = scaler.transform(test_data_perturbed)

    return test_data, test_data_perturbed, test_anomaly_indices, burst_locations


# Compute lls
def compute_likelihoods(data, model, num_chunks, skew_normal_nll):
  # Initialize a list to store the likelihoods of all training samples
  likelihoods = []

  # Loop through all training samples
  for chunk in range(num_chunks):
      # Select the training sample
      sample = data[chunk]
      sample = np.expand_dims(sample, axis=0)

      # Predict the output using the trained model
      ae_output = model.predict(sample, verbose=0)

      # Compute the likelihood for this sample using your skew_normal_nll function
      ll = skew_normal_nll(sample, ae_output)

      # Append the likelihood to the list
      likelihoods.append(ll)

  return likelihoods

# Generate perturbed test set, run multiple models, update global results dict, plot
def run_final_eval(distribution_params, num_samples, chunk_size, test_anomaly_perc,
                   burst_strengths, strength_lvl, scaler, peak_time, burst_duration, models,
                   bursts_type, results_test, results, threshold_perc, mean_train_ll_conv,
                   window_size=1):

    # Generating, scaling and perturbing data
    test_data, test_data_perturbed, test_anomaly_indices, _ = prepare_test_data(
      distribution_params, num_samples, chunk_size, test_anomaly_perc,
      burst_strengths, strength_lvl, scaler, peak_time, burst_duration,
      burst_type=bursts_type.lower().replace(" ", "_")  # "fixed_bursts" or "random_bursts"
)

    # Evaluation
    for model in models:
        print(f"Evaluating '{model.name}' with '{bursts_type}' of '{strength_lvl}' strength")

        if model.name in results and strength_lvl in results[model.name]:
            best_threshold = results[model.name][strength_lvl]["Best Threshold (F1)"]
        else:
            best_threshold = threshold_perc

        print(f"- Best F1 threshold for {model.name} is {best_threshold:.4f}")
        print(f"- 99.8th percentile threshold is: {threshold_perc:.4f}")

        # Evaluate model performance using the updated function
        metrics_fixed, conf_matrices_fixed, y_true_fixed, y_scores_fixed, predictions_fixed = evaluate_model_performance(
            test_data_perturbed, model, test_anomaly_indices,
            best_threshold, threshold_perc, mean_train_ll_conv,
            bursts_type, window_size=window_size
        )

        results_test[model.name][bursts_type].update(metrics_fixed)
        plot_conf_matrices(conf_matrices_fixed[0], conf_matrices_fixed[1])
        print("\n")

    return results_test, test_anomaly_indices, test_data

# Generate perturbed test set, run multiple models, update global results dict, plot
def run_final_eval_ma(distribution_params, num_samples, chunk_size, test_anomaly_perc,
                   burst_strengths, strength_lvl, scaler, burst_duration_range, 
                   lambda_range, models, results_test, results, threshold_perc,
                   mean_train_ll_conv, window_size=1):

    # Generating, scaling and perturbing data
    test_data, test_data_perturbed, test_anomaly_indices = prepare_test_data_ma(
      distribution_params, num_samples, chunk_size, test_anomaly_perc,
      burst_strengths, strength_lvl, scaler, burst_duration_range, lambda_range
)

    # Evaluation
    for model in models:
        print(f"Evaluating '{model.name}' with '{strength_lvl}' burst strength")

        if model.name in results and strength_lvl in results[model.name]:
            best_threshold = results[model.name][strength_lvl]["Best Threshold (F1)"]
        else:
            best_threshold = threshold_perc

        print(f"- Best F1 threshold for {model.name} is {best_threshold:.4f}")
        print(f"- 99.8th percentile threshold is: {threshold_perc:.4f}")

        # Evaluate model performance using the updated function
        metrics_fixed, conf_matrices_fixed, y_true_fixed, y_scores_fixed, predictions_fixed = evaluate_model_performance(
            test_data_perturbed, model, test_anomaly_indices,
            best_threshold, threshold_perc, mean_train_ll_conv,
            perturbation_type="Random burst — M.A.", window_size=window_size
        )

        results_test[model.name].update(metrics_fixed)
        plot_conf_matrices(conf_matrices_fixed[0], conf_matrices_fixed[1])
        print("\n")

    return results_test, test_anomaly_indices, test_data

# Generate & perturb data (aggregating function)
def prepare_test_data_ma(distribution_params, num_samples, chunk_size,
                      test_anomaly_perc, burst_strengths, strength_lvl,
                      scaler, burst_duration_range, lambda_range
                      ):
  
    alpha_fit, mu_fit, sigma_fit = distribution_params

    # Generate test data
    test_data = generate_skewnorm_data(alpha_fit, mu_fit, sigma_fit, num_samples // 4)
    amplitude_range = burst_strengths[strength_lvl]

    # Perturbing data
    test_data_perturbed, test_anomaly_mask, test_labels = inject_until_target(
                                                test_data, test_anomaly_perc,
                                                burst_duration_range, amplitude_range.values(),
                                                lambda_range
                                                )
    # Capture the anomalous indices
    test_anomaly_indices = np.where(test_labels == 1)[0]

    # Standardize
    test_data_perturbed = scaler.transform(test_data_perturbed)

    return test_data, test_data_perturbed, test_anomaly_indices

# Compute metrics for each model
def evaluate_model_performance(
    data, model, anomaly_indices,
    best_threshold_f1, percentile_threshold, mean_ll,
    perturbation_type,
    window_size=1
):
    num_chunks = data.shape[0]

    raw_lls = compute_likelihoods(data, model, num_chunks, skew_normal_nll)
    raw_lls = np.array(raw_lls)

    # Optionally apply moving average
    if window_size > 1:
        smoothed_lls = moving_average(raw_lls, w=window_size)
        scores = smoothed_lls
    else:
        scores = raw_lls

    # Labels
    y_true = np.zeros(num_chunks)
    y_true[anomaly_indices] = 1

    # Predictions
    y_pred_f1 = (scores > best_threshold_f1).astype(int)
    y_pred_perc = (scores > percentile_threshold).astype(int)

    # Confusion matrices
    cm_f1 = confusion_matrix(y_true, y_pred_f1)
    cm_perc = confusion_matrix(y_true, y_pred_perc)

    # Metrics (on F1 threshold)
    metrics = {
        "Avg ll": np.mean(scores),
        "Std dev": np.std(scores),
        "AUC": roc_auc_score(y_true, scores),
        "Precision": precision_score(y_true, y_pred_f1),
        "Recall": recall_score(y_true, y_pred_f1),
        "F1": f1_score(y_true, y_pred_f1)
    }

    # Plots
    plot_lls_anomalies(scores, mean_ll, anomaly_indices,
                       np.where(scores > best_threshold_f1)[0],
                       best_threshold_f1,
                       title=f"LLs for {model.name} - {perturbation_type} (F1 Threshold)",
                       percentile=False,
                       raw_lls=raw_lls if window_size > 1 else None)

    plot_lls_anomalies(scores, mean_ll, anomaly_indices,
                       np.where(scores > percentile_threshold)[0],
                       percentile_threshold,
                       percentile=True,
                       title=f"LLs for {model.name} - {perturbation_type} (Percentile Threshold)",
                       raw_lls=raw_lls if window_size > 1 else None)

    return metrics, (cm_f1, cm_perc), y_true, scores, (y_pred_f1, y_pred_perc)

# Multiple seeds run to acheive robustness about metrics
def run_ci_experiment(models_cfg, distribution_params, best_config_model, best_config_batchsize,
                     strength_lvl, burst_type, num_runs, amplitudes,
                     peak_time, burst_duration, alpha_fit, mu_fit, sigma_fit,
                     num_samples, chunk_size, val_anomaly_perc, test_anomaly_perc,
                     burst_strengths, callbacks, results):

    results_list = []

    results_test = {model: {label: {
    "Avg ll": None, "Std dev": None, "AUC": None,
    "Precision": None, "Recall": None, "F1": None
    } for label in ["Fixed bursts", "Random bursts"]}
    for model in models_cfg.keys()}

    for run in range(num_runs):
        print(f"Running seed {run}...")
        np.random.seed(run)
        random.seed(run)
        tf.random.set_seed(run)

        # Generate new training data
        training_data_original = generate_skewnorm_data(alpha_fit, mu_fit, sigma_fit, num_samples)
        training_data_original, num_chunks = reshape_data(training_data_original, chunk_size)

        scaler = StandardScaler()
        training_data = scaler.fit_transform(training_data_original)

        # Validation data
        val_data = generate_skewnorm_data(alpha_fit, mu_fit, sigma_fit,
                                          num_samples = num_samples // 3)
        val_data, num_chunks_val = reshape_data(val_data, chunk_size)
        val_anomaly_indices = np.random.choice(
            num_chunks_val, size=int(val_anomaly_perc * num_chunks_val), replace=False
        )
        if burst_type == "Random bursts":
          val_data_perturbed, _, _ = add_random_burst(val_data, val_anomaly_indices,
                                               val_anomaly_perc, amplitudes)
        else: 
          val_data_perturbed = add_burst_fixed(val_data, val_anomaly_indices, amplitudes,
                                         peak_time, burst_duration)

        val_data_perturbed = scaler.transform(val_data_perturbed)

        # Instantiate models
        models = []
        for name, builder in models_cfg.items():
            input_shape = (chunk_size, 1) if "conv" in name else (chunk_size,)
            model = builder(input_shape, **best_config_model[name])
            model.compile(optimizer="adam", loss=skew_normal_nll)
            models.append(model)

        # Train models
        for model in models:
            model.fit(x = training_data, y = training_data,
                      validation_data = (val_data_perturbed, val_data_perturbed),
                      epochs=30, batch_size=best_config_batchsize[model.name],
                      callbacks=callbacks, verbose=0)

        # Eval likelihoods for thresholding
        train_ll = compute_likelihoods(training_data, models[1], training_data.shape[0], skew_normal_nll)
        mean_train_ll_conv = np.mean(train_ll)
        threshold_perc = np.percentile(train_ll, 99.8)

        # Evaluate
        results_test, _, _ = run_final_eval(
            distribution_params, num_samples, chunk_size, test_anomaly_perc,
            burst_strengths, strength_lvl, scaler, peak_time, burst_duration,
            models, burst_type, results_test, results, threshold_perc, mean_train_ll_conv
        )

        for model in models:
            metrics = results_test[model.name][burst_type]
            results_list.append({
                "model": model.name,
                "seed": run,
                **metrics
            })

    # Convert to DataFrame and export
    results_df = pd.DataFrame(results_list)
    print("CI experiment completed.")

    return results_df

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
      