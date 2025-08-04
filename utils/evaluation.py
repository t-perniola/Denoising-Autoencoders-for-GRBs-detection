import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.moving_average import *
from utils.data_generation import *
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, precision_score, recall_score, f1_score, confusion_matrix
from utils.general_utils import *
from utils.visualization import plot_conf_matrices, plot_lls_anomalies
from utils.distribution import skew_normal_nll

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