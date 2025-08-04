import numpy as np
from utils.general_utils import reshape_data

# Moving average (smoothing LLs)
def moving_average(x, w):
    pad = w // 2
    x = np.pad(x, pad, mode='edge')
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode='valid')

def reshape_and_label(perturbed_signal, anomaly_mask, chunk_size = 100):
  """
  Chunk the signal and generate per-chunk anomaly labels.
  Returns:
  - chunks: 2D array (n_chunks x chunk_size)
  - labels: 1D array (n_chunks)
  """
  perturbed_data, n_chunks = reshape_data(perturbed_signal, chunk_size)
  labels_mask = anomaly_mask[:n_chunks * chunk_size].reshape(n_chunks, chunk_size)
  labels = (labels_mask.mean(axis=1) > 0.2).astype(int)
  #labels = labels_mask.any(axis=1).astype(int)
  return perturbed_data, labels

# Try to inject perturbations that reach the desired target percentage of anomalies
def inject_until_target(signal, target_perc, burst_duration_range,
                        amplitude_range, lambda_range, chunk_size=100, 
                        max_trials=500):

    perturbed = signal.copy()
    anomaly_mask = np.zeros_like(signal, dtype=bool)
    num_attempts = 0

    while True:
        num_attempts += 1
        # Inject a burst
        perturbed, anomaly_mask = inject_one_burst(
            perturbed, anomaly_mask, burst_duration_range, amplitude_range, lambda_range
        )

        # Recompute labels
        chunks, labels = reshape_and_label(perturbed, anomaly_mask, chunk_size)
        current_perc = labels.mean()

        if current_perc >= target_perc and current_perc <= target_perc + 0.01:
            break
        if num_attempts > max_trials:
            print("Max injection attempts reached; anomaly rate may be under or over target.")
            break

    return chunks, anomaly_mask, labels

# Just inject one burst!
def inject_one_burst(signal, anomaly_mask, duration_range, amp_range, lambda_range):
    signal = signal.copy()
    mask = anomaly_mask.copy()
    
    duration = np.random.randint(*duration_range)
    start = np.random.randint(0, len(signal) - duration)
    amplitude = np.random.uniform(*amp_range)
    lambd = np.random.uniform(*lambda_range)
    burst = amplitude * np.exp(-lambd * np.arange(duration))

    signal[start : start + duration] += burst
    mask[start : start + duration] = True

    return signal, mask