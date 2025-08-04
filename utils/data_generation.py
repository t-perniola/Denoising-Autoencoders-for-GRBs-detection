from scipy.stats import skewnorm
import numpy as np

# Generate data sampled from a Skew-Normal
def generate_skewnorm_data(alpha, mu, sigma, num_samples): # size: num of distribution to output
  data = skewnorm.rvs(alpha, loc=mu, scale=sigma, size=num_samples, random_state=42)
  return data

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

