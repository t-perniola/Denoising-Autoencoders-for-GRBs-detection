import numpy as np

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

# Reshaping data
def reshape_data(data, chunk_size):
  num_chunks = len(data) // chunk_size
  reshaped_data = data[:num_chunks * chunk_size].reshape((num_chunks, chunk_size))
  return reshaped_data, num_chunks