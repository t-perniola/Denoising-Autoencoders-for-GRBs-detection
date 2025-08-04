import tensorflow as tf
import tensorflow_probability as tfp

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
