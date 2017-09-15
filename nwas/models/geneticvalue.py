import edward as ed
import tensorflow as tf

from edward.models import RandomVariable
from tensorflow.contrib.distributions import (
    Distribution,
    FULLY_REPARAMETERIZED,
    RegisterKL,
)

class distribution_GeneticValue(Distribution):
    """Represent the local reparamterization eta = x * theta as proposed in Kingma, Salimans, & Welling 2015.

    We use this to avoid sampling from the spike-and-slab prior. We have:

    E_q[x theta] = x * (alpha .* beta)
    V_q[x theta] = x^2 * (alpha ./ gamma + alpha .* (1 - alpha) .* beta^2)

    Then, we can just sample from a Gaussian and scale appropriately to
    evaluate E_q[ln p(x | ...)].

    """
    def __init__(self, x, theta, validate_args=False, allow_nan_stats=False,
                 name='GeneticValue'):
        self._x = x
        self._theta = theta

        self._dist = ed.models.Normal(
            loc=tf.matmul(self._x, self._theta.mean()),
            scale=tf.sqrt(tf.matmul(tf.square(self._x),
                                    self._theta.variance()))
        )

        super(distribution_GeneticValue, self).__init__(
            allow_nan_stats=allow_nan_stats,
            dtype=self._theta.dtype,
            name=name,
            reparameterization_type=FULLY_REPARAMETERIZED,
            validate_args=validate_args,
        )

    def _batch_shape(self):
        return self._dist._batch_shape()

    def _log_prob(self, value):
        return self._dist._log_prob(value)

    def _sample_n(self, n, seed=None):
        return self._dist._sample_n(n, seed)

    def _mean(self):
        return self._dist._mean()

@RegisterKL(distribution_GeneticValue, distribution_GeneticValue)
def kl_geneticvalue(q, p, name=None):
    """Dummy KL divergence between genetic values

    """
    return 0.0

class GeneticValue(RandomVariable, distribution_GeneticValue):
    def __init__(self, *args, **kwargs):
        RandomVariable.__init__(self, *args, **kwargs)
