import edward as ed
import tensorflow as tf

from edward.models import RandomVariable
from tensorflow.contrib.distributions import (
    BernoulliWithSigmoidProbs,
    Distribution,
    FULLY_REPARAMETERIZED,
    Normal,
    RegisterKL,
    kl_divergence,
)
from tensorflow.python.ops import (
    array_ops,
    check_ops,
)

class distribution_SpikeSlab(Distribution):
    """Spike-and-slab prior (point-normal mixture; Mitchell & Beauchamp 1988).

    This is a compound distribution of indicator variables z and values theta.

    p(theta_j, z_j | pi, tau) = pi N(theta_j; 0, tau^{-1}) +
                                   (1 - pi) delta(theta_j)

    We want to use the variational approximation to efficiently estimate the
    posterior w.r.t. this prior. The conjugate mean-field variational
    approximation admits an analytical KL (Carbonetto & Stephens 2012).

    q(theta_j, z_j | alpha, beta, gamma) = alpha_j N(theta_j; beta_j, gamma_j^{-1}) + 
                                           (1 - \alpha_j) delta(\theta_j)

    This implementation does not support sampling (does not scale to high
    dimensions). Instead, use the local reparameterization trick (see
    nwas.model.GeneticValue; Kingma, Salimans, & Welling 2015).

    """
    def __init__(self, alpha, beta, gamma, validate_args=False,
                 allow_nan_stats=True, name='SpikeSlab'):
        # This is needed so that tf.python.ops.distributions.Distribution saves
        # positional arguments. Otherwise, ed.util.random_variables.copy
        # blows up
        parameters = locals()

        # c.f. ed.models.empirical
        with tf.name_scope(name, values=[alpha, beta, gamma]):
            with tf.control_dependencies([]):
                self._alpha = array_ops.identity(alpha, name="alpha")
                self._beta = array_ops.identity(beta, name="beta")
                self._gamma = array_ops.identity(gamma, name="gamma")
                check_ops.assert_same_float_dtype([self._alpha, self._beta, self._gamma])

        # c.f. tf.python.ops.distributions.normal.Normal.__init__
        super(distribution_SpikeSlab, self).__init__(
            allow_nan_stats=allow_nan_stats,
            dtype=self._beta.dtype,
            graph_parents=[self._alpha, self._beta, self._gamma],
            name=name,
            parameters=parameters,
            reparameterization_type=FULLY_REPARAMETERIZED,
            validate_args=validate_args,
        )

    def _log_prob(self, value):
        raise NotImplementedError("log_prob is not implemented")

    def _sample_n(self, n, seed=None):
        raise NotImplementedError("sample_n is not implemented")

    @property
    def pip(self):
        with self._name_scope('pip'):
            return self._alpha

    def _mean(self):
        return self._alpha * self._beta

    def _stddev(self):
        return tf.sqrt(self.variance())

    def _variance(self):
        return (self._alpha / self._gamma +
                self._alpha * (1 - self._alpha) * tf.square(self._beta))

@RegisterKL(distribution_SpikeSlab, distribution_SpikeSlab)
def kl_spikeslab(q, p, name=None):
    """KL divergence between point-normal mixture and conjugate mean-field variational approximation

    An analytic form was given without derivation in Carbonetto & Stephens
    2012. It can be derived using Rasmussen and Williams 2006, Eqs. A.22, A.23

    """
    kl_qtheta_ptheta = q._alpha * kl_divergence(
        Normal(loc=q._beta, scale=tf.reciprocal(q._gamma)),
        Normal(loc=p._beta, scale=tf.reciprocal(p._gamma))
    )
    kl_qz_pz = q._alpha * kl_divergence(
        BernoulliWithSigmoidProbs(q._alpha),
        BernoulliWithSigmoidProbs(p._alpha)
    )
    return tf.reduce_sum(kl_qtheta_ptheta + kl_qz_pz)

class SpikeSlab(RandomVariable, distribution_SpikeSlab):
    def __init__(self, *args, **kwargs):
        if 'beta' in kwargs:
            beta = kwargs['beta']
        else:
            beta = args[1]
        if 'value' not in kwargs:
            kwargs['value'] = tf.zeros_like(beta)
        RandomVariable.__init__(self, *args, **kwargs)

class distribution_GeneticValue(Distribution):
    """Represent the local reparamterization eta = x * theta as proposed in Kingma, Salimans, & Welling 2015.

    We use this to avoid sampling from the spike-and-slab prior. We have:

    E_q[x theta] = x * (alpha .* beta)
    V_q[x theta] = x^2 * (alpha ./ gamma + alpha .* (1 - alpha) .* beta^2)

    Then, we can just sample from a Gaussian and scale appropriately to
    evaluate E_q[ln p(x | ...)].

    """
    def __init__(self, x, theta, validate_args=False, allow_nan_stats=True,
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

class GeneticValue(RandomVariable, distribution_GeneticValue):
    def __init__(self, *args, **kwargs):
        RandomVariable.__init__(self, *args, **kwargs)
