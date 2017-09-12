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
from tensorflow.python.ops import array_ops

class SpikeSlab(RandomVariable, Distribution):
    """Spike-and-slab prior (point-normal mixture; Mitchell & Beauchamp 1988).

    This is a compound distribution of indicator variables z and values theta.

    p(theta_j, z_j | pi, tau) = pi N(theta_j; 0, tau^{-1}) +
                                   (1 - pi) delta(theta_j)

    We want to use the variational approximation to efficiently estimate the
    posterior w.r.t. this prior. The conjugate mean-field variational
    approximation admits an analytical KL (Carbonetto & Stephens, Bayesian Anal
    2012).

    q(theta_j, z_j | alpha, beta, gamma) = alpha_j N(theta_j; beta_j, gamma_j^{-1}) + 
                                           (1 - \alpha_j) delta(\theta_j)

    This distribution does not support sampling (does not scale to high
    dimensions). Instead, use the local reparameterization trick (see
    nwas.model.GeneticValue; Kingma, Salimans, & Welling arXiv 2015).

    """
    def __init__(self, alpha, beta, gamma, validate_args=False,
                 allow_nan_stats=True, name='SpikeSlab'):
        self._alpha = array_ops.identity(alpha, name="alpha")
        self._beta = array_ops.identity(beta, name="beta")
        self._gamma = array_ops.identity(gamma, name="gamma")

        super(SpikeSlab, self).__init__(
            allow_nan_stats=allow_nan_stats,
            dtype=self._beta.dtype,
            graph_parents=[self._alpha, self._beta, self._gamma],
            name=name,
            reparameterization_type=FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            value=tf.zeros_like(self._beta),
        )

    def _log_prob(self, value):
        raise NotImplementedError("log_prob is not implemented")

    def _sample_n(self, n, seed=None):
        raise NotImplementedError("sample_n is not implemented")

    def _mean(self):
        return self._alpha * self._beta

    def _stddev(self):
        return tf.sqrt(self.variance())

    def _variance(self):
        return (self._alpha / self._gamma +
                self._alpha * (1 - self._alpha) * tf.square(self._beta))

@RegisterKL(SpikeSlab, SpikeSlab)
def kl_spikeslab(q, p, name=None):
    """KL divergence between point-normal mixture and conjugate mean-field variational approximation

    An analytic form was given without derivation in Carbonetto & Stephens,
    Bayesian Anal 2012. It can be derived using Rasmussen and Williams, 2006,
    Eqs. A.22, A.23

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

class GeneticValue(RandomVariable, Distribution):
    def __init__(self, x, theta, validate_args=False, allow_nan_stats=True,
                 name='GeneticValue'):
        self._x = x
        self._theta = theta

        self._dist = edward.models.Normal(
            loc=tf.matmul(self._x, self._theta.mean()),
            scale=tf.sqrt(tf.matmul(tf.square(self._x),
                                    self._theta.variance()))
        )

        super(GeneticValue, self).__init__(
            allow_nan_stats=allow_nan_stats,
            dtype=self._theta.dtype,
            name=name,
            reparameterization_type=FULLY_REPARAMETERIZED,
            validate_args=validate_args,
        )

    def _log_prob(self, value):
        return self._dist.log_prob(value)

    def _sample_n(self, n, seed=None):
        return self._dist.sample_n(n, seed)
