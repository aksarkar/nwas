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

class SpikeSlab(RandomVariable, Distribution):
    """Spike-and-slab prior (point-normal mixture; George & Beauchamp XXX).

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
        parameters=locals()

        with tf.name_scope(name, values=[alpha, beta, gamma]):
            with tf.control_dependencies([]):
                self._alpha = alpha
                self._beta = beta
                self._gamma = gamma
                self._shape = tf.shape(beta)
                # if tf.shape(self._alpha) and tf.shape(self._alpha) != self._shape:
                #     raise ValueError('Shape mismatch: expected {}, got {}'.format(
                #         self._shape, tf.shape(self._alpha.shape)))
                # if tf.shape(self._gamma) and tf.shape(self._gamma) != self._shape:
                #     raise ValueError('Shape mismatch: expected {}, got {}'.format(
                #         self._shape, tf.shape(self._gamma.shape)))

        super(SpikeSlab, self).__init__(
            allow_nan_stats=allow_nan_stats,
            dtype=self._beta.dtype,
            graph_parents=[self._alpha, self._beta, self._gamma],
            name=name,
            reparameterization_type=FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            value=tf.zeros_like(self._beta)
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
        Normal(loc=p._beta, scale=tf.reciprocal(p._gamma)))
    kl_qz_pz = q._alpha * kl_divergence(
        BernoulliWithSigmoidProbs(q._alpha),
        BernoulliWithSigmoidProbs(p._alpha)
    )
    return tf.reduce_sum(kl_qtheta_ptheta + kl_qz_pz)

class GeneticValue(RandomVariable, Distribution):
    def __init__(self, x, theta, validate_args=False, allow_nan_stats=True,
                 name='GeneticValue'):
        super(GeneticValue, self).__init__(validate_args=validate_args,
                                           allow_nan_stats=allow_nan_stats,)
        self.x = x
        self.theta = theta
        self.dist = tf.contrib.distributions.Normal(loc=tf.matmul(x, theta),
                                                    scale=theta.stddev())

    def _log_prob(self, value):
        self.dist.log_prob(value)

    def _sample_n(self, n, seed=None):
        self.dist.sample(n, seed)
