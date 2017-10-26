import edward as ed
import tensorflow as tf

from edward.models import RandomVariable
from tensorflow.contrib.distributions import (
    BernoulliWithSigmoidProbs,
    Distribution,
    FULLY_REPARAMETERIZED,
    NormalWithSoftplusScale,
    RegisterKL,
    kl_divergence,
)
from tensorflow.python.ops import (
    array_ops,
    check_ops,
)

class distribution_SpikeSlab(Distribution):
    """Spike-and-slab prior (point-normal mixture; Mitchell & Beauchamp 1988).

    The prior is parameterized by logodds, loc, and scale. logodds is
    transparently sigmoid-transformed. Similarly, scale is transparently
    softplus-transformed.

    p(theta_j | logodds, loc, scale) =
        sigmoid(logodds) * Normal(loc, softplus(scale)) +
        (1 - sigmoid(logodds)) * PointMass(theta_j)

    We want to use the variational approximation to efficiently estimate the
    posterior w.r.t. this prior. The conjugate mean-field variational
    approximation admits an analytical KL (Carbonetto & Stephens 2012).

    This implementation does not support sampling (does not scale to high
    dimensions). Instead, add a local reparameterization (see
    nwas.model.GeneticValue) which admits efficient sampling to the model (see
    Kingma, Salimans, & Welling 2015).

    """
    def __init__(self, logodds, loc, scale, validate_args=False,
                 allow_nan_stats=False, name='SpikeSlab'):
        # This is needed so that tf.python.ops.distributions.Distribution saves
        # positional arguments. Otherwise, ed.util.random_variables.copy
        # blows up
        parameters = locals()

        # c.f. ed.models.empirical
        with tf.name_scope(name, values=[logodds, loc, scale]):
            with tf.control_dependencies([]):
                self._logodds = array_ops.identity(logodds, name="logodds")
                self._loc = array_ops.identity(loc, name="loc")
                self._scale = array_ops.identity(scale, name="scale")
                check_ops.assert_same_float_dtype([self._logodds, self._loc, self._scale])

        # c.f. tf.python.ops.distributions.normal.Normal.__init__
        super(distribution_SpikeSlab, self).__init__(
            allow_nan_stats=allow_nan_stats,
            dtype=self._loc.dtype,
            graph_parents=[self._logodds, self._loc, self._scale],
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
            return tf.sigmoid(self._logodds)

    def _mean(self):
        return tf.sigmoid(self._logodds) * self._loc

    def _stddev(self):
        return tf.sqrt(self.variance())

    def _variance(self):
        p = tf.sigmoid(self._logodds)
        return (p * tf.nn.softplus(self._scale) +
                p * (1 - p) * tf.square(self._loc))

@RegisterKL(distribution_SpikeSlab, distribution_SpikeSlab)
def kl_spikeslab(q, p, name=None):
    """KL divergence between point-normal mixture and conjugate mean-field variational approximation

    An analytic form was given without derivation in Carbonetto & Stephens
    2012. It can be derived using Rasmussen and Williams 2006, Eqs. A.22, A.23

    """
    kl_qtheta_ptheta = tf.sigmoid(q._logodds) * kl_divergence(
        NormalWithSoftplusScale(loc=q._loc, scale=q._scale),
        NormalWithSoftplusScale(loc=p._loc, scale=p._scale)
    )
    kl_qz_pz = tf.sigmoid(q._logodds) * kl_divergence(
        BernoulliWithSigmoidProbs(q._logodds),
        BernoulliWithSigmoidProbs(p._logodds)
    )
    return tf.reduce_sum(kl_qtheta_ptheta + kl_qz_pz)

class SpikeSlab(RandomVariable, distribution_SpikeSlab):
    def __init__(self, *args, **kwargs):
        if 'loc' in kwargs:
            loc = kwargs['loc']
        else:
            loc = args[1]
        if 'value' not in kwargs:
            kwargs['value'] = tf.zeros_like(loc)
        RandomVariable.__init__(self, *args, **kwargs)
