import edward as ed
import tensorflow as tf

from edward.models import RandomVariable
from tensorflow.contrib.distributions import (
    Distribution,
    FULLY_REPARAMETERIZED,
    RegisterKL,
)

class distribution_LocalReparameterization(Distribution):
    """Represent local reparameterizations (Kingma, Salimans, & Welling 2015).

    This reparameterization avoids sampling from q(z) when estimating \nabla
    E_q[ln p] when this is difficult. 

    For example, we might have ln p = N(X w, I), where w is high dimensional or
    has some difficult prior on it. In this case, rather than sampling w
    directly, we can sample X w from the reparameterized distribution q(X w)
    which could be easier.

    LocalReparameterization needs to appear in the latent_vars of the inference:

    eta = LocalReparameterization(...)
    q_eta = LocalReparameterization(...)

    inference = ed.ReparameterizationKLKLqp(
        latent_vars={...,
                     eta: q_eta
        },
        ...
    )

    The idea of this implementation is to wrap around an
    ed.models.RandomVariable to support sampling, and provide a dummy KL
    divergence implementation (just return 0). In this way, the extra node only
    changes the data part of the built objective function.

    Arguments:

    dist - A subclass of ed.models.RandomVariable
    *args - Arguments to dist

    """
    def __init__(self, dist, *args, validate_args=False, allow_nan_stats=False,
                 name='LocalReparameterization'):
        self._dist = dist

        super(distribution_LocalReparameterization, self).__init__(
            allow_nan_stats=allow_nan_stats,
            dtype=self._dist.dtype,
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

@RegisterKL(distribution_LocalReparameterization, distribution_LocalReparameterization)
def kl_geneticvalue(q, p, name=None):
    """Dummy KL divergence between genetic values

    """
    return 0.0

class LocalReparameterization(RandomVariable, distribution_LocalReparameterization):
    def __init__(self, *args, **kwargs):
        RandomVariable.__init__(self, *args, **kwargs)
