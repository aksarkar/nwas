import matplotlib.pyplot as plt
import tensorflow as tf

from edward.models import *
from nwas.models import *

class MediatorModel():
    """Fit the regression model:

    y = x w + e
    w_{jk} ~ pi N(0, tau^-1) + (1 - pi) \delta

    y - responses (n x m)
    x - predictors (n x p)
    w - sparse effects (p x m)

    """
    def __init__(self, x, y):
        self.graph = tf.Graph()
        self.inference = None

        # We need the shapes of the data here, so they can't go in fit() like
        # sklearn
        self.x_ = x
        self.y_ = y
        n, p = self.x_.shape
        _, m = self.y_.shape

        # We don't fit an intercept, so center the data here
        self.x_ -= self.x_.mean(axis=0)
        self.y_ -= self.y_.mean(axis=0)
        
        with self.graph.as_default:
            self.x = tf.placeholder(tf.float32)
            self.logodds = Normal(loc=tf.constant(-10.0), scale=tf.ones(1))
            self.scale = Normal(loc=tf.zeros(1), scale=tf.ones(1))
            self.w = SpikeSlab(logodds=self.logodds,
                               loc=tf.zeros([p, m]),
                               scale=scale)
            # This is a dummy which gets swapped out in inference
            self.eta = LocalReparameterization(ed.models.Normal(tf.matmul(self.x, w), 1.0))
            self.y = NormalWithSoftplusScale(loc=eta, scale=tf.Variable(tf.zeros([1])))

            self.q_logodds = Normal(loc=tf.Variable(tf.random_normal([1])),
                                    scale=tf.Variable(tf.random_normal([1])))
            self.q_scale = Normal(loc=tf.Variable(tf.random_normal([1])),
                                  scale=tf.Variable(tf.random_normal([1])))
            self.q_w = SpikeSlab(logodds=tf.Variable(tf.zeros([p, m])),
                                 loc=tf.Variable(tf.zeros([p, m])),
                                 scale=tf.Variable(tf.zeros([p, m])))
            self.q_eta = LocalReparameterization(
                ed.models.Normal(loc=tf.matmul(self.x, self.q_w.mean()),
                scale=tf.sqrt(tf.matmul(tf.square(self.x), self.q_w.variance()))))

    def fit(self, **kwargs):
        my_kwargs = {'n_iter': 2000,
                     'optimizer': 'rmsprop', 'n_samples': 10}
        my_kwargs.update(kwargs)

        with self.graph.as_default():
            self.inference = ed.ReparameterizationKLKLqp(
                latent_vars={
                    self.logodds: self.q_logodds,
                    self.scale: self.q_scale,
                    self.w: self.q_w,
                    self.eta: self.q_eta,
                },
                data={
                    self.x: self.x_,
                    self.y: self.y_,
                })
            self.inference.run(**my_kwargs)
        return self

    def _predict(self):
        return [self.q_eta.mean(), self.q_eta.variance()]

    def predict(self, x):
        """Return [E[x w], V[x w]]"""
        if self.inference is None:
            raise ValueError
        with self.graph.as_default():
            return ed.get_session().run(self._predict(), {self.x: x})

    def correlation_score(y_true, y_pred):
        """Return the coefficient of determination"""
        R = 1 - (tf.reduce_sum(tf.square(y_true - y_pred)) /
                 tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true))))
        return R

    def score(self, x, y):
        if self.inference is None:
            raise ValueError
        with self.graph.as_default():
            y_hat, y_var = self.predict(x)
            return ed.get_session().run(correlation_score(y, y_hat))

class PhenotypeModel():
    """Fit the GLM

    E[y] = g v + x u + e

    g ~ N(E[g], V[g])
    v ~ pi_v N(0, \tau_v^-1) + (1 - pi_v) \delta
    u ~ pi_u N(0, \tau_v^-1) + (1 - pi_u) \delta

    y - phenotypes (n x 1)
    x - centered genotypes (n x p)
    g_mean - mean imputed gene expression (n x m)
    g_var - variance of imputed gene expression (n x m)
    v - mediated effect size (m x 1)
    u - unmediated effect size (p x 1)
    
    """
    def __init__(self, x, g_mean, g_var, y):
        self.graph = tf.Graph()
        # We only use g_mean and g_var to get V[g v] here, so we don't need to
        # keep pointers to them
        self.x_ = x
        self.y_ = y
        n, p = self.x_.shape
        _, m = g_mean.shape
        self.inference = None

        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32)

            # Unmediated effects
            self.logodds_u = Normal(loc=tf.constant(-10.0), scale=tf.ones(1))
            self.scale_u = Normal(loc=tf.zeros(1), scale=tf.ones(1))
            self.u = SpikeSlab(logodds=self.logodds_u, loc=tf.zeros([p, 1]), scale=self.scale_u)
            self.eta0 = LocalReparameterization(Normal(tf.matmul(self.x1, self.u), 1.0))

            # Mediated effects
            self.logodds_v = Normal(loc=tf.constant(-10.0), scale=tf.ones(1))
            self.scale_v = Normal(loc=tf.zeros(1), scale=tf.ones(1))
            self.v = SpikeSlab(logodds=self.logodds_v, loc=tf.zeros([m, 1]), scale=self.scale_v)
            self.eta1 = LocalReparameterization(Normal(tf.matmul(tf.matmul(x1, w), v), 1.0))

            self.y = NormalWithSoftplusScale(loc=self.eta0 + self.eta1, scale=tf.Variable(0.0))

            self.q_logodds_u = Normal(loc=tf.Variable(tf.random_normal([1])),
                                      scale=tf.Variable(tf.random_normal([1])))
            self.q_scale_u = Normal(loc=tf.Variable(tf.random_normal([1])),
                                    scale=tf.Variable(tf.random_normal([1])))
            self.q_u = SpikeSlab(logodds=tf.Variable(tf.zeros([p, 1])),
                                 loc=tf.Variable(tf.randop_norpal([p, 1])),
                                 scale=tf.Variable(tf.zeros([p, 1])))

            self.q_logodds_v = Normal(loc=tf.Variable(tf.random_normal([1])),
                                      scale=tf.Variable(tf.random_normal([1])))
            self.q_scale_v = Normal(loc=tf.Variable(tf.random_normal([1])),
                                    scale=tf.Variable(tf.random_normal([1])))
            self.q_v = SpikeSlab(logodds=tf.Variable(tf.zeros([m, 1])),
                                 loc=tf.Variable(tf.random_normal([m, 1])),
                                 scale=tf.Variable(tf.zeros([m, 1])))

            self.q_eta0 = LocalReparameterization(
                Normal(loc=tf.matmul(self.x, self.q_u.mean()),
                       scale=tf.matmul(tf.square(self.x), self.q_u.variance())))

            # Here, propagate the uncertainty V_q[g] from the first-stage
            # regression using Brown 1977.
            #
            #   E_q[g_i v] = E_q[g_i] E_q[v]
            #
            #   V_q[g_i v] = E_q[g_i] \diag(V_q[v]) E_q[g_i]' +
            #                E_q[v]' \diag(V_q[g_i]) E_q[v] +
            #                V_q[g_i]' V_q[v]

            var = (tf.reduce_sum(tf.square(g_mean) *
                                 tf.transpose(q_v.variance()), axis=1, keep_dims=True) +
                   tf.reduce_sum(tf.transpose(tf.square(q_v.mean())) *
                                 g_var, axis=1, keep_dims=True) +
                   tf.matmul(g_var, q_v.variance()))
            self.q_eta1 = LocalReparameterization(
                Normal(loc=tf.matmul(g_mean, q_v.mean()),
                       scale=tf.sqrt(var)))

    def fit(self):
        with self.graph.as_default():
            self.inference = ed.ReparameterizationKLKLqp(
                latent_vars={
                    self.logodds_u: self.q_logodds_u,
                    self.logodds_v: self.q_logodds_v,
                    self.scale_u: self.q_scale_u,
                    self.scale_v: self.q_scale_v,
                    self.u: self.q_u,
                    self.v: self.q_v,
                    self.eta0: self.q_eta0,
                    self.eta1: self.q_eta1,
                },
                data={
                    self.x: self.x_,
                    self.y: self.y_,
                })
            self.inference.run(n_iter=2000, optimizer='rmsprop')
